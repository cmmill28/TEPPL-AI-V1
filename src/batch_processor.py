"""
Enhanced Batch Processor with Checkpoint/Resume System - COMPLETE VERSION
Fully integrated with all components: document processing, image processing, and vector storage
"""

import os
import warnings
import sys
import json
import logging
import signal
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import hashlib
import base64
from urllib.parse import urljoin, urlparse
from io import BytesIO

# Set environment variables BEFORE any ChromaDB imports
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY_IMPL"] = "chromadb.telemetry.NoopTelemetry"

# Suppress telemetry warnings
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*posthog.*")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Import local components
sys.path.append('src')

# Import processors
try:
    from document_processor import EnhancedDocumentProcessor
    PDF_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"âŒ PDF processor not available: {e}")
    PDF_PROCESSOR_AVAILABLE = False

try:
    from chroma_multimodal_store import EnhancedMultimodalVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    print(f"âŒ Vector store not available: {e}")
    VECTOR_STORE_AVAILABLE = False

try:
    from image_processor import TEPPLImageProcessor  # âœ… Use this instead
    IMAGE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    try:
        # Try alternative import path
        import sys
        sys.path.append('.')
        from image_processor import TEPPLImageProcessor
        IMAGE_PROCESSOR_AVAILABLE = True
    except ImportError as e2:
        print(f"âŒ Image processor not available: {e2}")
        IMAGE_PROCESSOR_AVAILABLE = False

# Multi-format processing imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    WEB_PROCESSING_AVAILABLE = True
except ImportError:
    WEB_PROCESSING_AVAILABLE = False

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

# Check for additional libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("âš ï¸ Requests not available. External image download disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("âš ï¸ PIL not available. Image analysis disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teppl_batch_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BatchProgressTracker:
    """
    Comprehensive progress tracking system for batch processing
    Handles checkpoints, resume functionality, and graceful shutdown
    """
    
    def __init__(self, progress_file: str = "batch_progress.json"):
        self.progress_file = Path(progress_file)
        self.lock = threading.Lock()
        
        # Progress data structure
        self.data = {
            'session_id': None,
            'started_at': None,
            'last_checkpoint': None,
            'last_resume': None,
            'processed_files': set(),
            'failed_files': {},  # file_path -> error_message
            'skipped_files': set(),
            'processing_stats': {
                'total_files': 0,
                'processed_successfully': 0,
                'processing_errors': 0,
                'files_skipped': 0,
                'average_processing_time': 0.0,
                'estimated_completion': None
            },
            'batch_info': {
                'current_batch': 0,
                'total_batches': 0,
                'batch_size': 50,
                'files_in_current_batch': 0
            },
            'system_info': {
                'multiformat_enabled': True,
                'supported_formats': [],
                'documents_path': '',
                'storage_path': ''
            }
        }
        
        self.load_progress()
        logger.info(f"ðŸ“Š Progress tracker initialized: {progress_file}")

    def load_progress(self):
        """Load existing progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    loaded_data = json.load(f)
                    
                # Convert processed_files back to set
                if 'processed_files' in loaded_data:
                    loaded_data['processed_files'] = set(loaded_data['processed_files'])
                else:
                    loaded_data['processed_files'] = set()
                    
                if 'skipped_files' in loaded_data:
                    loaded_data['skipped_files'] = set(loaded_data['skipped_files'])
                else:
                    loaded_data['skipped_files'] = set()
                
                # Merge with default structure
                self._merge_data(loaded_data)
                logger.info(f"âœ… Loaded progress: {len(self.data['processed_files'])} files processed")
                
            except Exception as e:
                logger.warning(f"âš ï¸ Could not load progress file: {e}. Starting fresh.")
                self._initialize_fresh_session()
        else:
            self._initialize_fresh_session()

    def _merge_data(self, loaded_data):
        """Safely merge loaded data with default structure"""
        for key, value in loaded_data.items():
            if key in self.data:
                if isinstance(self.data[key], dict) and isinstance(value, dict):
                    self.data[key].update(value)
                else:
                    self.data[key] = value

    def _initialize_fresh_session(self):
        """Initialize a fresh processing session"""
        self.data['session_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"ðŸ†• Starting fresh session: {self.data['session_id']}")

    def save_checkpoint(self):
        """Save current progress to file"""
        with self.lock:
            try:
                # Prepare data for JSON serialization
                save_data = self.data.copy()
                save_data['processed_files'] = list(self.data['processed_files'])
                save_data['skipped_files'] = list(self.data['skipped_files'])
                save_data['last_checkpoint'] = datetime.now().isoformat()
                
                # Atomic write
                temp_file = self.progress_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
                temp_file.replace(self.progress_file)
                
            except Exception as e:
                logger.error(f"âŒ Failed to save checkpoint: {e}")

    def start_session(self, total_files: int, system_info: Dict):
        """Start a new processing session"""
        if self.data['started_at'] is None:
            self.data['started_at'] = datetime.now().isoformat()
            
        self.data['last_resume'] = datetime.now().isoformat()
        self.data['processing_stats']['total_files'] = total_files
        self.data['system_info'].update(system_info)
        
        # Calculate batches
        batch_size = self.data['batch_info']['batch_size']
        self.data['batch_info']['total_batches'] = (total_files + batch_size - 1) // batch_size
        
        logger.info(f"ðŸš€ Session started/resumed - {total_files} total files, {len(self.data['processed_files'])} already processed")
        self.save_checkpoint()

    def mark_file_processed(self, file_path: str, processing_time: float = 0.0):
        """Mark a file as successfully processed"""
        file_str = str(file_path)
        
        with self.lock:
            if file_str not in self.data['processed_files']:
                self.data['processed_files'].add(file_str)
                self.data['processing_stats']['processed_successfully'] += 1
                
                # Update average processing time
                current_avg = self.data['processing_stats']['average_processing_time']
                processed_count = self.data['processing_stats']['processed_successfully']
                
                if processed_count > 1:
                    new_avg = ((current_avg * (processed_count - 1)) + processing_time) / processed_count
                else:
                    new_avg = processing_time
                    
                self.data['processing_stats']['average_processing_time'] = new_avg
                
                # Update estimated completion
                remaining_files = (self.data['processing_stats']['total_files'] - 
                                 len(self.data['processed_files']))
                                 
                if remaining_files > 0 and new_avg > 0:
                    estimated_seconds = remaining_files * new_avg
                    completion_time = datetime.now() + timedelta(seconds=estimated_seconds)
                    self.data['processing_stats']['estimated_completion'] = completion_time.isoformat()

    def mark_file_failed(self, file_path: str, error_message: str):
        """Mark a file as failed to process"""
        file_str = str(file_path)
        
        with self.lock:
            self.data['failed_files'][file_str] = error_message
            self.data['processing_stats']['processing_errors'] += 1

    def mark_file_skipped(self, file_path: str, reason: str = "Already processed"):
        """Mark a file as skipped"""
        file_str = str(file_path)
        
        with self.lock:
            self.data['skipped_files'].add(file_str)
            self.data['processing_stats']['files_skipped'] += 1

    def is_file_processed(self, file_path: str) -> bool:
        """Check if a file has already been processed"""
        return str(file_path) in self.data['processed_files']

    def update_batch_info(self, current_batch: int, files_in_batch: int):
        """Update current batch information"""
        with self.lock:
            self.data['batch_info']['current_batch'] = current_batch
            self.data['batch_info']['files_in_current_batch'] = files_in_batch

    def get_remaining_files(self, all_files: List[Path]) -> List[Path]:
        """Get list of files that still need processing"""
        return [f for f in all_files if not self.is_file_processed(f)]

    def get_progress_summary(self) -> Dict:
        """Get comprehensive progress summary"""
        with self.lock:
            total = self.data['processing_stats']['total_files']
            processed = len(self.data['processed_files'])
            progress_percentage = (processed / total * 100) if total > 0 else 0
            
            return {
                'session_id': self.data['session_id'],
                'progress_percentage': round(progress_percentage, 2),
                'files_processed': processed,
                'total_files': total,
                'files_remaining': total - processed,
                'files_failed': len(self.data['failed_files']),
                'files_skipped': len(self.data['skipped_files']),
                'average_processing_time': self.data['processing_stats']['average_processing_time'],
                'estimated_completion': self.data['processing_stats']['estimated_completion'],
                'current_batch': self.data['batch_info']['current_batch'],
                'total_batches': self.data['batch_info']['total_batches'],
                'started_at': self.data['started_at'],
                'last_resume': self.data['last_resume'],
                'last_checkpoint': self.data['last_checkpoint']
            }

    def print_progress_report(self):
        """Print a detailed progress report"""
        summary = self.get_progress_summary()
        
        print("\n" + "="*60)
        print("ðŸ“Š BATCH PROCESSING PROGRESS REPORT")
        print("="*60)
        print(f"ðŸ†” Session ID: {summary['session_id']}")
        print(f"ðŸ“ˆ Progress: {summary['progress_percentage']:.1f}%")
        print(f"âœ… Processed: {summary['files_processed']:,} / {summary['total_files']:,}")
        print(f"â³ Remaining: {summary['files_remaining']:,}")
        print(f"âŒ Failed: {summary['files_failed']}")
        print(f"â­ï¸ Skipped: {summary['files_skipped']}")
        
        if summary['average_processing_time'] > 0:
            print(f"â±ï¸ Avg Time: {summary['average_processing_time']:.2f}s per file")
            
        if summary['estimated_completion']:
            completion = datetime.fromisoformat(summary['estimated_completion'])
            print(f"ðŸŽ¯ Est. Completion: {completion.strftime('%Y-%m-%d %H:%M:%S')}")
            
        print(f"ðŸ“¦ Current Batch: {summary['current_batch']} / {summary['total_batches']}")
        print("="*60)


class GracefulShutdownHandler:
    """Handle graceful shutdown on SIGINT/SIGTERM"""
    
    def __init__(self, progress_tracker: BatchProgressTracker):
        self.progress_tracker = progress_tracker
        self.shutdown_requested = False
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if not self.shutdown_requested:
            self.shutdown_requested = True
            logger.info(f"\nðŸ›‘ Shutdown signal received ({signum}). Finishing current batch and saving progress...")
            print(f"\nðŸ›‘ Graceful shutdown initiated. Please wait for current batch to complete...")
            
            # Save final checkpoint
            self.progress_tracker.save_checkpoint()

    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested"""
        return self.shutdown_requested


class MultiFormatProcessor:
    """Enhanced processor supporting all TEPPL document formats with image processing"""
    
    def __init__(self, documents_path: str = "./documents"):
        self.documents_path = Path(documents_path)
        
        # Initialize PDF processor if available
        if PDF_PROCESSOR_AVAILABLE:
            self.pdf_processor = EnhancedDocumentProcessor(str(documents_path))
        else:
            self.pdf_processor = None
            
        # Initialize image processor if available
        if IMAGE_PROCESSOR_AVAILABLE:
            self.image_processor = TEPPLImageProcessor()
        else:
            self.image_processor = None
        
        # Format capabilities
        self.format_capabilities = {
            'pdf': {'text': True, 'images': True, 'metadata': True, 'processor': 'pdf'},
            'aspx': {'text': True, 'images': True, 'metadata': True, 'processor': 'web'},  # âœ… Changed to True
            'html': {'text': True, 'images': True, 'metadata': True, 'processor': 'web'},  # âœ… Changed to True
            'xhtml': {'text': True, 'images': True, 'metadata': True, 'processor': 'web'}, # âœ… Changed to True
            'docx': {'text': True, 'images': False, 'metadata': True, 'processor': 'office'},
            'doc': {'text': True, 'images': False, 'metadata': True, 'processor': 'office'},
            'dotx': {'text': True, 'images': False, 'metadata': True, 'processor': 'office'},
            'xlsx': {'text': True, 'images': False, 'metadata': True, 'processor': 'excel'},
            'xls': {'text': True, 'images': False, 'metadata': True, 'processor': 'excel'},
            'xlsm': {'text': True, 'images': False, 'metadata': True, 'processor': 'excel'},
            'xlsb': {'text': True, 'images': False, 'metadata': True, 'processor': 'excel'},
            'json': {'text': True, 'images': False, 'metadata': True, 'processor': 'structured'},
            'xml': {'text': True, 'images': False, 'metadata': True, 'processor': 'structured'},
            'txt': {'text': True, 'images': False, 'metadata': True, 'processor': 'text'},
            'zip': {'text': False, 'images': False, 'metadata': True, 'processor': 'archive'},
            'url': {'text': False, 'images': False, 'metadata': True, 'processor': 'reference'}
        }
        
        logger.info("ðŸŽ¯ Multi-Format Processor initialized - supports 17+ formats with enhanced image processing")

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process any supported document format with enhanced image processing"""
        file_ext = file_path.suffix.lower().lstrip('.')
        
        if file_ext not in self.format_capabilities:
            return self._create_unsupported_result(file_path, file_ext)
        
        processor_type = self.format_capabilities[file_ext]['processor']
        
        try:
            if processor_type == 'pdf':
                return self._process_pdf_enhanced(file_path)
            elif processor_type == 'web':
                return self._process_web_document(file_path)
            elif processor_type == 'office':
                return self._process_office_document(file_path)
            elif processor_type == 'excel':
                return self._process_excel_document(file_path)
            elif processor_type == 'structured':
                return self._process_structured_document(file_path)
            elif processor_type == 'text':
                return self._process_text_document(file_path)
            elif processor_type == 'archive':
                return self._process_archive(file_path)
            elif processor_type == 'reference':
                return self._process_reference(file_path)
            else:
                return self._create_error_result(file_path, f"Unknown processor: {processor_type}")
                
        except Exception as e:
            logger.error(f"âŒ Error processing {file_path.name}: {e}")
            return self._create_error_result(file_path, str(e))

    def _process_pdf_enhanced(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF with enhanced image processing"""
        if not self.pdf_processor:
            return self._create_error_result(file_path, "PDF processor not available")
        
        try:
            # Use the existing PDF processor for text and basic images
            base_result = self.pdf_processor.process_pdf(str(file_path))
            
            # If image processor is available, enhance with advanced image processing
            if self.image_processor and base_result.get('metadata'):
                try:
                    enhanced_images = self.image_processor.process_pdf_images_complete(
                        str(file_path), 
                        base_result['metadata']
                    )
                    
                    # Replace basic images with enhanced ones
                    if enhanced_images.get('images'):
                        base_result['images'] = enhanced_images['images']
                        base_result['metadata']['enhanced_image_processing'] = True
                        base_result['metadata']['image_processing_stats'] = enhanced_images.get('filtering_stats', {})
                        
                        logger.info(f"âœ… Enhanced image processing: {len(enhanced_images['images'])} high-quality images")
                    
                except Exception as e:
                    logger.warning(f"Enhanced image processing failed for {file_path.name}: {e}")
                    # Keep the basic images from the original processor
            
            return base_result
            
        except Exception as e:
            return self._create_error_result(file_path, f"PDF processing error: {e}")

    def _process_web_document(self, file_path: Path) -> Dict[str, Any]:
        """Process ASPX, HTML, XHTML documents with image extraction"""
        if not WEB_PROCESSING_AVAILABLE:
            return self._create_error_result(file_path, "Web processing libraries not available")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse HTML content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text()
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text() if title else file_path.stem
            
            # Create text chunks
            chunks = self._create_text_chunks(clean_text, file_path, "web")
            
            # ðŸŽ¨ NEW: Extract images from HTML
            extracted_images = self._extract_web_images(soup, file_path, title_text)
            
            # Extract links and form data
            links = [a.get('href') for a in soup.find_all('a', href=True)]
            forms = [form.get('action') for form in soup.find_all('form', action=True)]
            
            return {
                "chunks": chunks,
                "images": extracted_images,  # âœ… Now includes web images
                "metadata": {
                    "source_type": "web",
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat(),
                    "document_id": self._generate_doc_id(file_path),
                    "title": title_text,
                    "content_length": len(clean_text),
                    "links_count": len(links),
                    "forms_count": len(forms),
                    "images_extracted": len(extracted_images),  # âœ… Track image count
                    "external_links": links[:10],
                    "has_forms": len(forms) > 0,
                    "has_images": len(extracted_images) > 0,  # âœ… Image flag
                    "encoding": "utf-8"
                }
            }
            
        except Exception as e:
            return self._create_error_result(file_path, f"Web processing error: {e}")

    def _extract_web_images(self, soup, file_path: Path, page_title: str) -> list:
        """Extract and process images from HTML/ASPX content"""
        images = []
        img_tags = soup.find_all('img')
        logger.info(f"ðŸ–¼ï¸ HTML image extraction enabled (found {len(img_tags)} img tags)")
        if not img_tags or not PIL_AVAILABLE:
            return images
        doc_id = self._generate_doc_id(file_path)
        base_url = f"file://{file_path.parent.absolute()}/"
        for img_index, img_tag in enumerate(img_tags):
            try:
                src = img_tag.get('src')
                if not src:
                    continue
                # Resolve relative URLs
                if src.startswith('http'):
                    image_url = src
                    is_external = True
                else:
                    if src.startswith('./') or src.startswith('../'):
                        resolved_path = (file_path.parent / src).resolve()
                    else:
                        resolved_path = file_path.parent / src
                    image_url = str(resolved_path)
                    is_external = False
                alt_text = img_tag.get('alt', '')
                title_attr = img_tag.get('title', '')
                context_text = self._extract_image_context_web(img_tag, soup)
                image_bytes, image_format = self._fetch_web_image(image_url, is_external)
                if not image_bytes:
                    continue
                try:
                    pil_image = Image.open(BytesIO(image_bytes))
                    width, height = pil_image.size
                    if width < 150 or height < 150:
                        logger.debug(f"â­ï¸ Skipping small image: {width}x{height} px")
                        continue
                except Exception as e:
                    logger.warning(f"Could not analyze image {src}: {e}")
                    continue
                image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                image_id = f"{doc_id}_web_img_{image_hash}"
                image_filename = f"{image_id}.{image_format}"
                image_path = getattr(self, 'output_dir', None)
                if image_path:
                    image_path = image_path / image_filename
                    try:
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                        saved_path = str(image_path)
                    except Exception as e:
                        logger.warning(f"Could not save image {image_id}: {e}")
                        saved_path = None
                else:
                    saved_path = None
                image_type = self._classify_web_image_type(alt_text, title_attr, context_text)
                image_doc = {
                    "id": image_id,
                    "type": image_type,
                    "page": img_index + 1,
                    "file_path": saved_path,
                    "image_bytes_b64": base64.b64encode(image_bytes).decode('utf-8') if len(image_bytes) < 1024*1024 else None,
                    "source_url": str(file_path),
                    "image_url": image_url,
                    "caption": f"{alt_text} {title_attr}".strip(),
                    "context": context_text,
                    "bbox": None,
                    "dimensions": {"width": width, "height": height},
                    "format": image_format,
                    "size_bytes": len(image_bytes),
                    "extracted_text": f"{alt_text} {title_attr} {context_text}",
                    "classification": {
                        "primary_category": image_type,
                        "confidence": 0.7,
                        "method": "web_heuristic"
                    },
                    "analysis": {
                        "aspect_ratio": width / height if height > 0 else 1.0,
                        "size_category": self._categorize_size_web(width, height),
                        "has_alt_text": bool(alt_text),
                        "context_available": bool(context_text)
                    },
                    "metadata": {
                        "document_id": doc_id,
                        "extraction_method": "web_html",
                        "is_external": is_external,
                        "content_hash": image_hash
                    }
                }
                images.append(image_doc)
            except Exception as e:
                logger.warning(f"Error processing web image {img_index}: {e}")
                continue
        logger.info(f"ðŸ–¼ï¸ Web image extraction completed: {len(img_tags)} found â†’ {len(images)} kept after filters")
        return images

    def _fetch_web_image(self, image_url: str, is_external: bool):
        """Fetch image bytes from URL or file path with security controls"""
        try:
            if is_external:
                if not REQUESTS_AVAILABLE:
                    logger.debug("âš ï¸ Requests not available. Cannot fetch external images.")
                    return None, 'unknown'
                parsed_url = urlparse(image_url)
                allowed_domains = ['ncdot.gov', 'connect.ncdot.gov']
                if not any(domain in parsed_url.netloc for domain in allowed_domains):
                    logger.debug(f"âš ï¸ Skipping external image from non-whitelisted domain: {parsed_url.netloc}")
                    return None, 'unknown'
                headers = {'User-Agent': 'TEPPL-Bot/1.0'}
                response = requests.get(
                    image_url,
                    headers=headers,
                    timeout=10,
                    stream=True
                )
                content_type = response.headers.get('content-type', '').lower()
                allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/tiff', 'image/svg+xml']
                if not any(allowed_type in content_type for allowed_type in allowed_types):
                    logger.debug(f"âš ï¸ Skipping image with disallowed content type: {content_type}")
                    return None, 'unknown'
                content_length = int(response.headers.get('content-length', 0))
                if content_length > 10 * 1024 * 1024:
                    logger.debug(f"âš ï¸ Skipping large image: {content_length / (1024*1024):.1f}MB")
                    return None, 'unknown'
                image_bytes = response.content
                image_format = content_type.split('/')[-1]
            else:
                image_path = Path(image_url)
                if not image_path.exists():
                    return None, 'unknown'
                if image_path.stat().st_size > 10 * 1024 * 1024:
                    logger.debug(f"âš ï¸ Skipping large local image: {image_path.stat().st_size / (1024*1024):.1f}MB")
                    return None, 'unknown'
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                image_format = image_path.suffix.lstrip('.').lower()
            return image_bytes, image_format
        except Exception as e:
            logger.debug(f"Could not fetch image {image_url}: {e}")
            return None, 'unknown'

    def _extract_image_context_web(self, img_tag, soup):
        """Extract surrounding context for web images"""
        context_parts = []
        figure = img_tag.find_parent('figure')
        if figure:
            figcaption = figure.find('figcaption')
            if figcaption:
                context_parts.append(figcaption.get_text().strip())
        parent = img_tag.parent
        if parent and parent.name not in ['html', 'body']:
            parent_text = parent.get_text().strip()
            if len(parent_text) < 500:
                context_parts.append(parent_text)
        for sibling in img_tag.previous_siblings:
            if hasattr(sibling, 'get_text'):
                sibling_text = sibling.get_text().strip()
                if sibling_text and len(sibling_text) < 200:
                    context_parts.append(sibling_text)
                    break
        return ' '.join(context_parts)[:500]

    def _classify_web_image_type(self, alt_text: str, title_text: str, context_text: str) -> str:
        """Classify web image type using heuristics"""
        all_text = f"{alt_text} {title_text} {context_text}".lower()
        if any(term in all_text for term in ['sign', 'regulatory', 'warning', 'stop', 'yield']):
            return 'traffic_signs'
        elif any(term in all_text for term in ['signal', 'light', 'intersection', 'timing']):
            return 'traffic_signals'
        elif any(term in all_text for term in ['marking', 'stripe', 'lane', 'crosswalk']):
            return 'pavement_markings'
        elif any(term in all_text for term in ['diagram', 'plan', 'drawing', 'schematic']):
            return 'technical_drawing'
        elif any(term in all_text for term in ['chart', 'graph', 'table', 'data']):
            return 'charts_graphs'
        elif any(term in all_text for term in ['map', 'aerial', 'satellite']):
            return 'map'
        else:
            return 'photo'

    def _categorize_size_web(self, width: int, height: int) -> str:
        """Categorize web image size"""
        area = width * height
        if area < 50000:
            return "small"
        elif area < 500000:
            return "medium"
        else:
            return "large"

    def _process_structured_document(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON, XML documents"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if file_path.suffix.lower() == '.json':
                return self._process_json_content(content, file_path)
            elif file_path.suffix.lower() == '.xml':
                return self._process_xml_content(content, file_path)
        except Exception as e:
            return self._create_error_result(file_path, f"Structured processing error: {e}")

    def _process_json_content(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Process JSON content"""
        try:
            data = json.loads(content)
            json_text = json.dumps(data, indent=2, ensure_ascii=False)
            keys_count = len(data) if isinstance(data, dict) else 0
            items_count = len(data) if isinstance(data, list) else 0
            chunks = [{
                "content": json_text,
                "metadata": {
                    "source": str(file_path),
                    "source_type": "json",
                    "document_id": self._generate_doc_id(file_path),
                    "chunk_id": f"{self._generate_doc_id(file_path)}_json",
                    "content_type": "structured_data",
                    "data_type": "dict" if isinstance(data, dict) else "list" if isinstance(data, list) else "other",
                    "keys_count": keys_count,
                    "items_count": items_count
                }
            }]
            return {
                "chunks": chunks,
                "images": [],
                "metadata": {
                    "source_type": "json",
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat(),
                    "document_id": self._generate_doc_id(file_path),
                    "data_structure": type(data).__name__,
                    "content_length": len(content)
                }
            }
        except json.JSONDecodeError as e:
            return self._create_error_result(file_path, f"Invalid JSON: {e}")

    def _process_xml_content(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Process XML content"""
        try:
            if XML_AVAILABLE:
                root = ET.fromstring(content)
                xml_text = f"XML Root: {root.tag}\n"
                xml_text += f"Attributes: {root.attrib}\n\n"
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        xml_text += f"{elem.tag}: {elem.text.strip()}\n"
                chunks = [{
                    "content": xml_text,
                    "metadata": {
                        "source": str(file_path),
                        "source_type": "xml",
                        "document_id": self._generate_doc_id(file_path),
                        "chunk_id": f"{self._generate_doc_id(file_path)}_xml",
                        "content_type": "structured_data",
                        "root_tag": root.tag,
                        "elements_count": len(list(root.iter()))
                    }
                }]
                return {
                    "chunks": chunks,
                    "images": [],
                    "metadata": {
                        "source_type": "xml",
                        "file_path": str(file_path),
                        "processed_at": datetime.now().isoformat(),
                        "document_id": self._generate_doc_id(file_path),
                        "root_element": root.tag,
                        "total_elements": len(list(root.iter()))
                    }
                }
            else:
                return self._process_text_document(file_path)
        except ET.ParseError as e:
            return self._create_error_result(file_path, f"Invalid XML: {e}")

    def _process_excel_document(self, file_path: Path) -> Dict[str, Any]:
        """Process Excel files (XLSX, XLS, XLSM, XLSB)"""
        if not PANDAS_AVAILABLE:
            return self._create_error_result(file_path, "Pandas library not available")
        try:
            excel_data = pd.read_excel(file_path, sheet_name=None)
            chunks = []
            total_rows = 0
            for sheet_name, df in excel_data.items():
                if df.empty:
                    continue
                total_rows += len(df)
                sheet_text = f"Sheet: {sheet_name}\n"
                sheet_text += f"Columns: {', '.join(df.columns)}\n\n"
                for idx, row in df.head(50).iterrows():
                    row_text = ' | '.join(str(value) for value in row.values if pd.notna(value))
                    if row_text.strip():
                        sheet_text += f"Row {idx + 1}: {row_text}\n"
                chunk = {
                    "content": sheet_text,
                    "metadata": {
                        "source": str(file_path),
                        "sheet_name": sheet_name,
                        "source_type": "excel",
                        "document_id": self._generate_doc_id(file_path),
                        "chunk_id": f"{self._generate_doc_id(file_path)}_sheet_{sheet_name}",
                        "rows_count": len(df),
                        "columns_count": len(df.columns),
                        "content_type": "structured_data"
                    }
                }
                chunks.append(chunk)
            return {
                "chunks": chunks,
                "images": [],
                "metadata": {
                    "source_type": "excel",
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat(),
                    "document_id": self._generate_doc_id(file_path),
                    "sheets_count": len(excel_data),
                    "total_rows": total_rows,
                    "sheet_names": list(excel_data.keys())
                }
            }
        except Exception as e:
            return self._create_error_result(file_path, f"Excel processing error: {e}")

    def _process_office_document(self, file_path: Path) -> Dict[str, Any]:
        """Process DOCX, DOC, DOTX documents"""
        if not PYTHON_DOCX_AVAILABLE:
            return self._create_error_result(file_path, "Python-docx library not available")
        try:
            if file_path.suffix.lower() in ['.docx', '.dotx']:
                doc = DocxDocument(file_path)
                paragraphs = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        paragraphs.append(para.text.strip())
                text_content = '\n\n'.join(paragraphs)
                props = doc.core_properties
                chunks = self._create_text_chunks(text_content, file_path, "office")
                return {
                    "chunks": chunks,
                    "images": [],
                    "metadata": {
                        "source_type": "office",
                        "file_path": str(file_path),
                        "processed_at": datetime.now().isoformat(),
                        "document_id": self._generate_doc_id(file_path),
                        "title": props.title or file_path.stem,
                        "author": props.author or "Unknown",
                        "created": str(props.created) if props.created else None,
                        "modified": str(props.modified) if props.modified else None,
                        "paragraphs_count": len(paragraphs),
                        "content_length": len(text_content)
                    }
                }
            else:
                return self._process_text_document(file_path)
        except Exception as e:
            return self._create_error_result(file_path, f"Office processing error: {e}")

    def _process_text_document(self, file_path: Path) -> Dict[str, Any]:
        """Process plain text documents"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            chunks = self._create_text_chunks(content, file_path, "text")
            return {
                "chunks": chunks,
                "images": [],
                "metadata": {
                    "source_type": "text",
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat(),
                    "document_id": self._generate_doc_id(file_path),
                    "content_length": len(content),
                    "lines_count": len(content.split('\n'))
                }
            }
        except Exception as e:
            return self._create_error_result(file_path, f"Text processing error: {e}")

    def _process_archive(self, file_path: Path) -> Dict[str, Any]:
        """Process ZIP archives"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                archive_info = f"ZIP Archive: {file_path.name}\n"
                archive_info += f"Contains {len(file_list)} files:\n\n"
                for file_name in file_list[:20]:
                    try:
                        file_info = zip_file.getinfo(file_name)
                        archive_info += f"- {file_name} ({file_info.file_size} bytes)\n"
                    except:
                        archive_info += f"- {file_name}\n"
                if len(file_list) > 20:
                    archive_info += f"... and {len(file_list) - 20} more files\n"
                chunks = [{
                    "content": archive_info,
                    "metadata": {
                        "source": str(file_path),
                        "source_type": "archive",
                        "document_id": self._generate_doc_id(file_path),
                        "chunk_id": f"{self._generate_doc_id(file_path)}_archive",
                        "content_type": "archive_metadata",
                        "files_count": len(file_list),
                        "archive_format": "zip"
                    }
                }]
                return {
                    "chunks": chunks,
                    "images": [],
                    "metadata": {
                        "source_type": "archive",
                        "file_path": str(file_path),
                        "processed_at": datetime.now().isoformat(),
                        "document_id": self._generate_doc_id(file_path),
                        "files_count": len(file_list),
                        "file_list": file_list[:50]
                    }
                }
        except Exception as e:
            return self._create_error_result(file_path, f"Archive processing error: {e}")

    def _process_reference(self, file_path: Path) -> Dict[str, Any]:
        """Process URL reference files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            chunks = [{
                "content": f"Reference: {content}",
                "metadata": {
                    "source": str(file_path),
                    "source_type": "reference",
                    "document_id": self._generate_doc_id(file_path),
                    "chunk_id": f"{self._generate_doc_id(file_path)}_ref",
                    "content_type": "reference",
                    "reference_content": content
                }
            }]
            return {
                "chunks": chunks,
                "images": [],
                "metadata": {
                    "source_type": "reference",
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat(),
                    "document_id": self._generate_doc_id(file_path),
                    "reference_content": content
                }
            }
        except Exception as e:
            return self._create_error_result(file_path, f"Reference processing error: {e}")

    def _create_text_chunks(self, text: str, file_path: Path, source_type: str) -> List[Dict]:
        """Create text chunks from content"""
        paragraphs = text.split('\n\n') if '\n\n' in text else [text]
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > 50:
                chunk = {
                    "content": paragraph.strip(),
                    "metadata": {
                        "source": str(file_path),
                        "source_type": source_type,
                        "document_id": self._generate_doc_id(file_path),
                        "chunk_id": f"{self._generate_doc_id(file_path)}_chunk_{i}",
                        "chunk_index": i,
                        "content_type": "text",
                        "word_count": len(paragraph.split()),
                        "char_count": len(paragraph)
                    }
                }
                chunks.append(chunk)
        return chunks

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate document ID"""
        return f"teppl_{file_path.stem}".replace(" ", "_").replace("-", "_")

    def _create_unsupported_result(self, file_path: Path, file_ext: str) -> Dict[str, Any]:
        """Create result for unsupported file types"""
        return {
            "chunks": [],
            "images": [],
            "metadata": {
                "source_type": "unsupported",
                "file_path": str(file_path),
                "processed_at": datetime.now().isoformat(),
                "document_id": self._generate_doc_id(file_path),
                "error": f"Unsupported file type: {file_ext}",
                "file_extension": file_ext
            }
        }

    def _create_error_result(self, file_path: Path, error_msg: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "chunks": [],
            "images": [],
            "metadata": {
                "source_type": "error",
                "file_path": str(file_path),
                "processed_at": datetime.now().isoformat(),
                "document_id": self._generate_doc_id(file_path),
                "error": error_msg,
                "processing_failed": True
            }
        }


class EnhancedBatchProcessor:
    """Enhanced batch processor with checkpoint/resume functionality"""
    
    def __init__(
        self,
        documents_path: str = "./documents",
        storage_path: str = "./storage/chroma_enhanced",
        batch_size: int = 50,
        progress_file: str = "batch_progress.json",
        checkpoint_frequency: int = 10  # Save checkpoint every N files
    ):
        self.documents_path = Path(documents_path)
        self.storage_path = Path(storage_path)
        self.batch_size = batch_size
        self.checkpoint_frequency = checkpoint_frequency
        
        # Initialize components
        self.progress_tracker = BatchProgressTracker(progress_file)
        self.shutdown_handler = GracefulShutdownHandler(self.progress_tracker)
        self.multi_processor = MultiFormatProcessor(str(documents_path))
        
        if VECTOR_STORE_AVAILABLE:
            self.vector_store = EnhancedMultimodalVectorStore(str(storage_path))
        else:
            self.vector_store = None
        
        # Supported extensions
        self.supported_extensions = {
            'pdf', 'aspx', 'json', 'zip', 'docx', 'xlsx', 'xhtml',
            'txt', 'xml', 'doc', 'html', 'xlsm', 'url', 'xls',
            'xlsb', 'dotx'
        }
        
        logger.info("ðŸš€ Enhanced Batch Processor with Checkpoint System initialized")

    def discover_all_files(self) -> List[Path]:
        """Discover all supported files"""
        all_files = []
        
        for file_path in self.documents_path.rglob('*.*'):
            if file_path.is_file():
                ext = file_path.suffix.lower().lstrip('.')
                if ext in self.supported_extensions:
                    all_files.append(file_path)
        
        logger.info(f"ðŸ“ Discovered {len(all_files)} supported files")
        return all_files

    def process_all_files_with_resume(self, dry_run: bool = False) -> Dict[str, Any]:
        """Process all files with resume capability"""
        # Discover all files
        all_files = self.discover_all_files()
        
        if dry_run:
            return self._perform_dry_run(all_files)
        
        # Get files that still need processing
        remaining_files = self.progress_tracker.get_remaining_files(all_files)
        
        # Initialize session
        system_info = {
            'multiformat_enabled': True,
            'supported_formats': list(self.supported_extensions),
            'documents_path': str(self.documents_path),
            'storage_path': str(self.storage_path),
            'enhanced_image_processing': IMAGE_PROCESSOR_AVAILABLE,
            'vector_store_available': VECTOR_STORE_AVAILABLE
        }
        
        self.progress_tracker.start_session(len(all_files), system_info)
        
        logger.info(f"ðŸ”„ Starting/Resuming processing: {len(remaining_files)} files remaining")
        
        # Process in batches
        return self._process_files_in_batches(remaining_files)

    def _perform_dry_run(self, all_files: List[Path]) -> Dict[str, Any]:
        """Perform dry run analysis"""
        remaining_files = self.progress_tracker.get_remaining_files(all_files)
        
        # Count by file type
        files_by_type = {}
        for file_path in remaining_files:
            ext = file_path.suffix.lower().lstrip('.')
            files_by_type[ext] = files_by_type.get(ext, 0) + 1
        
        # Estimate processing time
        avg_time_per_file = 35  # seconds (conservative estimate)
        total_estimated_seconds = len(remaining_files) * avg_time_per_file
        estimated_hours = total_estimated_seconds / 3600
        
        progress = self.progress_tracker.get_progress_summary()
        
        return {
            'dry_run': True,
            'total_files_discovered': len(all_files),
            'already_processed': len(all_files) - len(remaining_files),
            'files_to_process': len(remaining_files),
            'files_by_type': files_by_type,
            'estimated_processing_time_hours': round(estimated_hours, 2),
            'estimated_processing_time_minutes': round(total_estimated_seconds / 60, 1),
            'batch_size': self.batch_size,
            'estimated_batches': (len(remaining_files) + self.batch_size - 1) // self.batch_size,
            'resume_session': progress['session_id'],
            'previous_progress': progress,
            'enhanced_processing_enabled': IMAGE_PROCESSOR_AVAILABLE
        }

    def _process_files_in_batches(self, files_to_process: List[Path]) -> Dict[str, Any]:
        """Process files in batches with checkpointing"""
        total_files = len(files_to_process)
        total_batches = (total_files + self.batch_size - 1) // self.batch_size
        
        results = {
            'processed_successfully': 0,
            'processing_errors': 0,
            'start_time': datetime.now().isoformat(),
            'batches_completed': 0
        }
        
        for batch_idx in range(0, total_files, self.batch_size):
            if self.shutdown_handler.should_shutdown():
                logger.info("ðŸ›‘ Shutdown requested - stopping after current batch")
                break
            
            # Get current batch
            batch_end = min(batch_idx + self.batch_size, total_files)
            current_batch = files_to_process[batch_idx:batch_end]
            batch_num = (batch_idx // self.batch_size) + 1
            
            # Update batch info
            self.progress_tracker.update_batch_info(batch_num, len(current_batch))
            
            logger.info(f"ðŸ“¦ Processing batch {batch_num}/{total_batches} ({len(current_batch)} files)")
            
            # Process batch
            batch_results = self._process_single_batch(current_batch, batch_num, total_batches)
            
            # Update results
            results['processed_successfully'] += batch_results['processed_successfully']
            results['processing_errors'] += batch_results['processing_errors']
            results['batches_completed'] += 1
            
            # Print progress report every few batches
            if batch_num % 5 == 0 or batch_num == total_batches:
                self.progress_tracker.print_progress_report()
        
        # Final results
        results['end_time'] = datetime.now().isoformat()
        results['final_progress'] = self.progress_tracker.get_progress_summary()
        
        # Final checkpoint
        self.progress_tracker.save_checkpoint()
        
        logger.info("ðŸŽ‰ Batch processing completed!")
        return results

    def _process_single_batch(self, batch_files: List[Path], batch_num: int, total_batches: int) -> Dict[str, Any]:
        """Process a single batch of files"""
        batch_results = {
            'processed_successfully': 0,
            'processing_errors': 0
        }
        
        for file_idx, file_path in enumerate(batch_files, 1):
            if self.shutdown_handler.should_shutdown():
                break
            
            try:
                # Record start time
                start_time = time.time()
                
                # Process the document
                processed_doc = self.multi_processor.process_document(file_path)
                
                # Check if processing was successful
                if processed_doc.get("chunks") or processed_doc.get("images"):
                    # Add to vector store if available
                    if self.vector_store:
                        success = self.vector_store.add_enhanced_document(processed_doc)
                        
                        if success:
                            processing_time = time.time() - start_time
                            self.progress_tracker.mark_file_processed(str(file_path), processing_time)
                            batch_results['processed_successfully'] += 1
                            logger.info(f"âœ… [{batch_num}/{total_batches}:{file_idx}] {file_path.name} ({processing_time:.1f}s)")
                        else:
                            self.progress_tracker.mark_file_failed(str(file_path), "Vector store insertion failed")
                            batch_results['processing_errors'] += 1
                            logger.error(f"âŒ [{batch_num}/{total_batches}:{file_idx}] {file_path.name} - Vector store failed")
                    else:
                        # No vector store, but processing succeeded
                        processing_time = time.time() - start_time
                        self.progress_tracker.mark_file_processed(str(file_path), processing_time)
                        batch_results['processed_successfully'] += 1
                        logger.info(f"âœ… [{batch_num}/{total_batches}:{file_idx}] {file_path.name} (no vector store)")
                else:
                    self.progress_tracker.mark_file_failed(str(file_path), "No content extracted")
                    batch_results['processing_errors'] += 1
                    logger.error(f"âŒ [{batch_num}/{total_batches}:{file_idx}] {file_path.name} - No content extracted")
                    
            except Exception as e:
                self.progress_tracker.mark_file_failed(str(file_path), str(e))
                batch_results['processing_errors'] += 1
                logger.error(f"âŒ [{batch_num}/{total_batches}:{file_idx}] {file_path.name} - {str(e)}")
            
            # Periodic checkpoint
            if file_idx % self.checkpoint_frequency == 0:
                self.progress_tracker.save_checkpoint()
        
        # Save checkpoint after each batch
        self.progress_tracker.save_checkpoint()
        
        return batch_results


def main():
    """Enhanced main function with checkpoint/resume support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced TEPPL batch processor with checkpoint/resume and image processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_processor_complete.py --dry-run      # Check what needs processing
  python batch_processor_complete.py --resume       # Resume from last checkpoint
  python batch_processor_complete.py --restart      # Start fresh (clear progress)
  python batch_processor_complete.py --batch-size 25 # Use smaller batches
        """
    )
    
    parser.add_argument('--documents-path', default='./documents',
                       help='Path to documents directory (default: ./documents)')
    parser.add_argument('--storage-path', default='./storage/chroma_enhanced',
                       help='Path to vector store storage (default: ./storage/chroma_enhanced)')
    parser.add_argument('--progress-file', default='batch_progress.json',
                       help='Progress tracking file (default: batch_progress.json)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of files to process in each batch (default: 50)')
    parser.add_argument('--checkpoint-frequency', type=int, default=10,
                       help='Save checkpoint every N files (default: 10)')
    
    # Processing options
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze files without processing')
    parser.add_argument('--resume', action='store_true',
                       help='Resume processing from last checkpoint')
    parser.add_argument('--restart', action='store_true',
                       help='Start fresh (clear existing progress)')
    parser.add_argument('--status', action='store_true',
                       help='Show current processing status')
    
    # System options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle restart option
    if args.restart:
        progress_file = Path(args.progress_file)
        if progress_file.exists():
            progress_file.unlink()
            logger.info("ðŸ—‘ï¸ Cleared existing progress - starting fresh")
    
    # Initialize processor
    processor = EnhancedBatchProcessor(
        documents_path=args.documents_path,
        storage_path=args.storage_path,
        batch_size=args.batch_size,
        progress_file=args.progress_file,
        checkpoint_frequency=args.checkpoint_frequency
    )
    
    # Handle status check
    if args.status:
        processor.progress_tracker.print_progress_report()
        return
    
    logger.info("ðŸš€ Starting Enhanced TEPPL Batch Processor with Complete Multimodal Processing")
    logger.info(f"ðŸ“ Documents path: {args.documents_path}")
    logger.info(f"ðŸ’¾ Storage path: {args.storage_path}")
    logger.info(f"ðŸ“Š Progress file: {args.progress_file}")
    logger.info(f"ðŸ“¦ Batch size: {args.batch_size}")
    logger.info(f"ðŸŽ¨ Enhanced image processing: {IMAGE_PROCESSOR_AVAILABLE}")
    logger.info(f"ðŸ’¾ Vector store available: {VECTOR_STORE_AVAILABLE}")
    
    try:
        # Process documents with resume capability
        results = processor.process_all_files_with_resume(dry_run=args.dry_run)
        
        # Display results
        if args.dry_run:
            print("\nðŸ” DRY RUN ANALYSIS WITH RESUME INFO")
            print("="*60)
            print(f"ðŸ“ Total files discovered: {results['total_files_discovered']:,}")
            print(f"âœ… Already processed: {results['already_processed']:,}")
            print(f"â³ Files to process: {results['files_to_process']:,}")
            print(f"â±ï¸ Estimated time: {results['estimated_processing_time_hours']:.1f} hours")
            print(f"ðŸ“¦ Estimated batches: {results['estimated_batches']}")
            print(f"ðŸ†” Resume session: {results['resume_session']}")
            print(f"ðŸŽ¨ Enhanced processing: {results['enhanced_processing_enabled']}")
            
            print(f"\nðŸ“‹ FILES BY TYPE:")
            for file_type, count in sorted(results['files_by_type'].items(),
                                         key=lambda x: x[1], reverse=True):
                print(f"  {file_type.upper():<6}: {count:>5,} files")
            
            if results['previous_progress']['files_processed'] > 0:
                print(f"\nðŸ“ˆ PREVIOUS PROGRESS:")
                prev = results['previous_progress']
                print(f"  Progress: {prev['progress_percentage']:.1f}%")
                print(f"  Processed: {prev['files_processed']:,}")
                print(f"  Failed: {prev['files_failed']}")
                
        else:
            print("\nðŸ“Š BATCH PROCESSING RESULTS")
            print("="*60)
            print(f"âœ… Successfully processed: {results['processed_successfully']:,}")
            print(f"âŒ Processing errors: {results['processing_errors']:,}")
            print(f"ðŸ“¦ Batches completed: {results['batches_completed']}")
            
            # Final progress report
            processor.progress_tracker.print_progress_report()
            
    except KeyboardInterrupt:
        logger.info("\nðŸ›‘ Processing interrupted by user")
        processor.progress_tracker.save_checkpoint()
        print(f"\nðŸ“Š Progress saved to: {args.progress_file}")
        print("Run with --resume to continue from this point")
        
    except Exception as e:
        logger.error(f"âŒ Critical error: {e}")
        processor.progress_tracker.save_checkpoint()
        raise


if __name__ == "__main__":
    main()