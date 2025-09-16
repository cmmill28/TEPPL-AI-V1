# src/document_processor.py
"""
Enhanced PDF Document Processor for TEPPL
Focused on PDF processing only - no circular imports
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

# PDF processing imports
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("❌ PyMuPDF not available. Install with: pip install PyMuPDF")

# Image processing imports
try:
    from PIL import Image
    import io
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# OCR imports (optional)
try:
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """Enhanced PDF document processor for TEPPL documents"""
    
    def __init__(self, documents_path: str = "./documents"):
        if not FITZ_AVAILABLE:
            raise ImportError("PyMuPDF required for PDF processing")
        
        self.documents_path = Path(documents_path)
        self.output_dir = Path("./storage/extracted_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TEPPL-specific patterns for enhanced processing
        self.teppl_patterns = {
            'regulatory_terms': [
                'MUTCD', 'CFR', 'USC', 'NCDOT', 'speed limit', 'mph',
                'regulatory sign', 'warning sign', 'guide sign',
                'traffic signal', 'intersection', 'crosswalk',
                'pavement marking', 'lane marking', 'school zone'
            ],
            'technical_terms': [
                'clearance interval', 'sight distance', 'signal timing',
                'geometric design', 'traffic control', 'work zone'
            ],
            'measurement_patterns': [
                r'\d+\.?\d*\s*(?:inch|inches|feet|ft|mph|degrees?)',
                r'\d+\.?\d*\s*x\s*\d+\.?\d*',
                r'(?:width|height|diameter|radius)\s*:?\s*\d+'
            ]
        }
        
        logger.info("✅ Enhanced PDF Document Processor initialized")

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file and extract text, images, and metadata"""
        try:
            logger.info(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Extract document metadata
            doc_metadata = self._extract_document_metadata(doc, pdf_path)
            
            # Extract text content with enhanced chunking
            text_chunks = self._extract_enhanced_text_chunks(doc, doc_metadata)
            
            # Extract images if available
            images = []
            if PILLOW_AVAILABLE:
                images = self._extract_pdf_images(doc, doc_metadata)
            
            # Close document
            doc.close()
            
            # Create final result
            result = {
                "chunks": text_chunks,
                "images": images,
                "metadata": doc_metadata
            }
            
            logger.info(f"✅ Processed PDF: {len(text_chunks)} chunks, {len(images)} images")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF {pdf_path}: {e}")
            return {
                "chunks": [],
                "images": [],
                "metadata": {
                    "error": str(e),
                    "file_path": pdf_path,
                    "processed_at": datetime.now().isoformat()
                }
            }

    def _extract_document_metadata(self, doc, pdf_path: str) -> Dict[str, Any]:
        """Extract comprehensive document metadata"""
        try:
            # Basic PDF metadata
            pdf_metadata = doc.metadata
            
            # Generate document ID
            doc_id = self._generate_document_id(pdf_path)
            
            # Extract TEPPL-specific metadata
            first_page_text = doc[0].get_text() if len(doc) > 0 else ""
            teppl_metadata = self._extract_teppl_metadata(first_page_text)
            
            metadata = {
                "document_id": doc_id,
                "file_path": pdf_path,
                "filename": os.path.basename(pdf_path),
                "total_pages": len(doc),
                "processed_at": datetime.now().isoformat(),
                "processing_method": "enhanced_pdf_processor",
                
                # PDF metadata
                "title": pdf_metadata.get("title", "").strip() or os.path.basename(pdf_path),
                "author": pdf_metadata.get("author", "").strip(),
                "subject": pdf_metadata.get("subject", "").strip(),
                "creator": pdf_metadata.get("creator", "").strip(),
                "producer": pdf_metadata.get("producer", "").strip(),
                
                # TEPPL-specific metadata
                "teppl_metadata": teppl_metadata,
                
                # Content analysis
                "has_images": self._has_images(doc),
                "estimated_word_count": self._estimate_word_count(doc),
                
                # Technical analysis
                "technical_terms": self._extract_technical_terms_from_doc(doc),
                "regulatory_references": self._extract_regulatory_references(doc),
                "sections_detected": self._detect_document_sections(doc)
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {
                "document_id": self._generate_document_id(pdf_path),
                "file_path": pdf_path,
                "processed_at": datetime.now().isoformat(),
                "error": str(e)
            }

    def _generate_document_id(self, pdf_path: str) -> str:
        """Generate a unique document ID"""
        filename = os.path.basename(pdf_path)
        name_part = re.sub(r'[^\w\-_]', '_', filename.replace('.pdf', ''))
        return f"teppl_{name_part}".lower()

    def _extract_teppl_metadata(self, first_page_text: str) -> Dict[str, Any]:
        """Extract TEPPL-specific metadata from document text"""
        teppl_meta = {
            "teppl_topic": "Unknown",
            "publication_year": "Unknown",
            "document_type": "Unknown",
            "official_name": "Unknown"
        }
        
        try:
            text = first_page_text.lower()
            
            # Extract topic
            if "adopt" in text and "highway" in text:
                teppl_meta["teppl_topic"] = "Adopt-A-Highway"
            elif "traffic" in text and "signal" in text:
                teppl_meta["teppl_topic"] = "Traffic Signals"
            elif "sign" in text and ("regulatory" in text or "warning" in text):
                teppl_meta["teppl_topic"] = "Traffic Signs"
            elif "pavement" in text and "marking" in text:
                teppl_meta["teppl_topic"] = "Pavement Markings"
            elif "geometric" in text and "design" in text:
                teppl_meta["teppl_topic"] = "Geometric Design"
            
            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', first_page_text)
            if year_match:
                teppl_meta["publication_year"] = year_match.group()
            
            # Extract document type
            if "manual" in text:
                teppl_meta["document_type"] = "Manual"
            elif "specification" in text:
                teppl_meta["document_type"] = "Specification"
            elif "standard" in text:
                teppl_meta["document_type"] = "Standard"
            elif "guideline" in text:
                teppl_meta["document_type"] = "Guideline"
            
            # Extract official name (first non-empty line that's not just numbers/symbols)
            lines = first_page_text.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                cleaned = line.strip()
                if (len(cleaned) > 10 and 
                    not cleaned.isdigit() and 
                    any(c.isalpha() for c in cleaned)):
                    teppl_meta["official_name"] = cleaned[:100]  # Limit length
                    break
                    
        except Exception as e:
            logger.warning(f"Error extracting TEPPL metadata: {e}")
        
        return teppl_meta

    def _extract_enhanced_text_chunks(self, doc, doc_metadata: Dict) -> List[Dict[str, Any]]:
        """Extract text with enhanced chunking for better semantic search"""
        chunks = []
        
        try:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                
                if not page_text.strip():
                    continue
                
                # Split into paragraphs and process
                paragraphs = self._split_into_paragraphs(page_text)
                
                for para_idx, paragraph in enumerate(paragraphs):
                    if len(paragraph.strip()) < 50:  # Skip very short paragraphs
                        continue
                    
                    # Create chunk with enhanced metadata
                    chunk = {
                        "content": paragraph.strip(),
                        "metadata": {
                            "source": doc_metadata["file_path"],
                            "document_id": doc_metadata["document_id"],
                            "chunk_id": f"{doc_metadata['document_id']}_p{page_num}_c{para_idx}",
                            "page_number": page_num,
                            "chunk_index": para_idx,
                            "content_type": "text",
                            
                            # Content analysis
                            "word_count": len(paragraph.split()),
                            "char_count": len(paragraph),
                            "has_technical_terms": self._has_technical_terms(paragraph),
                            "has_measurements": self._has_measurements(paragraph),
                            "has_regulatory_refs": self._has_regulatory_references(paragraph),
                            
                            # TEPPL-specific analysis
                            "teppl_terms": self._extract_teppl_terms(paragraph),
                            "regulatory_level": self._assess_regulatory_level(paragraph),
                            "content_category": self._categorize_content(paragraph)
                        }
                    }
                    
                    chunks.append(chunk)
            
            logger.info(f"✅ Extracted {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text chunks: {e}")
            return []

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split on double newlines first
        paragraphs = text.split('\n\n')
        
        # If no double newlines, split on single newlines but group related lines
        if len(paragraphs) == 1:
            lines = text.split('\n')
            current_para = []
            paragraphs = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_para:
                        paragraphs.append('\n'.join(current_para))
                        current_para = []
                else:
                    current_para.append(line)
            
            if current_para:
                paragraphs.append('\n'.join(current_para))
        
        # Further split very long paragraphs
        final_paragraphs = []
        for para in paragraphs:
            if len(para) > 2000:  # Split very long paragraphs
                sentences = para.split('. ')
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) > 1500 and current_chunk:
                        final_paragraphs.append('. '.join(current_chunk) + '.')
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                
                if current_chunk:
                    final_paragraphs.append('. '.join(current_chunk))
            else:
                final_paragraphs.append(para)
        
        return [p for p in final_paragraphs if p.strip()]

    def _extract_pdf_images(self, doc, doc_metadata: Dict) -> List[Dict[str, Any]]:
        """Extract images from PDF document"""
        images = []
        
        if not PILLOW_AVAILABLE:
            logger.warning("⚠️ Pillow not available - skipping image extraction")
            return images
        
        try:
            doc_id = doc_metadata["document_id"]
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get images from this page
                image_list = page.get_images(full=True)
                
                if not image_list:
                    continue
                    
                logger.info(f"📷 Found {len(image_list)} images on page {page_num + 1}")
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image data
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image.get("ext", "png")
                        
                        # Skip very small images (likely decorative)
                        if len(image_bytes) < 1024:  # Less than 1KB
                            continue
                        
                        # Generate unique image ID
                        image_id = f"{doc_id}_p{page_num + 1}_img{img_index + 1}"
                        
                        # Save image to storage
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = self.output_dir / image_filename
                        
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # Get image properties
                        try:
                            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                                width, height = pil_img.size
                                mode = pil_img.mode
                        except Exception:
                            width, height, mode = 0, 0, "unknown"
                        
                        # Extract text context around image (simplified)
                        page_text = page.get_text()
                        
                        # Create image metadata
                        image_data = {
                            "id": image_id,
                            "type": "embedded_image",
                            "page": page_num + 1,
                            "file_path": str(image_path),
                            "format": image_ext,
                            "size_bytes": len(image_bytes),
                            "dimensions": {
                                "width": width,
                                "height": height
                            },
                            "extracted_text": "",  # Could add OCR here later
                            "context": {
                                "nearby_text": page_text[:200],  # First 200 chars of page
                                "page_number": page_num + 1
                            },
                            "metadata": {
                                "document_id": doc_id,
                                "extraction_method": "pymupdf",
                                "xref": xref,
                                "original_format": image_ext,
                                "color_mode": mode,
                                "aspect_ratio": round(width / height, 2) if height > 0 else 1.0
                            },
                            # Classification (basic)
                            "teppl_category": self._classify_teppl_image(page_text),
                            "confidence": 0.7
                        }
                        
                        images.append(image_data)
                        logger.info(f"✅ Extracted image: {image_id} ({width}x{height}, {len(image_bytes)} bytes)")
                        
                    except Exception as e:
                        logger.error(f"❌ Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            logger.info(f"🎨 Total images extracted: {len(images)}")
            return images
            
        except Exception as e:
            logger.error(f"❌ Error in PDF image extraction: {e}")
            return []

    def _classify_teppl_image(self, page_text: str) -> str:
        """Basic TEPPL image classification based on page text"""
        text_lower = page_text.lower()
        
        if any(term in text_lower for term in ['sign', 'regulatory', 'warning', 'guide']):
            return 'traffic_signs'
        elif any(term in text_lower for term in ['signal', 'light', 'intersection']):
            return 'traffic_signals'
        elif any(term in text_lower for term in ['marking', 'stripe', 'lane', 'crosswalk']):
            return 'pavement_markings'
        elif any(term in text_lower for term in ['plan', 'section', 'detail', 'drawing']):
            return 'engineering_drawings'
        elif any(term in text_lower for term in ['chart', 'graph', 'table', 'data']):
            return 'charts_graphs'
        else:
            return 'general'

    def _save_image(self, image_bytes: bytes, image_id: str, ext: str) -> str:
        """Save image bytes to file"""
        filename = f"{image_id}.{ext}"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            return str(filepath)
        except Exception as e:
            logger.warning(f"Error saving image {image_id}: {e}")
            return ""

    def _analyze_image_properties(self, image_bytes: bytes, image_ext: str) -> Dict:
        """Analyze basic image properties"""
        try:
            if PILLOW_AVAILABLE:
                image = Image.open(io.BytesIO(image_bytes))
                width, height = image.size
                
                return {
                    'dimensions': {'width': width, 'height': height},
                    'aspect_ratio': width / height if height > 0 else 1.0,
                    'format': image_ext,
                    'mode': image.mode
                }
            else:
                return {'dimensions': {'width': 0, 'height': 0}}
        except Exception as e:
            logger.warning(f"Error analyzing image properties: {e}")
            return {'dimensions': {'width': 0, 'height': 0}}

    def _extract_image_context(self, page, page_num: int) -> Dict[str, str]:
        """Extract text context around images"""
        try:
            page_text = page.get_text()
            # For now, just return page text as context
            # Could be enhanced to find text specifically near images
            return {
                'nearby_text': page_text[:500] if page_text else "",
                'page_number': page_num,
                'full_page_text': page_text[:1000] if page_text else ""
            }
        except Exception as e:
            logger.warning(f"Error extracting image context: {e}")
            return {'nearby_text': '', 'page_number': page_num}

    # Helper methods for content analysis
    def _has_technical_terms(self, text: str) -> bool:
        """Check if text contains technical terms"""
        text_lower = text.lower()
        return any(term in text_lower for term in self.teppl_patterns['technical_terms'])

    def _has_measurements(self, text: str) -> bool:
        """Check if text contains measurements"""
        for pattern in self.teppl_patterns['measurement_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _has_regulatory_references(self, text: str) -> bool:
        """Check if text contains regulatory references"""
        return bool(re.search(r'\b(?:MUTCD|CFR|USC)\b', text, re.IGNORECASE))

    def _extract_teppl_terms(self, text: str) -> List[str]:
        """Extract TEPPL-specific terms from text"""
        text_lower = text.lower()
        found_terms = []
        
        for term in self.teppl_patterns['regulatory_terms'] + self.teppl_patterns['technical_terms']:
            if term in text_lower:
                found_terms.append(term)
        
        return list(set(found_terms))

    def _assess_regulatory_level(self, text: str) -> str:
        """Assess the regulatory level of text content"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['shall', 'must', 'required', 'mandatory']):
            return 'mandatory'
        elif any(term in text_lower for term in ['should', 'recommended', 'preferred']):
            return 'recommended'
        elif any(term in text_lower for term in ['may', 'optional', 'permitted']):
            return 'optional'
        else:
            return 'informational'

    def _categorize_content(self, text: str) -> str:
        """Categorize content type"""
        text_lower = text.lower()
        
        if 'sign' in text_lower:
            return 'signage'
        elif 'signal' in text_lower:
            return 'traffic_signals'
        elif 'marking' in text_lower:
            return 'pavement_markings'
        elif 'geometric' in text_lower or 'design' in text_lower:
            return 'geometric_design'
        elif 'maintenance' in text_lower:
            return 'maintenance'
        else:
            return 'general'

    # Document-level analysis methods
    def _has_images(self, doc) -> bool:
        """Check if document contains images"""
        try:
            for page in doc:
                if page.get_images():
                    return True
            return False
        except:
            return False

    def _estimate_word_count(self, doc) -> int:
        """Estimate total word count in document"""
        try:
            total_words = 0
            for page in doc:
                page_text = page.get_text()
                total_words += len(page_text.split())
            return total_words
        except:
            return 0

    def _extract_technical_terms_from_doc(self, doc) -> List[str]:
        """Extract technical terms from entire document"""
        all_terms = set()
        
        try:
            for page in doc:
                page_text = page.get_text()
                terms = self._extract_teppl_terms(page_text)
                all_terms.update(terms)
        except Exception as e:
            logger.warning(f"Error extracting technical terms: {e}")
        
        return sorted(list(all_terms))

    def _extract_regulatory_references(self, doc) -> List[str]:
        """Extract regulatory references from document"""
        references = set()
        
        try:
            for page in doc:
                page_text = page.get_text()
                # Find MUTCD references
                mutcd_refs = re.findall(r'\bMUTCD\s*(?:Section\s*)?\d*[A-Z]?(?:\.\d+)*\b', page_text, re.IGNORECASE)
                references.update(mutcd_refs)
                
                # Find CFR references
                cfr_refs = re.findall(r'\bCFR\s*\d+(?:\.\d+)*\b', page_text, re.IGNORECASE)
                references.update(cfr_refs)
                
                # Find USC references
                usc_refs = re.findall(r'\bUSC\s*\d+(?:\.\d+)*\b', page_text, re.IGNORECASE)
                references.update(usc_refs)
        except Exception as e:
            logger.warning(f"Error extracting regulatory references: {e}")
        
        return sorted(list(references))

    def _detect_document_sections(self, doc) -> List[str]:
        """Detect major sections in the document"""
        sections = []
        
        try:
            for page in doc:
                page_text = page.get_text()
                # Look for section headers (simple heuristic)
                lines = page_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # Check if line looks like a section header
                    if (len(line) > 5 and len(line) < 100 and 
                        (line.isupper() or 
                         re.match(r'^\d+\.?\s+[A-Z]', line) or
                         re.match(r'^[A-Z][^.!?]*$', line))):
                        sections.append(line)
        except Exception as e:
            logger.warning(f"Error detecting sections: {e}")
        
        return sections[:20]  # Limit to first 20 sections


if __name__ == "__main__":
    # Test the processor
    processor = EnhancedDocumentProcessor()
    logger.info("✅ Enhanced PDF Document Processor ready for testing")

