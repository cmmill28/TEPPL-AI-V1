# src/multimodal_drawing_processor.py
"""
Integrated multimodal drawing processor for TEPPL documents.

Combines drawing extraction, linking, and vector store integration
with enhanced citation and searchability features.
"""

import os
import uuid
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import re
from PIL import Image
import io

logger = logging.getLogger(__name__)

class MultimodalDrawingProcessor:
    """
    Integrated processor for extracting, analyzing, and linking technical drawings
    from TEPPL documents with full multimodal search capabilities.
    """
    
    def __init__(self, documents_path: str = "./documents"):
        self.documents_path = documents_path
        self.drawings_folder = os.path.join(documents_path, "drawings")
        self.thumbnails_folder = os.path.join(documents_path, "thumbnails")
        
        # Create directories
        os.makedirs(self.drawings_folder, exist_ok=True)
        os.makedirs(self.thumbnails_folder, exist_ok=True)
        
        # TEPPL domain knowledge for drawing classification
        self.drawing_patterns = {
            'signage': {
                'keywords': ['sign', 'signage', 'regulatory', 'warning', 'guide'],
                'text_patterns': [r'sign\s+\d+', r'figure\s+\d+.*sign', r'dimensions?\s*:\s*\d+'],
                'typical_dimensions': [(30, 30), (36, 36), (48, 48)]  # common sign sizes
            },
            'traffic_control': {
                'keywords': ['traffic', 'signal', 'light', 'intersection', 'control'],
                'text_patterns': [r'signal\s+timing', r'clearance\s+interval', r'phase'],
                'typical_dimensions': [(12, 12), (8, 8)]  # signal head sizes
            },
            'pavement_markings': {
                'keywords': ['marking', 'stripe', 'lane', 'crosswalk', 'pavement'],
                'text_patterns': [r'\d+\s*inch\s+line', r'stripe\s+width', r'marking\s+pattern'],
                'typical_dimensions': [(4, 24), (6, 24), (8, 24)]  # line widths and lengths
            },
            'geometric_design': {
                'keywords': ['radius', 'curve', 'grade', 'sight', 'distance'],
                'text_patterns': [r'radius\s*=?\s*\d+', r'sight\s+distance', r'vertical\s+curve'],
                'typical_dimensions': None
            }
        }
        
        logger.info("Multimodal Drawing Processor initialized")

    def extract_and_link_drawings(
        self, 
        pdf_path: str, 
        text_chunks: List[Dict[str, Any]], 
        doc_metadata: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract drawings from PDF and intelligently link them to relevant text chunks.
        Returns (drawings, enhanced_chunks_with_drawing_links)
        """
        logger.info(f"🎨 Extracting drawings from {os.path.basename(pdf_path)}")
        
        drawings = []
        enhanced_chunks = text_chunks.copy()
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_number, page in enumerate(doc, start=1):
                # Extract images from page
                page_drawings = self._extract_page_drawings(
                    page, page_number, pdf_path, doc_metadata
                )
                
                # Link drawings to nearby text
                linked_drawings = self._link_drawings_to_text(
                    page_drawings, page, enhanced_chunks, page_number
                )
                
                drawings.extend(linked_drawings)
            
            doc.close()
            
            # Create bidirectional links between drawings and chunks
            enhanced_chunks = self._create_bidirectional_links(enhanced_chunks, drawings)
            
            logger.info(f"✅ Extracted {len(drawings)} drawings with text links")
            
        except Exception as e:
            logger.error(f"Error extracting drawings: {e}")
        
        return drawings, enhanced_chunks

    def _extract_page_drawings(
        self, 
        page, 
        page_number: int, 
        pdf_path: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract all images from a single page with enhanced metadata"""
        
        drawings = []
        image_list = page.get_images(full=True)
        
        if not image_list:
            return drawings
        
        doc_id = doc_metadata.get('document_id', 'unknown')
        
        for img_index, img in enumerate(image_list):
            try:
                # Extract image data
                xref = img[0]
                base_image = page.get_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")
                
                # Skip very small images (likely decorative)
                if len(image_bytes) < 1024:  # Less than 1KB
                    continue
                
                # Generate unique drawing ID
                drawing_id = f"{doc_id}_p{page_number}_img{img_index}"
                
                # Create drawing filename
                drawing_filename = f"{drawing_id}.{image_ext}"
                drawing_path = os.path.join(self.drawings_folder, drawing_filename)
                
                # Save image
                with open(drawing_path, "wb") as f:
                    f.write(image_bytes)
                
                # Create thumbnail
                thumbnail_path = self._create_thumbnail(drawing_path, drawing_id)
                
                # Analyze image content
                image_analysis = self._analyze_image_content(image_bytes, image_ext)
                
                # Extract surrounding text context
                text_context = self._extract_text_context(page, img, page_number)
                
                # Classify drawing type
                drawing_type = self._classify_drawing_type(text_context, image_analysis)
                
                # Create drawing metadata
                drawing_metadata = {
                    "drawing_id": drawing_id,
                    "source_document": doc_metadata.get('document_id'),
                    "source_pdf": pdf_path,
                    "page_number": page_number,
                    "image_index": img_index,
                    "file_path": drawing_path,
                    "thumbnail_path": thumbnail_path,
                    "image_format": image_ext,
                    "file_size": len(image_bytes),
                    "drawing_type": drawing_type,
                    "extracted_at": datetime.now().isoformat(),
                    
                    # Content analysis
                    "text_context": text_context,
                    "image_analysis": image_analysis,
                    "technical_terms": self._extract_technical_terms(text_context),
                    "dimensions_detected": self._extract_dimensions(text_context),
                    "regulatory_references": self._extract_regulatory_refs(text_context),
                    
                    # TEPPL-specific metadata
                    "teppl_topic": doc_metadata.get('teppl_metadata', {}).get('teppl_topic', 'Unknown'),
                    "teppl_category": self._categorize_for_teppl(drawing_type, text_context),
                    
                    # Linking metadata
                    "linked_chunks": [],  # Will be populated by linking process
                    "spatial_context": self._get_spatial_context(page, img),
                    
                    # Citation information
                    "citation_info": {
                        "figure_number": self._extract_figure_number(text_context),
                        "caption": self._extract_caption(text_context),
                        "page_reference": f"Page {page_number}",
                        "source_citation": self._generate_drawing_citation(doc_metadata, page_number, img_index)
                    }
                }
                
                drawings.append(drawing_metadata)
                
            except Exception as e:
                logger.error(f"Error processing image {img_index} on page {page_number}: {e}")
                continue
        
        return drawings

    def _create_thumbnail(self, image_path: str, drawing_id: str) -> str:
        """Create thumbnail for quick preview"""
        try:
            thumbnail_filename = f"{drawing_id}_thumb.jpg"
            thumbnail_path = os.path.join(self.thumbnails_folder, thumbnail_filename)
            
            with Image.open(image_path) as img:
                # Create thumbnail maintaining aspect ratio
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                img.convert('RGB').save(thumbnail_path, 'JPEG', quality=85)
            
            return thumbnail_path
            
        except Exception as e:
            logger.warning(f"Could not create thumbnail for {drawing_id}: {e}")
            return ""

    def _analyze_image_content(self, image_bytes: bytes, image_ext: str) -> Dict[str, Any]:
        """Basic image content analysis"""
        try:
            # Load image for analysis
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            
            # Basic statistics
            analysis = {
                "dimensions": {"width": width, "height": height},
                "aspect_ratio": round(width / height, 2),
                "estimated_complexity": self._estimate_complexity(width, height),
                "color_mode": image.mode,
                "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            # Size-based classification
            if width > 800 or height > 600:
                analysis["size_category"] = "large_diagram"
            elif width > 400 or height > 300:
                analysis["size_category"] = "medium_diagram"
            else:
                analysis["size_category"] = "small_diagram"
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing image: {e}")
            return {"dimensions": {"width": 0, "height": 0}, "error": str(e)}

    def _estimate_complexity(self, width: int, height: int) -> str:
        """Estimate drawing complexity based on size"""
        pixel_count = width * height
        if pixel_count > 500000:
            return "high"
        elif pixel_count > 100000:
            return "medium"
        else:
            return "low"

    def _extract_text_context(self, page, img_info, page_number: int, context_radius: int = 200) -> Dict[str, str]:
        """Extract text context around an image"""
        try:
            # Get image bounding box
            img_rect = fitz.Rect(img_info[1:5])  # x0, y0, x1, y1
            
            # Expand rectangle to capture surrounding text
            context_rect = fitz.Rect(
                max(0, img_rect.x0 - context_radius),
                max(0, img_rect.y0 - context_radius),
                min(page.rect.width, img_rect.x1 + context_radius),
                min(page.rect.height, img_rect.y1 + context_radius)
            )
            
            # Extract text from context area
            context_text = page.get_text("text", clip=context_rect)
            
            # Also get text blocks for better structure
            blocks = page.get_text("blocks")
            
            # Find blocks that overlap with context area
            relevant_blocks = []
            for block in blocks:
                block_rect = fitz.Rect(block[:4])
                if block_rect.intersects(context_rect):
                    relevant_blocks.append(block[4])
            
            return {
                "raw_text": context_text.strip(),
                "structured_blocks": relevant_blocks,
                "before_image": self._get_text_before_image(page, img_rect),
                "after_image": self._get_text_after_image(page, img_rect),
                "same_page_text": page.get_text()[:500]  # First 500 chars of page
            }
            
        except Exception as e:
            logger.warning(f"Error extracting text context: {e}")
            return {"raw_text": "", "structured_blocks": [], "error": str(e)}

    def _get_text_before_image(self, page, img_rect: fitz.Rect) -> str:
        """Get text that appears before the image on the page"""
        try:
            # Get text blocks above the image
            blocks = page.get_text("blocks")
            before_text = []
            
            for block in blocks:
                block_rect = fitz.Rect(block[:4])
                if block_rect.y1 < img_rect.y0:  # Block ends before image starts
                    before_text.append(block[4])
            
            return " ".join(before_text).strip()
        except:
            return ""

    def _get_text_after_image(self, page, img_rect: fitz.Rect) -> str:
        """Get text that appears after the image on the page"""
        try:
            # Get text blocks below the image
            blocks = page.get_text("blocks")
            after_text = []
            
            for block in blocks:
                block_rect = fitz.Rect(block[:4])
                if block_rect.y0 > img_rect.y1:  # Block starts after image ends
                    after_text.append(block[4])
            
            return " ".join(after_text).strip()
        except:
            return ""

    def _classify_drawing_type(self, text_context: Dict[str, str], image_analysis: Dict[str, Any]) -> str:
        """Classify the type of technical drawing based on context and image"""
        
        all_text = " ".join([
            text_context.get("raw_text", ""),
            text_context.get("before_image", ""),
            text_context.get("after_image", "")
        ]).lower()
        
        # Score each drawing type
        scores = {}
        
        for drawing_type, patterns in self.drawing_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in patterns['keywords']:
                score += all_text.count(keyword) * 2
            
            # Pattern matching
            for pattern in patterns['text_patterns']:
                matches = len(re.findall(pattern, all_text, re.IGNORECASE))
                score += matches * 3
            
            # Dimension matching (if applicable)
            if patterns['typical_dimensions']:
                img_dims = image_analysis.get('dimensions', {})
                img_width = img_dims.get('width', 0)
                img_height = img_dims.get('height', 0)
                
                for typ_width, typ_height in patterns['typical_dimensions']:
                    # Allow for some tolerance in dimension matching
                    if (abs(img_width - typ_width * 10) < 50 or  # Rough scaling
                        abs(img_height - typ_height * 10) < 50):
                        score += 5
            
            scores[drawing_type] = score
        
        # Return the highest scoring type, or 'technical_diagram' as default
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return 'technical_diagram'

    def _extract_technical_terms(self, text_context: Dict[str, str]) -> List[str]:
        """Extract technical terms from text context"""
        all_text = " ".join(text_context.values()).lower()
        
        terms = set()
        
        # TEPPL-specific terms
        teppl_terms = [
            'speed limit', 'mph', 'regulatory sign', 'warning sign', 'guide sign',
            'traffic signal', 'intersection', 'crosswalk', 'pavement marking',
            'lane marking', 'school zone', 'work zone', 'sight distance',
            'clearance interval', 'signal timing', 'mutcd'
        ]
        
        for term in teppl_terms:
            if term in all_text:
                terms.add(term)
        
        # Technical measurements
        measurements = re.findall(r'\d+\.?\d*\s*(?:inch|inches|feet|ft|mph|degrees?)', all_text)
        terms.update(measurements)
        
        return sorted(list(terms))

    def _extract_dimensions(self, text_context: Dict[str, str]) -> List[str]:
        """Extract dimensional information from text context"""
        all_text = " ".join(text_context.values())
        
        dimension_patterns = [
            r'\d+\.?\d*\s*(?:inch|inches|in\.?)',
            r'\d+\.?\d*\s*(?:feet|foot|ft\.?)',
            r'\d+\.?\d*\s*x\s*\d+\.?\d*\s*(?:inch|inches|feet|ft)',
            r'\d+\.?\d*\s*(?:mm|cm|meters?|m)',
            r'(?:width|height|diameter|radius)\s*:?\s*\d+\.?\d*'
        ]
        
        dimensions = []
        for pattern in dimension_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            dimensions.extend(matches)
        
        return sorted(list(set(dimensions)))

    def _extract_regulatory_refs(self, text_context: Dict[str, str]) -> List[str]:
        """Extract regulatory references from text context"""
        all_text = " ".join(text_context.values())
        
        ref_patterns = [
            r'\bMUTCD\s*(?:Section\s*)?\d*[A-Z]?(?:\.\d+)*\b',
            r'\b(?:Section\s*)?\d+[A-Z]?(?:\.\d+)*\b',
            r'\bCFR\s*\d+(?:\.\d+)*\b',
            r'\bUSC\s*\d+(?:\.\d+)*\b'
        ]
        
        refs = []
        for pattern in ref_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            refs.extend(matches)
        
        return sorted(list(set(refs)))

    def _categorize_for_teppl(self, drawing_type: str, text_context: Dict[str, str]) -> str:
        """Categorize drawing for TEPPL classification"""
        
        # Map drawing types to TEPPL categories
        teppl_mapping = {
            'signage': 'Traffic Control Devices',
            'traffic_control': 'Traffic Signals and Control',
            'pavement_markings': 'Pavement Markings and Delineation',
            'geometric_design': 'Geometric Design Standards'
        }
        
        return teppl_mapping.get(drawing_type, 'General Technical Diagram')

    def _extract_figure_number(self, text_context: Dict[str, str]) -> Optional[str]:
        """Extract figure number from text context"""
        all_text = " ".join(text_context.values())
        
        figure_patterns = [
            r'(?:Figure|Fig\.?)\s+(\d+[A-Z]?(?:-\d+)?)',
            r'(?:Exhibit|Diagram)\s+(\d+[A-Z]?(?:-\d+)?)'
        ]
        
        for pattern in figure_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def _extract_caption(self, text_context: Dict[str, str]) -> Optional[str]:
        """Extract figure caption from text context"""
        all_text = " ".join(text_context.values())
        
        # Look for text following figure numbers
        caption_patterns = [
            r'(?:Figure|Fig\.?)\s+\d+[A-Z]?(?:-\d+)?[:.]?\s*([^.]+\.?)',
            r'(?:Exhibit|Diagram)\s+\d+[A-Z]?(?:-\d+)?[:.]?\s*([^.]+\.?)'
        ]
        
        for pattern in caption_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _generate_drawing_citation(self, doc_metadata: Dict[str, Any], page_num: int, img_index: int) -> str:
        """Generate formal citation for the drawing"""
        
        teppl_meta = doc_metadata.get('teppl_metadata', {})
        doc_title = teppl_meta.get('official_name', 'Unknown Document')
        topic = teppl_meta.get('teppl_topic', 'Unknown')
        year = teppl_meta.get('publication_year', 'Unknown')
        
        return f"NCDOT TEPPL {topic}: {doc_title}, Page {page_num}, Figure {img_index + 1} ({year})"

    def _get_spatial_context(self, page, img_info) -> Dict[str, float]:
        """Get spatial context of image on page"""
        try:
            page_rect = page.rect
            img_rect = fitz.Rect(img_info[1:5])
            
            return {
                "x_position": img_rect.x0 / page_rect.width,
                "y_position": img_rect.y0 / page_rect.height,
                "width_ratio": img_rect.width / page_rect.width,
                "height_ratio": img_rect.height / page_rect.height,
                "center_x": (img_rect.x0 + img_rect.x1) / 2 / page_rect.width,
                "center_y": (img_rect.y0 + img_rect.y1) / 2 / page_rect.height
            }
        except:
            return {}

    def _link_drawings_to_text(
        self,
        drawings: List[Dict[str, Any]],
        page,
        text_chunks: List[Dict[str, Any]],
        page_number: int
    ) -> List[Dict[str, Any]]:
        """Link drawings to relevant text chunks on the same page"""
        
        # Find text chunks from the same page
        page_chunks = [
            chunk for chunk in text_chunks 
            if chunk.get('metadata', {}).get('page_number') == page_number
        ]
        
        for drawing in drawings:
            linked_chunk_ids = []
            
            # Get drawing's technical terms and context
            drawing_terms = set(drawing.get('technical_terms', []))
            drawing_context = drawing.get('text_context', {}).get('raw_text', '').lower()
            
            # Score each text chunk for relevance
            chunk_scores = []
            
            for chunk in page_chunks:
                score = 0
                chunk_content = chunk.get('content', '').lower()
                chunk_terms = set(chunk.get('metadata', {}).get('technical_terms', []))
                
                # Term overlap score
                term_overlap = len(drawing_terms.intersection(chunk_terms))
                score += term_overlap * 3
                
                # Content similarity (simple word overlap)
                drawing_words = set(drawing_context.split())
                chunk_words = set(chunk_content.split())
                word_overlap = len(drawing_words.intersection(chunk_words))
                score += word_overlap * 1
                
                # Spatial proximity bonus (if both are on same page)
                if drawing.get('page_number') == chunk.get('metadata', {}).get('page_number'):
                    score += 2
                
                chunk_scores.append((chunk, score))
            
            # Link to top-scoring chunks (threshold = 3)
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            for chunk, score in chunk_scores[:3]:  # Top 3 chunks
                if score >= 3:  # Minimum relevance threshold
                    linked_chunk_ids.append(chunk.get('metadata', {}).get('chunk_id'))
            
            drawing['linked_chunks'] = linked_chunk_ids
        
        return drawings

    def _create_bidirectional_links(
        self,
        text_chunks: List[Dict[str, Any]],
        drawings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create bidirectional links between text chunks and drawings"""
        
        enhanced_chunks = []
        
        for chunk in text_chunks:
            chunk_id = chunk.get('metadata', {}).get('chunk_id')
            associated_drawings = []
            
            # Find drawings that link to this chunk
            for drawing in drawings:
                if chunk_id in drawing.get('linked_chunks', []):
                    associated_drawings.append({
                        'drawing_id': drawing['drawing_id'],
                        'drawing_type': drawing['drawing_type'],
                        'thumbnail_path': drawing.get('thumbnail_path'),
                        'file_path': drawing['file_path'],
                        'caption': drawing.get('citation_info', {}).get('caption'),
                        'figure_number': drawing.get('citation_info', {}).get('figure_number')
                    })
            
            # Add drawing associations to chunk metadata
            enhanced_chunk = chunk.copy()
            enhanced_chunk['metadata']['associated_drawings'] = associated_drawings
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

# Integration function for existing workflow
def integrate_multimodal_processing(
    pdf_path: str,
    text_chunks: List[Dict[str, Any]],
    doc_metadata: Dict[str, Any],
    documents_path: str = "./documents"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convenience function to integrate multimodal processing into existing workflow.
    
    Returns:
        Tuple[drawings, enhanced_text_chunks]
    """
    processor = MultimodalDrawingProcessor(documents_path)
    return processor.extract_and_link_drawings(pdf_path, text_chunks, doc_metadata)