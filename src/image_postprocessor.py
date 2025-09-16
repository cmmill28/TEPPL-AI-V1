# src/image_post_processor.py

"""
Image Post-Processor for TEPPL System - Enhanced Version
Filters extracted images to keep only those with highly relevant visual content
Generates high-quality thumbnails and moves curated images to documents/images
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageFilter, ImageStat
import numpy as np
import cv2
import hashlib
import json
from datetime import datetime
import traceback

# Try to import advanced image analysis
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TEPPLImagePostProcessor:
    """
    Post-processes extracted images to filter for meaningful visual content
    and prepares them for the web interface with enhanced filtering
    """
    
    def __init__(
        self,
        extracted_images_dir: str = "./storage/extracted_images",
        output_images_dir: str = "./documents/images",
        thumbnails_dir: str = "./documents/thumbnails",
        metadata_file: str = "./documents/image_metadata.json"
    ):
        self.extracted_dir = Path(extracted_images_dir)
        self.output_dir = Path(output_images_dir)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.metadata_file = Path(metadata_file)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced visual content detection thresholds
        self.min_dimensions = (120, 120)  # Increased minimum width/height
        self.max_dimensions = (2000, 2000)  # Maximum to avoid huge images
        self.min_complexity_score = 20  # Increased minimum visual complexity
        self.min_content_diversity = 0.25  # Color/texture diversity threshold
        
        # Text vs Object detection thresholds
        self.max_text_area_ratio = 0.7  # Maximum text area before considering "mostly text"
        self.min_edge_density = 0.03  # Minimum edge density for technical content
        self.min_contour_count = 12  # Minimum number of contours for complex objects
        
        # Image metadata storage
        self.processed_metadata = {}
        self.load_existing_metadata()
        
        logger.info(f"📸 Enhanced Image Post-Processor initialized")
        logger.info(f"   Input: {self.extracted_dir}")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Thumbnails: {self.thumbnails_dir}")

    def load_existing_metadata(self):
        """Load existing processed image metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.processed_metadata = json.load(f)
                logger.info(f"📋 Loaded metadata for {len(self.processed_metadata)} images")
            except Exception as e:
                logger.warning(f"Could not load existing metadata: {e}")
                self.processed_metadata = {}

    def save_metadata(self):
        """Save processed image metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.processed_metadata, f, indent=2, default=str)
            logger.info(f"💾 Saved metadata for {len(self.processed_metadata)} images")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def process_all_images(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process all images in the extracted directory with enhanced filtering
        """
        if not self.extracted_dir.exists():
            logger.error(f"❌ Extracted images directory not found: {self.extracted_dir}")
            return {"error": "Input directory not found"}
        
        # Find all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']:
            image_files.extend(self.extracted_dir.glob(f"**/{ext}"))
        
        logger.info(f"🔍 Found {len(image_files)} image files to process")
        
        results = {
            "total_found": len(image_files),
            "processed": 0,
            "kept": 0,
            "filtered_out": 0,
            "errors": 0,
            "categories": {},
            "filter_reasons": {}
        }
        
        for image_path in image_files:
            try:
                # Skip if already processed and not forcing reprocess
                image_id = self._generate_image_id(image_path)
                if not force_reprocess and image_id in self.processed_metadata:
                    if self.processed_metadata[image_id].get("kept", False):
                        results["kept"] += 1
                    else:
                        results["filtered_out"] += 1
                        reason = self.processed_metadata[image_id].get("analysis", {}).get("reason", "unknown")
                        results["filter_reasons"][reason] = results["filter_reasons"].get(reason, 0) + 1
                    results["processed"] += 1
                    continue
                
                # Analyze image with enhanced filtering
                analysis_result = self._analyze_image_enhanced(image_path)
                
                if analysis_result["keep"]:
                    # Copy to output directory and create high-quality thumbnail
                    success = self._process_keeper_image_enhanced(image_path, analysis_result)
                    if success:
                        results["kept"] += 1
                        category = analysis_result.get("category", "general")
                        results["categories"][category] = results["categories"].get(category, 0) + 1
                    else:
                        results["errors"] += 1
                        logger.error(f"Failed to process keeper image: {image_path}")
                else:
                    results["filtered_out"] += 1
                    reason = analysis_result.get("reason", "unknown")
                    results["filter_reasons"][reason] = results["filter_reasons"].get(reason, 0) + 1
                    
                results["processed"] += 1
                
                # Store metadata with original filename preservation
                self.processed_metadata[image_id] = {
                    "original_path": str(image_path),
                    "original_filename": image_path.name,
                    "analysis": analysis_result,
                    "processed_at": datetime.now().isoformat(),
                    "kept": analysis_result["keep"]
                }
                
                # Periodic save and progress update
                if results["processed"] % 25 == 0:  # More frequent saves
                    self.save_metadata()
                    logger.info(f"📊 Progress: {results['processed']}/{len(image_files)} processed "
                              f"(Kept: {results['kept']}, Filtered: {results['filtered_out']})")
                
            except KeyboardInterrupt:
                logger.info("⏹️ Processing interrupted by user. Saving progress...")
                self.save_metadata()
                break
            except Exception as e:
                logger.error(f"❌ Error processing {image_path}: {e}")
                # Log the full traceback for debugging
                logger.debug(traceback.format_exc())
                results["errors"] += 1
        
        # Final save
        self.save_metadata()
        
        # Log comprehensive results
        logger.info("📊 ENHANCED IMAGE POST-PROCESSING RESULTS")
        logger.info(f"   Total found: {results['total_found']}")
        logger.info(f"   Processed: {results['processed']}")
        logger.info(f"   Kept: {results['kept']} ({results['kept']/max(results['processed'], 1)*100:.1f}%)")
        logger.info(f"   Filtered out: {results['filtered_out']}")
        logger.info(f"   Errors: {results['errors']}")
        logger.info(f"   Categories: {results['categories']}")
        logger.info(f"   Filter reasons: {results['filter_reasons']}")
        
        return results

    def _analyze_image_enhanced(self, image_path: Path) -> Dict[str, Any]:
        """
        Enhanced image analysis with better text vs object detection
        """
        try:
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Basic properties
                width, height = img.size
                aspect_ratio = width / height
                
                # Size filter
                if width < self.min_dimensions[0] or height < self.min_dimensions[1]:
                    return {
                        "keep": False,
                        "reason": "too_small",
                        "dimensions": (width, height)
                    }
                
                if width > self.max_dimensions[0] or height > self.max_dimensions[1]:
                    return {
                        "keep": False,
                        "reason": "too_large", 
                        "dimensions": (width, height)
                    }
                
                # Enhanced text vs object detection
                text_vs_object_result = self._detect_text_vs_object(img)
                
                if not text_vs_object_result["is_object_based"]:
                    return {
                        "keep": False,
                        "reason": "mostly_text",
                        "dimensions": (width, height),
                        "text_analysis": text_vs_object_result
                    }
                
                # Visual complexity analysis (with error handling)
                complexity_score = self._calculate_visual_complexity_safe(img)
                
                # Content diversity analysis
                diversity_score = self._calculate_content_diversity(img)
                
                # Technical content detection
                technical_features = self._detect_technical_content_enhanced(img)
                
                # Pattern detection (charts, diagrams, etc.)
                content_type = self._classify_visual_content_enhanced(img, technical_features)
                
                # Enhanced decision logic
                keep_decision = self._make_enhanced_keep_decision(
                    complexity_score,
                    diversity_score, 
                    technical_features,
                    content_type,
                    text_vs_object_result,
                    (width, height),
                    aspect_ratio
                )
                
                return {
                    "keep": keep_decision["keep"],
                    "reason": keep_decision["reason"],
                    "confidence": keep_decision["confidence"],
                    "dimensions": (width, height),
                    "aspect_ratio": aspect_ratio,
                    "complexity_score": complexity_score,
                    "diversity_score": diversity_score,
                    "technical_features": technical_features,
                    "content_type": content_type,
                    "category": keep_decision.get("category", "general"),
                    "text_vs_object": text_vs_object_result
                }
                
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            logger.debug(traceback.format_exc())
            return {
                "keep": False,
                "reason": "analysis_error",
                "error": str(e)
            }

    def _detect_text_vs_object(self, img: Image.Image) -> Dict[str, Any]:
        """
        Enhanced detection to differentiate text-heavy images from object-based images
        """
        try:
            # Convert to grayscale
            gray = np.array(img.convert('L'))
            height, width = gray.shape
            total_area = width * height
            
            # Edge detection for structural analysis
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Find contours for object detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter meaningful contours
            meaningful_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Minimum area threshold
                    meaningful_contours.append(contour)
            
            contour_count = len(meaningful_contours)
            
            # Text region detection using morphological operations
            # Binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to detect text regions
            text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
            dilated = cv2.dilate(binary, text_kernel, iterations=2)
            
            text_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate text area
            text_area = 0
            text_regions = 0
            for contour in text_contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Heuristic for text regions: wider than tall, reasonable size
                if 1.5 < aspect_ratio < 15 and 800 < area < 100000:
                    text_area += area
                    text_regions += 1
            
            text_area_ratio = text_area / total_area
            
            # Look for table-like structures (grid patterns)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_density = np.sum(horizontal_lines > 0) / horizontal_lines.size
            v_line_density = np.sum(vertical_lines > 0) / vertical_lines.size
            
            has_table_structure = (h_line_density > 0.002 and v_line_density > 0.001)
            
            # Decision logic for object vs text classification
            is_object_based = True
            classification_reason = "object_based"
            
            # Strong text indicators
            if text_area_ratio > self.max_text_area_ratio and text_regions > 8:
                is_object_based = False
                classification_reason = "high_text_density"
            
            # Low structural complexity (likely just text)
            elif edge_density < self.min_edge_density and contour_count < self.min_contour_count:
                is_object_based = False
                classification_reason = "low_structural_complexity"
            
            # Table of contents / list detection
            elif text_regions > 15 and not has_table_structure:
                is_object_based = False
                classification_reason = "list_or_toc"
            
            # Keep tables and structured data
            elif has_table_structure:
                is_object_based = True
                classification_reason = "table_structure"
            
            # Technical drawings with some text are OK
            elif edge_density > 0.05 and contour_count > 20:
                is_object_based = True
                classification_reason = "technical_drawing"
            
            return {
                "is_object_based": is_object_based,
                "classification_reason": classification_reason,
                "text_area_ratio": text_area_ratio,
                "text_regions_count": text_regions,
                "edge_density": edge_density,
                "contour_count": contour_count,
                "has_table_structure": has_table_structure,
                "h_line_density": h_line_density,
                "v_line_density": v_line_density
            }
            
        except Exception as e:
            logger.debug(f"Error in text vs object detection: {e}")
            return {
                "is_object_based": True,  # Default to keeping on error
                "classification_reason": "detection_error",
                "error": str(e)
            }

    def _calculate_visual_complexity_safe(self, img: Image.Image) -> float:
        """
        Calculate visual complexity score with better error handling
        """
        try:
            # Convert to grayscale for analysis
            gray = img.convert('L')
            
            # Convert to numpy for OpenCV operations
            img_array = np.array(gray)
            
            # Edge detection
            edges = cv2.Canny(img_array, 50, 150)
            edge_density = np.mean(edges) / 255.0
            
            # Texture analysis (standard deviation of pixel values)
            texture_variance = np.std(img_array) / 255.0
            
            # Safer Laplacian variance calculation
            try:
                laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
                laplacian_var = np.var(laplacian) / 10000.0
            except Exception as e:
                logger.debug(f"Laplacian calculation failed: {e}")
                laplacian_var = 0.1  # Default value
            
            # Combined complexity score
            complexity = (edge_density * 40) + (texture_variance * 30) + (laplacian_var * 30)
            
            return min(100, complexity)  # Cap at 100
            
        except Exception as e:
            logger.debug(f"Error calculating visual complexity: {e}")
            return 25  # Default moderate complexity

    def _calculate_content_diversity(self, img: Image.Image) -> float:
        """
        Calculate content diversity based on color and spatial distribution
        """
        try:
            # Color diversity
            colors = img.getcolors(maxcolors=256*256*256)
            if colors:
                # Calculate color entropy
                total_pixels = img.width * img.height
                color_probs = [count/total_pixels for count, color in colors]
                color_entropy = -sum(p * np.log2(p) for p in color_probs if p > 0)
                color_diversity = min(1.0, color_entropy / 8.0)  # Normalize
            else:
                color_diversity = 0.5  # Default if too many colors
            
            # Spatial distribution (using image statistics)
            stat = ImageStat.Stat(img)
            
            # Standard deviation across channels
            if hasattr(stat, 'stddev'):
                spatial_variance = np.mean(stat.stddev) / 255.0
            else:
                spatial_variance = 0.3
            
            # Combined diversity score
            diversity = (color_diversity * 0.6) + (spatial_variance * 0.4)
            
            return diversity
            
        except Exception as e:
            logger.debug(f"Error calculating content diversity: {e}")
            return 0.3  # Default moderate diversity

    def _detect_technical_content_enhanced(self, img: Image.Image) -> Dict[str, Any]:
        """
        Enhanced technical content detection with better heuristics
        """
        features = {
            "has_lines": False,
            "has_geometric_shapes": False,
            "has_text_regions": False,
            "is_chart_like": False,
            "line_density": 0.0,
            "has_tables": False,
            "diagram_complexity": "low"
        }
        
        try:
            # Convert to grayscale
            gray = np.array(img.convert('L'))
            
            # Enhanced line detection using HoughLinesP for better accuracy
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
            
            if lines is not None and len(lines) > 15:
                features["has_lines"] = True
                features["line_density"] = min(1.0, len(lines) / 150.0)
                
                if len(lines) > 50:
                    features["diagram_complexity"] = "high"
                elif len(lines) > 25:
                    features["diagram_complexity"] = "medium"
            
            # Enhanced geometric shape detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            geometric_shapes = 0
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it looks like a geometric shape
                area = cv2.contourArea(contour)
                if 3 <= len(approx) <= 12 and area > 300:
                    geometric_shapes += 1
            
            features["has_geometric_shapes"] = geometric_shapes > 5
            
            # Enhanced table detection
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_density = np.sum(horizontal_lines > 0) / horizontal_lines.size
            v_line_density = np.sum(vertical_lines > 0) / vertical_lines.size
            
            features["is_chart_like"] = (h_line_density > 0.002 and v_line_density > 0.001)
            features["has_tables"] = (h_line_density > 0.003 and v_line_density > 0.002)
            
            # Text region detection (simplified)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(binary, kernel, iterations=2)
            
            text_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = 0
            for contour in text_contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Text regions tend to be wider than tall and of reasonable size
                if 2 < aspect_ratio < 10 and 500 < area < 50000:
                    text_regions += 1
            
            features["has_text_regions"] = text_regions > 3
            
        except Exception as e:
            logger.debug(f"Error detecting enhanced technical content: {e}")
        
        return features

    def _classify_visual_content_enhanced(self, img: Image.Image, technical_features: Dict) -> str:
        """
        Enhanced visual content classification using technical features
        """
        try:
            # Basic classification based on image properties
            width, height = img.size
            aspect_ratio = width / height
            
            # Get image statistics
            stat = ImageStat.Stat(img)
            
            # Color analysis
            if len(stat.mean) >= 3:
                r_mean, g_mean, b_mean = stat.mean[:3]
                color_variance = np.var(stat.mean[:3])
            else:
                r_mean = g_mean = b_mean = stat.mean[0] if stat.mean else 128
                color_variance = 0
            
            # Enhanced classification using technical features
            if technical_features.get("has_tables", False):
                return "data_table"
            
            elif technical_features.get("is_chart_like", False):
                return "chart_or_graph"
            
            elif (technical_features.get("has_lines", False) and 
                  technical_features.get("line_density", 0) > 0.4):
                return "technical_drawing"
            
            elif technical_features.get("has_geometric_shapes", False):
                return "engineering_diagram"
            
            # Fallback to original classification
            elif aspect_ratio > 3 or aspect_ratio < 0.3:
                return "banner_or_decorative"
            
            elif color_variance < 100:  # Low color variation
                if max(r_mean, g_mean, b_mean) > 240:  # Very light
                    return "diagram_or_schematic"
                else:
                    return "simple_graphic"
            
            elif aspect_ratio > 1.3 and aspect_ratio < 2.0:
                return "chart_or_graph"
            
            elif 0.8 <= aspect_ratio <= 1.2:  # Square-ish
                return "technical_diagram"
            
            return "general_figure"
            
        except Exception as e:
            logger.debug(f"Error classifying visual content: {e}")
            return "unknown"

    def _make_enhanced_keep_decision(
        self,
        complexity: float,
        diversity: float,
        technical_features: Dict,
        content_type: str,
        text_vs_object: Dict,
        dimensions: Tuple[int, int],
        aspect_ratio: float
    ) -> Dict[str, Any]:
        """
        Enhanced decision making with better text vs object filtering
        """
        keep = False
        confidence = 0.0
        reason = "filtered_out"
        category = "general"
        
        # Strong keep signals for technical content
        if technical_features.get("has_tables", False):
            keep = True
            confidence = 0.95
            reason = "data_table"
            category = "data_tables"
            
        elif content_type == "data_table":
            keep = True
            confidence = 0.9
            reason = "structured_data"
            category = "data_tables"
            
        elif (technical_features.get("has_lines", False) and 
              technical_features.get("line_density", 0) > 0.4):
            keep = True
            confidence = 0.9
            reason = "technical_drawing"
            category = "technical_drawings"
            
        elif technical_features.get("is_chart_like", False):
            keep = True
            confidence = 0.85
            reason = "chart_or_graph"
            category = "charts_graphs"
            
        elif content_type in ["engineering_diagram", "technical_drawing"]:
            keep = True
            confidence = 0.8
            reason = "engineering_content"
            category = "technical_diagrams"
            
        # Moderate keep signals
        elif (complexity > self.min_complexity_score and 
              diversity > self.min_content_diversity and
              text_vs_object.get("is_object_based", False)):
            if content_type not in ["banner_or_decorative", "simple_graphic"]:
                keep = True
                confidence = 0.7
                reason = "sufficient_complexity"
                category = "general_figures"
        
        # Strong filter signals
        elif not text_vs_object.get("is_object_based", True):
            keep = False
            confidence = 0.9
            reason = f"mostly_text_{text_vs_object.get('classification_reason', 'unknown')}"
            
        elif content_type == "banner_or_decorative":
            keep = False
            confidence = 0.9
            reason = "decorative_content"
            
        elif complexity < 10:
            keep = False
            confidence = 0.8
            reason = "too_simple"
            
        elif diversity < 0.15:
            keep = False
            confidence = 0.8
            reason = "low_content_diversity"
        
        # Edge cases
        elif aspect_ratio > 6 or aspect_ratio < 0.15:
            keep = False
            confidence = 0.85
            reason = "extreme_aspect_ratio"
        
        return {
            "keep": keep,
            "confidence": confidence,
            "reason": reason,
            "category": category
        }

    def _process_keeper_image_enhanced(self, image_path: Path, analysis: Dict) -> bool:
        """
        Process an image that was decided to be kept with enhanced naming and thumbnails
        """
        try:
            # Generate readable output filename preserving original structure
            output_filename = self._generate_readable_filename(image_path)
            output_path = self.output_dir / output_filename
            
            # Generate image ID for metadata
            image_id = self._generate_image_id(image_path)
            
            # Load and save as PNG (standardizes format)
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'P'):
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save optimized version
                img.save(output_path, 'PNG', optimize=True)
                
                # Generate high-quality thumbnail
                thumbnail_filename = f"{output_filename.stem}_thumb.jpg"
                thumbnail_path = self.thumbnails_dir / thumbnail_filename
                self._create_high_quality_thumbnail(img, thumbnail_path)
            
            # Update metadata with file paths and original filename
            if image_id not in self.processed_metadata:
                self.processed_metadata[image_id] = {}
                
            self.processed_metadata[image_id].update({
                "output_path": str(output_path),
                "thumbnail_path": str(thumbnail_path),
                "web_path": f"images/{output_filename}",
                "thumbnail_web_path": f"thumbnails/{thumbnail_filename}",
                "readable_filename": output_filename.name,
                "original_filename": image_path.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing keeper image {image_path}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _generate_readable_filename(self, image_path: Path) -> Path:
        """
        Generate a readable filename that preserves original structure with uniqueness
        """
        try:
            # Extract base components from original filename
            original_stem = image_path.stem
            original_suffix = image_path.suffix.lower()
            
            # Generate short hash for uniqueness
            file_stat = image_path.stat()
            unique_string = f"{original_stem}_{file_stat.st_size}_{int(file_stat.st_mtime)}"
            hash_suffix = hashlib.md5(unique_string.encode()).hexdigest()[:8]
            
            # Create readable filename: original_name_hash.png
            if len(original_stem) > 50:  # Truncate very long names
                original_stem = original_stem[:47] + "..."
                
            readable_name = f"{original_stem}_{hash_suffix}.png"  # Standardize to PNG
            
            return Path(readable_name)
            
        except Exception as e:
            logger.debug(f"Error generating readable filename for {image_path}: {e}")
            # Fallback to hash-based naming
            hash_name = hashlib.md5(str(image_path).encode()).hexdigest()[:16]
            return Path(f"{hash_name}.png")

    def _create_high_quality_thumbnail(self, img: Image.Image, thumbnail_path: Path, size: Tuple[int, int] = (400, 400)):
        """
        Create a high-quality thumbnail for better visibility
        """
        try:
            # Create a copy for thumbnail processing
            thumb_img = img.copy()
            
            # Maintain aspect ratio while fitting in size
            thumb_img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save as high-quality JPEG
            thumb_img.save(thumbnail_path, 'JPEG', quality=95, optimize=True)
            
        except Exception as e:
            logger.error(f"Error creating high-quality thumbnail {thumbnail_path}: {e}")

    def _generate_image_id(self, image_path: Path) -> str:
        """
        Generate a unique ID for an image based on path and content
        """
        # Use path and file size for uniqueness
        try:
            file_stat = image_path.stat()
            unique_string = f"{image_path.stem}_{file_stat.st_size}_{file_stat.st_mtime}"
            return hashlib.md5(unique_string.encode()).hexdigest()[:16]
        except:
            # Fallback to just filename
            return hashlib.md5(str(image_path).encode()).hexdigest()[:16]

    def get_processed_images_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for all processed images that were kept
        """
        kept_images = {
            image_id: metadata
            for image_id, metadata in self.processed_metadata.items()
            if metadata.get("kept", False)
        }
        
        return {
            "total_kept": len(kept_images),
            "images": kept_images,
            "last_processed": datetime.now().isoformat()
        }

    def cleanup_old_images(self, days_old: int = 30):
        """
        Clean up old processed images (optional maintenance function)
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            removed_count = 0
            for image_id, metadata in list(self.processed_metadata.items()):
                try:
                    processed_date = datetime.fromisoformat(metadata.get("processed_at", ""))
                    if processed_date < cutoff_date:
                        # Remove files if they exist
                        for path_key in ["output_path", "thumbnail_path"]:
                            if path_key in metadata:
                                file_path = Path(metadata[path_key])
                                if file_path.exists():
                                    file_path.unlink()
                        
                        # Remove from metadata
                        del self.processed_metadata[image_id]
                        removed_count += 1
                        
                except Exception as e:
                    logger.debug(f"Error cleaning up {image_id}: {e}")
            
            if removed_count > 0:
                self.save_metadata()
                logger.info(f"🧹 Cleaned up {removed_count} old processed images")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """
    Main function to run enhanced image post-processing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Post-process TEPPL extracted images')
    parser.add_argument('--input', default='./storage/extracted_images',
                       help='Input directory with extracted images')
    parser.add_argument('--output', default='./documents/images',
                       help='Output directory for curated images')
    parser.add_argument('--thumbnails', default='./documents/thumbnails',
                       help='Thumbnails directory')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of all images')
    parser.add_argument('--cleanup', type=int, metavar='DAYS',
                       help='Clean up processed images older than DAYS')
    
    args = parser.parse_args()
    
    # Initialize enhanced processor
    processor = TEPPLImagePostProcessor(
        extracted_images_dir=args.input,
        output_images_dir=args.output,
        thumbnails_dir=args.thumbnails
    )
    
    if args.cleanup:
        processor.cleanup_old_images(args.cleanup)
        return
    
    # Process all images with enhanced filtering
    results = processor.process_all_images(force_reprocess=args.force)
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("📸 ENHANCED IMAGE POST-PROCESSING COMPLETE")
    print("="*70)
    print(f"✅ Kept: {results['kept']} high-quality images")
    print(f"⏭️ Filtered: {results['filtered_out']} low-quality/text images")
    print(f"❌ Errors: {results['errors']}")
    
    if results['categories']:
        print(f"\n📂 Categories:")
        for category, count in results['categories'].items():
            print(f"   {category}: {count}")
    
    if results['filter_reasons']:
        print(f"\n🔍 Filter reasons:")
        for reason, count in results['filter_reasons'].items():
            print(f"   {reason}: {count}")
    
    print(f"\n📁 Output locations:")
    print(f"   Images: {args.output}")
    print(f"   Thumbnails: {args.thumbnails}")
    print(f"   Metadata: ./documents/image_metadata.json")
    print("="*70)


if __name__ == "__main__":
    main()