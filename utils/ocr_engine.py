"""
OCR Engine Module
Handles text extraction from images using Tesseract and optionally Google Vision
"""

import pytesseract
from PIL import Image
import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    """Handles OCR text extraction from images"""
    
    def __init__(self, config):
        """Initialize OCR engine with configuration"""
        self.config = config
        self.engine = config.OCR_ENGINE
        self.language = config.OCR_LANGUAGE
        
        # Set Tesseract path if specified
        if hasattr(config, 'TESSERACT_CMD') and config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        
        logger.info(f"OCR Engine initialized: {self.engine}")
    
    def extract_text(self, image: Image.Image) -> Dict:
        """
        Extract text from image using OCR
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with 'text', 'boxes', and 'confidence'
        """
        try:
            if self.engine == 'tesseract':
                return self._tesseract_ocr(image)
            elif self.engine == 'google_vision':
                return self._google_vision_ocr(image)
            elif self.engine == 'both':
                # Try both and combine results
                tesseract_result = self._tesseract_ocr(image)
                try:
                    google_result = self._google_vision_ocr(image)
                    # Combine or choose best result
                    return self._combine_ocr_results(tesseract_result, google_result)
                except:
                    logger.warning("Google Vision failed, using Tesseract only")
                    return tesseract_result
            else:
                logger.error(f"Unknown OCR engine: {self.engine}")
                return {'text': '', 'boxes': [], 'confidence': 0.0}
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return {'text': '', 'boxes': [], 'confidence': 0.0}
    
    def _tesseract_ocr(self, image: Image.Image) -> Dict:
        """
        Extract text using Tesseract OCR
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=self.language)
            
            # Extract bounding boxes with text and confidence
            boxes = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 0:  # Only confident detections
                    boxes.append({
                        'text': data['text'][i],
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': float(data['conf'][i]) / 100.0
                    })
            
            # Calculate average confidence
            confidences = [b['confidence'] for b in boxes if b['confidence'] > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'text': text,
                'boxes': boxes,
                'confidence': avg_confidence,
                'engine': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            return {'text': '', 'boxes': [], 'confidence': 0.0, 'engine': 'tesseract'}
    
    def _google_vision_ocr(self, image: Image.Image) -> Dict:
        """
        Extract text using Google Cloud Vision API
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with OCR results
        """
        try:
            from google.cloud import vision
            import io
            
            # Initialize client
            client = vision.ImageAnnotatorClient()
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create vision image
            vision_image = vision.Image(content=img_byte_arr)
            
            # Perform text detection
            response = client.text_detection(image=vision_image)
            texts = response.text_annotations
            
            if not texts:
                return {'text': '', 'boxes': [], 'confidence': 0.0, 'engine': 'google_vision'}
            
            # First annotation contains full text
            full_text = texts[0].description
            
            # Extract bounding boxes
            boxes = []
            for text in texts[1:]:  # Skip first (full text)
                vertices = text.bounding_poly.vertices
                boxes.append({
                    'text': text.description,
                    'left': vertices[0].x,
                    'top': vertices[0].y,
                    'width': vertices[1].x - vertices[0].x,
                    'height': vertices[2].y - vertices[0].y,
                    'confidence': 0.9  # Google doesn't provide per-word confidence
                })
            
            return {
                'text': full_text,
                'boxes': boxes,
                'confidence': 0.9,
                'engine': 'google_vision'
            }
            
        except Exception as e:
            logger.error(f"Google Vision OCR failed: {str(e)}")
            raise
    
    def _combine_ocr_results(self, result1: Dict, result2: Dict) -> Dict:
        """
        Combine results from multiple OCR engines
        
        Args:
            result1: First OCR result
            result2: Second OCR result
            
        Returns:
            Combined result (uses higher confidence)
        """
        if result1['confidence'] >= result2['confidence']:
            return result1
        else:
            return result2
    
    def extract_text_near_keyword(
        self, 
        boxes: List[Dict], 
        keyword: str, 
        max_distance: int = 100
    ) -> List[str]:
        """
        Extract text near a specific keyword
        
        Args:
            boxes: List of bounding boxes with text
            keyword: Keyword to search for
            max_distance: Maximum pixel distance from keyword
            
        Returns:
            List of nearby text strings
        """
        nearby_texts = []
        keyword_lower = keyword.lower()
        
        # Find keyword boxes
        keyword_boxes = [
            box for box in boxes 
            if keyword_lower in box['text'].lower()
        ]
        
        if not keyword_boxes:
            return []
        
        # Find boxes near keywords
        for kw_box in keyword_boxes:
            kw_center_x = kw_box['left'] + kw_box['width'] / 2
            kw_center_y = kw_box['top'] + kw_box['height'] / 2
            
            for box in boxes:
                if box == kw_box:
                    continue
                
                box_center_x = box['left'] + box['width'] / 2
                box_center_y = box['top'] + box['height'] / 2
                
                # Calculate distance
                distance = ((kw_center_x - box_center_x)**2 + 
                           (kw_center_y - box_center_y)**2)**0.5
                
                if distance <= max_distance:
                    nearby_texts.append(box['text'])
        
        return nearby_texts
    
    def get_text_in_region(
        self, 
        boxes: List[Dict], 
        x: int, 
        y: int, 
        width: int, 
        height: int
    ) -> str:
        """
        Get all text within a specific region
        
        Args:
            boxes: List of bounding boxes
            x, y: Top-left corner of region
            width, height: Region dimensions
            
        Returns:
            Combined text in region
        """
        region_texts = []
        
        for box in boxes:
            box_x = box['left']
            box_y = box['top']
            
            # Check if box is within region
            if (x <= box_x <= x + width and 
                y <= box_y <= y + height):
                region_texts.append(box['text'])
        
        return ' '.join(region_texts)