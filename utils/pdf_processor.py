"""
PDF Processor Module
Handles PDF to image conversion and preprocessing
"""

from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF processing and image conversion"""
    
    def __init__(self, config):
        """Initialize PDF processor with configuration"""
        self.config = config
        self.dpi = config.OCR_DPI
        self.poppler_path = config.POPPLER_PATH if hasattr(config, 'POPPLER_PATH') else None
    
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL Image objects (one per page)
        """
        try:
            # Convert PDF to images
            if self.poppler_path:
                images = convert_from_path(
                    pdf_path,
                    dpi=self.dpi,
                    poppler_path=self.poppler_path
                )
            else:
                images = convert_from_path(
                    pdf_path,
                    dpi=self.dpi
                )
            
            # Apply preprocessing if enabled
            if self.config.PREPROCESS_IMAGE:
                images = [self.preprocess_image(img) for img in images]
            
            # Limit number of pages
            max_pages = self.config.MAX_PAGES_PER_PDF
            if len(images) > max_pages:
                logger.warning(f"PDF has {len(images)} pages. Processing only first {max_pages}")
                images = images[:max_pages]
            
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply image preprocessing for better OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply denoising if enabled
            if self.config.APPLY_DENOISING:
                gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Apply adaptive thresholding if enabled
            if self.config.APPLY_THRESHOLDING:
                gray = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11, 2
                )
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(gray)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(1.5)
            
            # Sharpen
            processed_image = processed_image.filter(ImageFilter.SHARPEN)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Error preprocessing image: {str(e)}. Using original.")
            return image
    
    def detect_orientation(self, image: Image.Image) -> float:
        """
        Detect image rotation angle (basic implementation)
        
        Args:
            image: PIL Image object
            
        Returns:
            Rotation angle in degrees
        """
        # This is a placeholder - in production, you might use:
        # - pytesseract.image_to_osd() for orientation detection
        # - Or a dedicated ML model
        try:
            from pytesseract import image_to_osd
            osd = image_to_osd(image)
            # Parse rotation from OSD output
            rotation = 0
            for line in osd.split('\n'):
                if 'Rotate:' in line:
                    rotation = int(line.split(':')[1].strip())
                    break
            return rotation
        except:
            return 0
    
    def rotate_image(self, image: Image.Image, angle: float) -> Image.Image:
        """
        Rotate image by given angle
        
        Args:
            image: PIL Image object
            angle: Rotation angle in degrees
            
        Returns:
            Rotated PIL Image
        """
        if angle == 0:
            return image
        return image.rotate(-angle, expand=True, fillcolor='white')