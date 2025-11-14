"""
Configuration file for Customer ID Extractor
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings for the extractor"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    GOOGLE_CLOUD_VISION_API_KEY = os.getenv('GOOGLE_CLOUD_VISION_API_KEY', '')
    
    # Paths
    INPUT_FOLDER = 'data/input_pdfs'
    OUTPUT_CSV = 'data/output/extracted_customer_ids.csv'
    TEST_DATASET_FOLDER = 'test_dataset/sample_pdfs'
    
    # Tesseract Configuration (Windows path - adjust for your system)
    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # For Mac/Linux, leave as None or set to: '/usr/local/bin/tesseract'
    
    # Poppler Configuration (for pdf2image on Windows)
    POPPLER_PATH = r'C:\Users\gdhru\poppler-24.08.0\Library\bin'
    # For Mac/Linux, leave as None
    
    # OCR Settings
    OCR_ENGINE = 'tesseract'  # Options: 'tesseract', 'google_vision', 'both'
    OCR_LANGUAGE = 'eng'  # Language for OCR
    OCR_DPI = 300  # DPI for PDF to image conversion
    
    # Image Preprocessing
    PREPROCESS_IMAGE = True
    APPLY_DENOISING = True
    APPLY_THRESHOLDING = True
    
    # Pattern Matching
    CUSTOMER_ID_PATTERNS = [
        # Common customer ID patterns
        r'(?:customer|client|account|member|cust)[:\s#-]*([A-Z0-9]{4,15})',
        r'\b(?:CID|CUSTID|ACCT)[-:\s]*([A-Z0-9]{4,15})\b',
        r'\b[C|A][0-9]{5,10}\b',  # C12345 or A12345 format
        r'\b\d{6,12}\b',  # Pure numeric IDs
    ]
    
    # Keywords that indicate customer ID proximity
    CUSTOMER_ID_KEYWORDS = [
        'customer id', 'customer no', 'customer number', 'customer code',
        'client id', 'client no', 'client number',
        'account number', 'account no', 'account id',
        'member id', 'member number',
        'cust id', 'cust no',
        'cid', 'customer#'
    ]
    
    # LLM Settings
    LLM_MODEL = 'gpt-3.5-turbo'  # Options: 'gpt-3.5-turbo', 'gpt-4'
    LLM_TEMPERATURE = 0.1  # Low temperature for consistent outputs
    LLM_MAX_TOKENS = 500
    
    # Confidence Score Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.5
    LOW_CONFIDENCE_THRESHOLD = 0.3
    
    # Processing Settings
    MAX_PAGES_PER_PDF = 10  # Limit pages to process per PDF
    ENABLE_PARALLEL_PROCESSING = False  # Set to True for faster processing
    
    # Logging
    LOG_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check API keys
        if not cls.OPENAI_API_KEY and cls.LLM_MODEL.startswith('gpt'):
            errors.append("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        # Check Tesseract installation
        if cls.OCR_ENGINE in ['tesseract', 'both']:
            if not os.path.exists(cls.TESSERACT_CMD) if cls.TESSERACT_CMD else True:
                errors.append(f"Tesseract not found at: {cls.TESSERACT_CMD}")
        
        # Check input folder
        if not os.path.exists(cls.INPUT_FOLDER):
            print(f"Warning: Input folder not found: {cls.INPUT_FOLDER}")
            print("Creating folder...")
            os.makedirs(cls.INPUT_FOLDER, exist_ok=True)
        
        if errors:
            print("\nConfiguration Errors:")
            for error in errors:
                print(f"  - {error}")
            print("\nPlease fix these errors before running the extractor.\n")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\nCurrent Configuration:")
        print("-" * 50)
        print(f"OCR Engine: {cls.OCR_ENGINE}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Input Folder: {cls.INPUT_FOLDER}")
        print(f"Output CSV: {cls.OUTPUT_CSV}")
        print(f"Tesseract Path: {cls.TESSERACT_CMD}")
        print(f"Poppler Path: {cls.POPPLER_PATH}")
        print("-" * 50 + "\n")


# Validate configuration on import
if __name__ == "__main__":
    Config.print_config()
    if Config.validate():
        print("✓ Configuration valid!")
    else:
        print("✗ Configuration has errors!")