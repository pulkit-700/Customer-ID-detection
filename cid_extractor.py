"""
Customer ID Extractor - Main Script
Extracts customer IDs from image-based PDFs using OCR, VLM, and LLM reasoning
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging
from datetime import datetime

# Import custom modules
from utils.pdf_processor import PDFProcessor
from utils.ocr_engine import OCREngine
from utils.pattern_matcher import PatternMatcher
from utils.llm_reasoner import LLMReasoner
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CustomerIDExtractor:
    """Main class for extracting customer IDs from PDFs"""
    
    def __init__(self, config: Config):
        """Initialize extractor with configuration"""
        self.config = config
        self.pdf_processor = PDFProcessor(config)
        self.ocr_engine = OCREngine(config)
        self.pattern_matcher = PatternMatcher(config)
        self.llm_reasoner = LLMReasoner(config)
        
        # Results storage
        self.results = []
        
        logger.info("CustomerIDExtractor initialized")
    
    def process_single_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Process a single PDF and extract customer IDs
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extraction results (one per page)
        """
        pdf_name = os.path.basename(pdf_path)
        logger.info(f"Processing: {pdf_name}")
        
        page_results = []
        
        try:
            # Step 1: Convert PDF to images
            images = self.pdf_processor.pdf_to_images(pdf_path)
            logger.info(f"Converted {len(images)} pages to images")
            
            # Step 2: Process each page
            for page_num, image in enumerate(images, start=1):
                logger.info(f"Processing page {page_num}/{len(images)}")
                
                # Step 2a: Apply OCR
                ocr_result = self.ocr_engine.extract_text(image)
                text = ocr_result['text']
                boxes = ocr_result['boxes']  # Bounding boxes for layout
                
                if not text.strip():
                    logger.warning(f"No text extracted from page {page_num}")
                    page_results.append({
                        'file_name': pdf_name,
                        'page_number': page_num,
                        'customer_id': None,
                        'confidence': 0.0,
                        'method': 'OCR Failed',
                        'notes': 'No text could be extracted'
                    })
                    continue
                
                # Step 2b: Pattern matching (regex + heuristics)
                candidates = self.pattern_matcher.find_candidates(
                    text, boxes
                )
                
                logger.info(f"Found {len(candidates)} candidate IDs")
                
                # Step 2c: LLM reasoning for best candidate
                if candidates:
                    best_match = self.llm_reasoner.select_best_candidate(
                        text=text,
                        candidates=candidates,
                        document_type=self._detect_document_type(text)
                    )
                    
                    page_results.append({
                        'file_name': pdf_name,
                        'page_number': page_num,
                        'customer_id': best_match['id'],
                        'confidence': best_match['confidence'],
                        'method': best_match['method'],
                        'notes': best_match['reasoning']
                    })
                else:
                    # No candidates found
                    logger.warning(f"No customer ID candidates found on page {page_num}")
                    page_results.append({
                        'file_name': pdf_name,
                        'page_number': page_num,
                        'customer_id': None,
                        'confidence': 0.0,
                        'method': 'No Match',
                        'notes': 'No customer ID patterns detected'
                    })
            
        except Exception as e:
            logger.error(f"Error processing {pdf_name}: {str(e)}")
            page_results.append({
                'file_name': pdf_name,
                'page_number': 1,
                'customer_id': None,
                'confidence': 0.0,
                'method': 'Error',
                'notes': f'Processing error: {str(e)}'
            })
        
        return page_results
    
    def process_folder(self, input_folder: str, output_csv: str):
        """
        Process all PDFs in a folder
        
        Args:
            input_folder: Path to folder containing PDFs
            output_csv: Path for output CSV file
        """
        # Get all PDF files
        pdf_files = list(Path(input_folder).glob('**/*.pdf'))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {input_folder}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        all_results = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            results = self.process_single_pdf(str(pdf_path))
            all_results.extend(results)
        
        # Save results to CSV
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to: {output_csv}")
        
        # Print summary
        self._print_summary(df)
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type from text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['invoice', 'bill', 'amount due']):
            return 'invoice'
        elif any(word in text_lower for word in ['receipt', 'paid', 'transaction']):
            return 'receipt'
        elif any(word in text_lower for word in ['agreement', 'contract', 'terms']):
            return 'agreement'
        elif any(word in text_lower for word in ['application', 'form', 'applicant']):
            return 'form'
        else:
            return 'unknown'
    
    def _print_summary(self, df: pd.DataFrame):
        """Print extraction summary statistics"""
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total pages processed: {len(df)}")
        print(f"Successful extractions: {df['customer_id'].notna().sum()}")
        print(f"Failed extractions: {df['customer_id'].isna().sum()}")
        print(f"\nAverage confidence: {df['confidence'].mean():.2f}")
        print(f"High confidence (>0.8): {(df['confidence'] > 0.8).sum()}")
        print(f"Medium confidence (0.5-0.8): {((df['confidence'] >= 0.5) & (df['confidence'] <= 0.8)).sum()}")
        print(f"Low confidence (<0.5): {(df['confidence'] < 0.5).sum()}")
        print("\nExtraction Methods:")
        print(df['method'].value_counts())
        print("="*60 + "\n")


def main():
    """Main execution function"""
    print("="*60)
    print("Customer ID Extractor")
    print("Intelligent extraction using OCR, VLM, and LLM")
    print("="*60 + "\n")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/output', exist_ok=True)
    
    # Initialize configuration
    config = Config()
    
    # Initialize extractor
    extractor = CustomerIDExtractor(config)
    
    # Define paths
    input_folder = config.INPUT_FOLDER
    output_csv = config.OUTPUT_CSV
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        logger.error(f"Input folder not found: {input_folder}")
        print(f"\nERROR: Please create the folder '{input_folder}' and add PDF files.")
        return
    
    # Start processing
    start_time = datetime.now()
    logger.info("Starting batch processing...")
    
    extractor.process_folder(input_folder, output_csv)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\nProcessing completed in {duration:.2f} seconds")
    print(f"Output saved to: {output_csv}\n")


if __name__ == "__main__":
    main()