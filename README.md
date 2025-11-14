Customer ID Extractor - KPMG Technical Assignment
================================================

This project extracts customer IDs from image-based PDFs using OCR and AI.

Author: Pulkit Gupta
Date: October 2025

How It Works
------------

The system processes PDFs in 4 stages:

1. PDF → Images (converts each page to PNG)
2. OCR (extracts text using Tesseract)
3. Pattern Matching (finds potential customer IDs using regex)
4. LLM Selection (uses Google Gemini to pick the best candidate)

Project Structure
-----------------

customer_id_extractor/
├── cid_extractor.py          # main script
├── config.py                 # settings and API keys
├── requirements.txt          # dependencies
├── README.md                 # this file
├── .env                      # API keys (not included)
├── data/
│   ├── input/                # put your PDFs here
│   ├── output/               # results CSV goes here
│   └── temp/                 # temporary images (auto-created)
├── logs/                     # log files (auto-created)
└── utils/
    ├── pdf_processor.py      # PDF to image conversion
    ├── ocr_engine.py         # Tesseract OCR
    ├── pattern_matcher.py    # regex pattern matching
    └── llm_reasoner.py       # Google Gemini integration

Installation
------------

1. Install Python dependencies:
   pip install -r requirements.txt

2. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: brew install tesseract
   - Linux: sudo apt-get install tesseract-ocr

3. Install Poppler (for PDF processing):
   - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
   - Mac: brew install poppler
   - Linux: sudo apt-get install poppler-utils

4. Get Google Gemini API key:
   - Go to https://aistudio.google.com/app/apikey
   - Create API key
   - Add to .env file: GEMINI_API_KEY=your_key_here

Usage
-----

1. Put your PDF files in data/input/
2. Make sure .env file has your API key
3. Run: python cid_extractor.py
4. Check results in data/output/extraction_results.csv

The script processes all PDFs in the input folder and creates extracted_customer_ids.csv with:
- File Name
- Page Number
- Extracted Customer ID
- Confidence Score (0-1)
- Extraction Method (LLM/Pattern Matching/Failed)
- Notes/Discrepancies (detailed reasoning)

Results
-------

Tested on 20 diverse documents:
- Extraction Rate: 60% (12/20 documents)
- Accuracy: 40% (8/20 correct)
- LLM Usage: 11/20 documents used Gemini successfully

The system works well on clean, structured documents. It struggles with:
- Poor image quality / blurry scans
- Handwritten IDs
- Non-standard formats
- Documents without clear customer ID labels

Key Features
------------

- Multi-stage pipeline with fallback mechanisms
- Intelligent LLM reasoning with detailed explanations
- Confidence scoring for transparency
- Comprehensive error handling
- Detailed logging for debugging

Technical Decisions
-------------------

1. Chose Tesseract over cloud OCR services (Google Vision, AWS Textract) because:
   - Free and works offline
   - Good enough for most documents
   - No API rate limits

2. Switched from OpenAI to Google Gemini because:
   - OpenAI free tier ran out quickly
   - Gemini offers 1500 requests/day free
   - Similar performance for this task

3. Used regex patterns before LLM to:
   - Reduce API calls
   - Provide structured candidates for LLM
   - Have fallback if LLM fails

Challenges & Learnings
----------------------

Main challenges I faced:

1. OCR quality - Tesseract struggles with poor scans. Tried preprocessing (grayscale, thresholding) which helped a bit.

2. API quota limits - Hit OpenAI limits during testing, had to switch to Gemini mid-project.

3. Pattern matching - Hard to cover all ID formats. Ended up with 6 different regex patterns.

4. LLM timeouts - Sometimes Gemini takes too long. Added fallback to pattern matching.

Future Improvements
-------------------

If I had more time:

- Try better OCR preprocessing techniques
- Add support for table-based extraction
- Train a custom model on customer ID formats
- Build a web interface for easy testing
- Add batch processing with progress bars

Contact
-------

Pulkit Gupta
gpulkit13@gmail.com
+91 9968120700

Submitted for: KPMG Technical Assignment
Date: October 2025