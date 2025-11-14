"""
Pattern Matcher Module
Uses regex patterns and heuristics to find customer ID candidates
"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PatternMatcher:
    """Finds customer ID candidates using regex and heuristics"""
    
    def __init__(self, config):
        """Initialize pattern matcher with configuration"""
        self.config = config
        self.patterns = config.CUSTOMER_ID_PATTERNS
        self.keywords = config.CUSTOMER_ID_KEYWORDS
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        
        logger.info(f"Pattern Matcher initialized with {len(self.patterns)} patterns")
    
    def find_candidates(self, text: str, boxes: List[Dict] = None) -> List[Dict]:
        """
        Find all customer ID candidates in text
        
        Args:
            text: OCR extracted text
            boxes: Optional bounding boxes for spatial analysis
            
        Returns:
            List of candidate dictionaries with ID, confidence, and method
        """
        candidates = []
        
        # Method 1: Keyword-based extraction
        keyword_candidates = self._find_by_keywords(text, boxes)
        candidates.extend(keyword_candidates)
        
        # Method 2: Pattern-based extraction
        pattern_candidates = self._find_by_patterns(text)
        candidates.extend(pattern_candidates)
        
        # Method 3: Spatial analysis (if boxes available)
        if boxes:
            spatial_candidates = self._find_by_spatial_analysis(boxes)
            candidates.extend(spatial_candidates)
        
        # Remove duplicates
        candidates = self._deduplicate_candidates(candidates)
        
        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Found {len(candidates)} candidate customer IDs")
        
        return candidates
    
    def _find_by_keywords(self, text: str, boxes: List[Dict] = None) -> List[Dict]:
        """
        Find customer IDs near keywords like "Customer ID", "Client No", etc.
        
        Args:
            text: Full text
            boxes: Optional bounding boxes
            
        Returns:
            List of candidates
        """
        candidates = []
        
        for keyword in self.keywords:
            # Create pattern to find text after keyword
            # Matches: "Customer ID: 12345" or "Customer ID 12345" or "Customer ID:C-12345"
            pattern = rf'{re.escape(keyword)}[\s:]*([A-Z0-9\-/]+)'
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                customer_id = match.group(1).strip()
                
                # Validate ID (not too short, not too long)
                if 3 <= len(customer_id) <= 20:
                    candidates.append({
                        'id': customer_id,
                        'confidence': 0.9,  # High confidence - near keyword
                        'method': f'Keyword: {keyword}',
                        'context': match.group(0)
                    })
        
        return candidates
    
    def _find_by_patterns(self, text: str) -> List[Dict]:
        """
        Find customer IDs matching common patterns
        
        Args:
            text: Full text
            
        Returns:
            List of candidates
        """
        candidates = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.finditer(text)
            
            for match in matches:
                # Get the captured group (the ID itself)
                if match.groups():
                    customer_id = match.group(1).strip()
                else:
                    customer_id = match.group(0).strip()
                
                # Validate ID
                if self._is_valid_customer_id(customer_id):
                    candidates.append({
                        'id': customer_id,
                        'confidence': 0.7,  # Medium confidence - pattern match
                        'method': f'Pattern {i+1}',
                        'context': match.group(0)
                    })
        
        return candidates
    
    def _find_by_spatial_analysis(self, boxes: List[Dict]) -> List[Dict]:
        """
        Find customer IDs based on spatial location in document
        (e.g., top-right corner common for invoices)
        
        Args:
            boxes: Bounding boxes with text
            
        Returns:
            List of candidates
        """
        candidates = []
        
        if not boxes:
            return candidates
        
        # Find document dimensions
        max_x = max(box['left'] + box['width'] for box in boxes)
        max_y = max(box['top'] + box['height'] for box in boxes)
        
        # Check top-right region (common for customer IDs)
        top_right_boxes = [
            box for box in boxes
            if box['left'] > max_x * 0.5 and box['top'] < max_y * 0.3
        ]
        
        for box in top_right_boxes:
            text = box['text']
            
            # Check if looks like an ID
            if self._looks_like_id(text):
                candidates.append({
                    'id': text,
                    'confidence': 0.6,  # Lower confidence - just location
                    'method': 'Spatial: Top-right',
                    'context': f"Position: ({box['left']}, {box['top']})"
                })
        
        return candidates
    
    def _is_valid_customer_id(self, text: str) -> bool:
        """
        Check if text looks like a valid customer ID
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid
        """
        # Length check
        if not (3 <= len(text) <= 20):
            return False
        
        # Must contain at least one number
        if not re.search(r'\d', text):
            return False
        
        # Should not be all numbers if very short (might be date/phone)
        if len(text) <= 4 and text.isdigit():
            return False
        
        # Should not be common date formats
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', text):
            return False
        
        # Should not be phone number format
        if re.match(r'^\d{10}$', text) or re.match(r'^\+?\d{10,12}$', text):
            return False
        
        return True
    
    def _looks_like_id(self, text: str) -> bool:
        """
        Quick heuristic check if text looks like an ID
        
        Args:
            text: Text to check
            
        Returns:
            True if looks like ID
        """
        # Clean text
        text = text.strip()
        
        # Check patterns
        # Alphanumeric with possible dashes/slashes
        if re.match(r'^[A-Z0-9\-/]{4,15}$', text, re.IGNORECASE):
            return True
        
        # Starts with letter, has numbers
        if re.match(r'^[A-Z]\d+$', text, re.IGNORECASE) and 4 <= len(text) <= 12:
            return True
        
        return False
    
    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Remove duplicate candidates, keeping highest confidence
        
        Args:
            candidates: List of candidate dictionaries
            
        Returns:
            Deduplicated list
        """
        seen = {}
        
        for candidate in candidates:
            cid = candidate['id']
            
            # Keep candidate with highest confidence
            if cid not in seen or candidate['confidence'] > seen[cid]['confidence']:
                seen[cid] = candidate
        
        return list(seen.values())
    
    def score_candidate(
        self, 
        candidate_id: str, 
        full_text: str, 
        document_type: str = 'unknown'
    ) -> float:
        """
        Score a candidate based on context
        
        Args:
            candidate_id: The ID to score
            full_text: Full document text
            document_type: Type of document
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.5  # Base score
        
        # Boost if near keywords
        for keyword in self.keywords:
            if keyword in full_text.lower():
                # Check proximity
                keyword_pos = full_text.lower().find(keyword)
                id_pos = full_text.find(candidate_id)
                
                if abs(keyword_pos - id_pos) < 50:
                    score += 0.3
                    break
        
        # Boost based on format
        if re.match(r'^[A-Z]\d+$', candidate_id, re.IGNORECASE):
            score += 0.1  # Letter followed by numbers is common
        
        if '-' in candidate_id or '/' in candidate_id:
            score += 0.05  # Structured IDs often have separators
        
        # Document-specific rules
        if document_type == 'invoice':
            # In invoices, customer IDs often in header
            if full_text.index(candidate_id) < len(full_text) * 0.3:
                score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)