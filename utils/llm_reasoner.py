"""
LLM Reasoner Module
Uses Google Gemini for intelligent customer ID selection
"""

import google.generativeai as genai
from typing import List, Dict
import json
import logging
import re
import os

logger = logging.getLogger(__name__)


class LLMReasoner:
    """Uses LLM reasoning to select best customer ID candidate"""
    
    def __init__(self, config):
        """Initialize LLM reasoner with configuration"""
        self.config = config
        
        # Initialize Gemini directly (no LangChain)
        try:
            api_key = config.GOOGLE_API_KEY or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No Google API key found")
            
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('models/gemini-2.5-flash')
            logger.info("LLM initialized: Gemini 1.5 Flash (Direct API)")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            self.llm = None
    
    def select_best_candidate(
        self, 
        text: str, 
        candidates: List[Dict],
        document_type: str = 'unknown'
    ) -> Dict:
        """
        Use LLM to select the best customer ID from candidates
        
        Args:
            text: Full document text
            candidates: List of candidate dictionaries
            document_type: Type of document
            
        Returns:
            Best candidate with reasoning
        """
        # If no LLM available, fall back to heuristic selection
        if not self.llm:
            logger.warning("LLM not available, using heuristic selection")
            return self._heuristic_selection(candidates)
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return {
                'id': candidates[0]['id'],
                'confidence': candidates[0]['confidence'],
                'method': candidates[0]['method'],
                'reasoning': 'Only one candidate found'
            }
        
        # If no candidates, return None
        if len(candidates) == 0:
            return {
                'id': None,
                'confidence': 0.0,
                'method': 'No candidates',
                'reasoning': 'No customer ID patterns detected'
            }
        
        try:
            # Use LLM to reason about best candidate
            result = self._llm_selection(text, candidates, document_type)
            return result
        except Exception as e:
            logger.error(f"LLM selection failed: {str(e)}")
            return self._heuristic_selection(candidates)
    
    def _llm_selection(
        self, 
        text: str, 
        candidates: List[Dict],
        document_type: str
    ) -> Dict:
        """
        Use LLM to intelligently select best candidate
        
        Args:
            text: Document text
            candidates: List of candidates
            document_type: Document type
            
        Returns:
            Selected candidate with reasoning
        """
        # Prepare candidate summary
        candidate_summary = []
        for i, cand in enumerate(candidates[:5], 1):  # Limit to top 5
            candidate_summary.append(
                f"{i}. ID: '{cand['id']}' | Method: {cand['method']} | "
                f"Confidence: {cand['confidence']:.2f} | Context: {cand.get('context', 'N/A')}"
            )
        
        # Truncate text if too long (keep first 2000 chars)
        text_sample = text[:2000] + "..." if len(text) > 2000 else text
        
        # Create prompt
        prompt = f"""You are an expert at analyzing documents and identifying customer IDs.
Your task is to select the most likely customer ID from the given candidates.

Consider:
1. Proximity to keywords like "Customer ID", "Client Number", etc.
2. Format and structure of the ID
3. Position in the document (customer IDs often appear in headers)
4. Document type context
5. Avoid selecting location names, business names, or common words

Document Type: {document_type}

Document Text Sample:
{text_sample}

Candidate Customer IDs:
{chr(10).join(candidate_summary)}

Which candidate is most likely the actual customer ID? 

Respond ONLY with a JSON object in this exact format:
{{
    "selected_id": "the actual ID string",
    "confidence": 0.85,
    "reasoning": "brief explanation of why this is the customer ID"
}}"""
        
        # Call Gemini API directly
        response = self.llm.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response_text)
            
            # Validate response
            if 'selected_id' in result and 'confidence' in result and 'reasoning' in result:
                return {
                    'id': result['selected_id'],
                    'confidence': float(result['confidence']),
                    'method': 'LLM Reasoning (Gemini)',
                    'reasoning': result['reasoning']
                }
            else:
                raise ValueError("Invalid JSON structure")
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            logger.debug(f"LLM response was: {response_text}")
            return self._heuristic_selection(candidates)
    
    def _heuristic_selection(self, candidates: List[Dict]) -> Dict:
        """
        Fallback heuristic selection when LLM unavailable
        
        Args:
            candidates: List of candidates
            
        Returns:
            Best candidate based on confidence score
        """
        if not candidates:
            return {
                'id': None,
                'confidence': 0.0,
                'method': 'No candidates',
                'reasoning': 'No customer ID patterns detected'
            }
        
        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        best = candidates[0]
        
        return {
            'id': best['id'],
            'confidence': best['confidence'],
            'method': best['method'] + ' (Heuristic)',
            'reasoning': f"Selected based on highest confidence score from {best['method']}"
        }
    
    def validate_customer_id(self, customer_id: str, text: str) -> float:
        """
        Validate if extracted ID is actually a customer ID using LLM
        
        Args:
            customer_id: Extracted ID
            text: Document text
            
        Returns:
            Validation confidence score
        """
        if not self.llm:
            return 0.5  # Neutral if no LLM
        
        try:
            prompt = f"""Is "{customer_id}" likely a customer ID in this document context?

Document snippet:
{text[:500]}

Respond with ONLY a number between 0.0 and 1.0 indicating confidence that this is a customer ID.
1.0 = definitely a customer ID
0.0 = definitely NOT a customer ID"""
            
            response = self.llm.generate_content(prompt)
            
            # Extract number from response
            match = re.search(r'0?\.\d+|[01]\.0', response.text)
            if match:
                return float(match.group())
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return 0.5