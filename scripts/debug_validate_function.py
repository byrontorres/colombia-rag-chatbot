#!/usr/bin/env python3
"""
Debug the validate_content_length function specifically.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import re
import unicodedata
from app.services.data_extractor import WikipediaExtractor
from app.services.text_processor import TextProcessor


def debug_clean_text_function(text):
    """Debug the clean_text function step by step."""
    
    print("DEBUGGING clean_text FUNCTION")
    print("=" * 40)
    
    print(f"Input text length: {len(text)}")
    print(f"Input preview: {text[:100]}...")
    
    # Step 1: Check if empty
    if not text:
        print("Text is empty, returning empty string")
        return ""
    
    # Step 2: Normalize unicode
    text_normalized = unicodedata.normalize('NFKC', text)
    print(f"After unicode normalize: {len(text_normalized)} chars")
    
    # Step 3: Remove extra whitespace
    text_whitespace = re.sub(r'\s+', ' ', text_normalized)
    print(f"After whitespace normalize: {len(text_whitespace)} chars")
    
    # Step 4: Remove control characters only
    text_cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text_whitespace)
    print(f"After control char removal: {len(text_cleaned)} chars")
    
    # Step 5: Strip
    text_final = text_cleaned.strip()
    print(f"After strip: {len(text_final)} chars")
    
    print(f"Final preview: {text_final[:100]}...")
    
    return text_final


def debug_validate_content_length(content, min_length=50, max_length=100000):
    """Debug validate_content_length step by step."""
    
    print("\nDEBUGGING validate_content_length FUNCTION")
    print("=" * 50)
    
    print(f"Input content length: {len(content)}")
    print(f"Min length: {min_length}")
    print(f"Max length: {max_length}")
    
    # Step 1: Check if content exists
    if not content:
        print("Content is falsy, returning False")
        return False
    
    # Step 2: Clean the text
    cleaned_content = debug_clean_text_function(content)
    
    # Step 3: Get length
    content_length = len(cleaned_content)
    print(f"Cleaned content length: {content_length}")
    
    # Step 4: Check bounds
    result = min_length <= content_length <= max_length
    print(f"Length check: {min_length} <= {content_length} <= {max_length} = {result}")
    
    return result


def main():
    # Get text to test
    extractor = WikipediaExtractor()
    documents = extractor.extract_colombia_content(use_cache=True, sections=None)
    doc = documents[0]
    
    processor = TextProcessor()
    
    # Process text
    text = doc.content
    text = processor._apply_patterns(text, processor.html_patterns)
    text = processor._apply_patterns(text, processor.reference_patterns)
    text = processor._remove_unwanted_content(text)
    text = processor._normalize_unicode(text)
    text = processor._apply_patterns(text, processor.whitespace_patterns)
    
    # Debug the validation
    result = debug_validate_content_length(text, 100)
    print(f"\nFINAL RESULT: {result}")


if __name__ == "__main__":
    main()