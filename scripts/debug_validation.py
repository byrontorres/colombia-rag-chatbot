#!/usr/bin/env python3
"""
Debug the validation function specifically.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.data_extractor import WikipediaExtractor
from app.services.text_processor import TextProcessor
from app.utils.validators import validate_content_length, clean_text


def debug_validation():
    """Debug the validation step specifically."""
    
    print("DEBUGGING VALIDATION")
    print("=" * 30)
    
    # Get processed text
    extractor = WikipediaExtractor()
    documents = extractor.extract_colombia_content(use_cache=True, sections=None)
    doc = documents[0]
    
    processor = TextProcessor()
    
    # Process step by step
    text = doc.content
    text = processor._apply_patterns(text, processor.html_patterns)
    text = processor._apply_patterns(text, processor.reference_patterns)
    text = processor._remove_unwanted_content(text)
    text = processor._normalize_unicode(text)
    text = processor._apply_patterns(text, processor.whitespace_patterns)
    processed_text = clean_text(text)
    
    print(f"Processed text length: {len(processed_text)}")
    print(f"Processed text stripped length: {len(processed_text.strip())}")
    print(f"Is processed text empty?: {not processed_text}")
    print(f"Is processed text stripped empty?: {not processed_text.strip()}")
    
    # Test validation function directly
    result = validate_content_length(processed_text, 100)
    print(f"Validation result: {result}")
    
    # Manual validation
    manual_check = len(processed_text.strip()) >= 100
    print(f"Manual validation (len >= 100): {manual_check}")
    
    # Test with smaller string to ensure function works
    test_small = "This is a small test string."
    test_large = "A" * 200
    
    print(f"Small string validation: {validate_content_length(test_small, 100)}")  # Should be False
    print(f"Large string validation: {validate_content_length(test_large, 100)}")  # Should be True
    
    # Check first 200 chars of processed text
    print(f"First 200 chars: '{processed_text[:200]}'")
    
    # Check if there are any invisible characters
    print(f"Text starts with: {repr(processed_text[:50])}")


if __name__ == "__main__":
    debug_validation()