#!/usr/bin/env python3
"""
Debug the text processor to see why it removes all content.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.data_extractor import WikipediaExtractor
from app.services.text_processor import TextProcessor


def debug_text_processing():
    """Debug text processing step by step."""
    
    print("DEBUGGING TEXT PROCESSING")
    print("=" * 50)
    
    # Get a document
    extractor = WikipediaExtractor()
    documents = extractor.extract_colombia_content(use_cache=True, sections=None)
    
    if not documents:
        print("ERROR: No documents to process")
        return
    
    doc = documents[0]
    print(f"Original document: {len(doc.content)} chars")
    print(f"Content preview: {doc.content[:200]}...")
    
    # Create processor
    processor = TextProcessor()
    text = doc.content
    
    print(f"\nStep-by-step processing:")
    print(f"0. Original: {len(text)} chars")
    
    # Step 1: HTML cleaning
    text_after_html = processor._apply_patterns(text, processor.html_patterns)
    print(f"1. After HTML cleaning: {len(text_after_html)} chars")
    if len(text_after_html) < len(text):
        print(f"   Removed {len(text) - len(text_after_html)} chars")
    print(f"   Preview: {text_after_html[:200]}...")
    
    # Step 2: Reference removal
    text_after_refs = processor._apply_patterns(text_after_html, processor.reference_patterns)
    print(f"2. After reference removal: {len(text_after_refs)} chars")
    if len(text_after_refs) < len(text_after_html):
        print(f"   Removed {len(text_after_html) - len(text_after_refs)} chars")
    print(f"   Preview: {text_after_refs[:200]}...")
    
    # Step 3: Unwanted content removal
    text_after_unwanted = processor._remove_unwanted_content(text_after_refs)
    print(f"3. After unwanted content removal: {len(text_after_unwanted)} chars")
    if len(text_after_unwanted) < len(text_after_refs):
        print(f"   Removed {len(text_after_refs) - len(text_after_unwanted)} chars")
    print(f"   Preview: {text_after_unwanted[:200]}...")
    
    # Step 4: Unicode normalization
    text_after_unicode = processor._normalize_unicode(text_after_unwanted)
    print(f"4. After Unicode normalization: {len(text_after_unicode)} chars")
    print(f"   Preview: {text_after_unicode[:200]}...")
    
    # Step 5: Whitespace normalization
    text_after_whitespace = processor._apply_patterns(text_after_unicode, processor.whitespace_patterns)
    print(f"5. After whitespace normalization: {len(text_after_whitespace)} chars")
    print(f"   Preview: {text_after_whitespace[:200]}...")
    
    # Step 6: Final cleaning
    from app.utils.validators import clean_text
    text_final = clean_text(text_after_whitespace)
    print(f"6. After final cleaning: {len(text_final)} chars")
    print(f"   Preview: {text_final[:200]}...")
    
    # Validation
    from app.utils.validators import validate_content_length
    is_valid = validate_content_length(text_final, 100)
    print(f"\nValidation: Content length valid (min 100 chars): {is_valid}")
    
    # Relevance check
    relevance = processor._calculate_colombia_relevance_score(text_final)
    print(f"Colombia relevance score: {relevance:.3f}")


if __name__ == "__main__":
    debug_text_processing()