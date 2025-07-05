#!/usr/bin/env python3
"""
Debug the exact types and values causing the validation issue.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def debug_comparison():
    """Debug the exact comparison that's failing."""
    
    # Simulate the exact scenario
    content_length = 186522
    min_length = 100
    max_length = 100000
    
    print("DEBUGGING COMPARISON")
    print("=" * 30)
    
    print(f"content_length = {content_length} (type: {type(content_length)})")
    print(f"min_length = {min_length} (type: {type(min_length)})")
    print(f"max_length = {max_length} (type: {type(max_length)})")
    
    # Test individual comparisons
    comp1 = min_length <= content_length
    comp2 = content_length <= max_length
    comp_combined = min_length <= content_length <= max_length
    
    print(f"\nIndividual comparisons:")
    print(f"min_length <= content_length: {min_length} <= {content_length} = {comp1}")
    print(f"content_length <= max_length: {content_length} <= {max_length} = {comp2}")
    print(f"Combined: {comp_combined}")
    
    # Test with explicit conversion
    content_length_int = int(content_length)
    min_length_int = int(min_length)
    max_length_int = int(max_length)
    
    comp_int = min_length_int <= content_length_int <= max_length_int
    print(f"With int conversion: {comp_int}")
    
    # Manual check
    print(f"\nManual checks:")
    print(f"186522 >= 100: {186522 >= 100}")
    print(f"186522 <= 100000: {186522 <= 100000}")
    print(f"100 <= 186522 <= 100000: {100 <= 186522 <= 100000}")


def debug_function_directly():
    """Import and test the actual function."""
    
    print("\nTESTING ACTUAL FUNCTION")
    print("=" * 30)
    
    # Import the actual function
    from app.utils.validators import validate_content_length
    
    # Test with known values
    test_content = "A" * 186522
    result = validate_content_length(test_content, 100, 100000)
    print(f"Function result with 186522 chars: {result}")
    
    # Test with simpler content
    simple_content = "This is a test content that should definitely pass validation."
    simple_result = validate_content_length(simple_content, 10, 1000)
    print(f"Function result with simple content: {simple_result}")
    
    # Test the function source
    import inspect
    source = inspect.getsource(validate_content_length)
    print(f"\nFunction source:\n{source}")


if __name__ == "__main__":
    debug_comparison()
    debug_function_directly()