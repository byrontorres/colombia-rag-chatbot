#!/usr/bin/env python3
"""
Debug script for the Wikipedia extractor.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bs4 import BeautifulSoup
from app.services.data_extractor import WikipediaExtractor


def debug_extraction_step_by_step():
    """Debug the extraction process step by step."""
    
    print("DEBUGGING WIKIPEDIA EXTRACTION")
    print("=" * 50)
    
    extractor = WikipediaExtractor()
    url = "https://es.wikipedia.org/wiki/Colombia"
    
    # Step 1: Fetch HTML
    print("1. Fetching HTML...")
    html = extractor._fetch_page_content(url, True)
    print(f"   HTML length: {len(html):,} characters")
    
    # Step 2: Parse HTML
    print("\n2. Parsing HTML...")
    soup = BeautifulSoup(html, 'html.parser')
    
    # Step 3: Find main content area
    print("\n3. Finding main content area...")
    content_div = soup.find('div', {'class': 'mw-parser-output'})
    
    if content_div:
        print(f"   FOUND: mw-parser-output div with {len(str(content_div)):,} characters")
    else:
        print("   NOT FOUND: mw-parser-output div")
        # Try alternatives
        alternatives = [
            ('div', {'id': 'mw-content-text'}),
            ('div', {'class': 'mw-body-content'}),
            ('div', {'id': 'bodyContent'})
        ]
        
        for tag, attrs in alternatives:
            alt_div = soup.find(tag, attrs)
            if alt_div:
                print(f"   FOUND ALTERNATIVE: {tag} {attrs}")
                content_div = alt_div
                break
        
        if not content_div:
            print("   ERROR: No content div found!")
            return
    
    # Step 4: Check introduction extraction
    print("\n4. Checking introduction extraction...")
    intro_elements = []
    element_count = 0
    
    for element in content_div.children:
        element_count += 1
        if element_count > 20:  # Limit for debugging
            break
            
        if hasattr(element, 'name'):
            print(f"   Element {element_count}: {element.name}")
            if element.name in ['h2', 'h3', 'h4']:
                print(f"   HEADER FOUND: {element.get_text(strip=True)}")
                break
            if element.name in ['p', 'div', 'ul', 'ol']:
                intro_elements.append(element)
                print(f"   INTRO ELEMENT: {element.name} - {len(str(element))} chars")
    
    print(f"   Total intro elements found: {len(intro_elements)}")
    
    # Test intro content extraction
    if intro_elements:
        print("\n5. Testing intro content extraction...")
        for i, elem in enumerate(intro_elements[:3]):
            content = extractor._extract_content_from_element(elem)
            if content:
                print(f"   Intro {i+1}: {len(content)} chars - {content[:100]}...")
            else:
                print(f"   Intro {i+1}: No content extracted")
    
    # Step 6: Find section headers
    print("\n6. Finding section headers...")
    sections = content_div.find_all(['h2', 'h3', 'h4'])
    print(f"   Total headers found: {len(sections)}")
    
    if sections:
        print("   First 10 headers:")
        for i, header in enumerate(sections[:10]):
            title = header.get_text(strip=True)
            clean_title = extractor._clean_section_title(title)
            is_relevant = extractor._is_section_relevant(title)
            print(f"   {i+1}. '{title}' -> '{clean_title}' (Relevant: {is_relevant})")
    
    # Step 7: Test section content extraction
    print("\n7. Testing section content extraction...")
    relevant_headers = [h for h in sections[:5] if extractor._is_section_relevant(h.get_text(strip=True))]
    
    if relevant_headers:
        test_header = relevant_headers[0]
        title = test_header.get_text(strip=True)
        print(f"   Testing section: {title}")
        
        # Find content elements
        section_content = []
        current = test_header.next_sibling
        element_count = 0
        
        while current and element_count < 10:  # Limit for debugging
            if hasattr(current, 'name'):
                if current.name in ['h2', 'h3', 'h4']:
                    print(f"   Next header found: {current.get_text(strip=True)}")
                    break
                if current.name in ['p', 'div', 'ul', 'ol']:
                    section_content.append(current)
                    print(f"   Content element {len(section_content)}: {current.name}")
                    element_count += 1
            current = current.next_sibling
        
        print(f"   Content elements found: {len(section_content)}")
        
        # Test content extraction
        if section_content:
            test_elem = section_content[0]
            content = extractor._extract_content_from_element(test_elem)
            if content:
                print(f"   Sample content: {len(content)} chars - {content[:100]}...")
            else:
                print("   No content extracted from test element")


if __name__ == "__main__":
    debug_extraction_step_by_step()