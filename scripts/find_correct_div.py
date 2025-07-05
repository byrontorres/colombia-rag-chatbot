#!/usr/bin/env python3
"""
Find the correct content div in Wikipedia.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bs4 import BeautifulSoup
from app.services.data_extractor import WikipediaExtractor


def find_content_divs():
    """Find all possible content divs."""
    
    extractor = WikipediaExtractor()
    html = extractor._fetch_page_content("https://es.wikipedia.org/wiki/Colombia", True)
    soup = BeautifulSoup(html, 'html.parser')
    
    print("FINDING CONTENT DIVS")
    print("=" * 50)
    
    # Test different selectors
    selectors = [
        ('div.mw-parser-output', 'Main parser output'),
        ('div#mw-content-text', 'Content text by ID'),
        ('div.mw-body-content', 'Body content'),
        ('div#bodyContent', 'Body content by ID'),
        ('div#content', 'Content by ID'),
        ('main', 'Main element'),
        ('article', 'Article element'),
        ('.mw-parser-output', 'Parser output class'),
    ]
    
    for selector, description in selectors:
        elements = soup.select(selector)
        print(f"\n{description} ({selector}):")
        print(f"   Found {len(elements)} elements")
        
        for i, elem in enumerate(elements):
            size = len(str(elem))
            headers = elem.find_all(['h1', 'h2', 'h3', 'h4'])
            paragraphs = elem.find_all('p')
            print(f"   Element {i+1}: {size:,} chars, {len(headers)} headers, {len(paragraphs)} paragraphs")
            
            if headers:
                print(f"      First header: {headers[0].get_text(strip=True)}")
            if paragraphs:
                first_p_text = paragraphs[0].get_text(strip=True)[:100]
                print(f"      First paragraph: {first_p_text}...")
    
    # Check the page structure
    print(f"\nPAGE STRUCTURE ANALYSIS")
    print("=" * 30)
    
    # Find all divs with meaningful content
    all_divs = soup.find_all('div')
    large_divs = [div for div in all_divs if len(str(div)) > 10000]
    
    print(f"Total divs: {len(all_divs)}")
    print(f"Large divs (>10k chars): {len(large_divs)}")
    
    for i, div in enumerate(large_divs[:5]):  # Top 5 largest
        classes = div.get('class', [])
        div_id = div.get('id', 'no-id')
        size = len(str(div))
        headers = div.find_all(['h2', 'h3', 'h4'])
        
        print(f"\nLarge div {i+1}:")
        print(f"   ID: {div_id}")
        print(f"   Classes: {classes}")
        print(f"   Size: {size:,} chars")
        print(f"   Headers: {len(headers)}")
        
        if headers:
            sample_headers = [h.get_text(strip=True) for h in headers[:3]]
            print(f"   Sample headers: {sample_headers}")


if __name__ == "__main__":
    find_content_divs()