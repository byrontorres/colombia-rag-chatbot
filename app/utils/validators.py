"""
Validation utilities for content and data processing.
"""

import re
import unicodedata
from typing import List, Optional
from urllib.parse import urlparse


def is_valid_url(url: str) -> bool:
    """Check if URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_colombia_related(text: str) -> bool:
    """Check if text content is related to Colombia."""
    colombia_keywords = {
        'colombia', 'colombian', 'colombiano', 'colombiana', 'bogotá',
        'medellín', 'cali', 'barranquilla', 'cartagena', 'bucaramanga',
        'pereira', 'manizales', 'villavicencio', 'pasto', 'montería',
        'neiva', 'soledad', 'ibagué', 'cucuta', 'popayán'
    }
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in colombia_keywords)


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Only remove control characters, keep all printable characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text.strip()


def validate_content_length(content: str, min_length: int = 50, max_length: int = 500000) -> bool:
    """Validate content length is within acceptable bounds."""
    if not content:
        return False
    
    content_length = len(content.strip())
    return min_length <= content_length <= max_length