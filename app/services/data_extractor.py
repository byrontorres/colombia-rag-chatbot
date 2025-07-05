"""
Data extraction service for Wikipedia Colombia content.
Handles web scraping, content extraction, and data validation.
"""

import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config.settings import settings
from app.config.logging import logger, log_error
from app.core.exceptions import (
    DataExtractionError,
    NetworkError,
    TimeoutError,
    ConfigurationError
)
from app.models.documents import (
    RawDocument,
    SourceMetadata,
    ContentType,
    ProcessingStatus
)


class WikipediaExtractor:
    """
    Wikipedia content extractor with robust error handling and caching.
    """
    
    def __init__(self):
        """Initialize the Wikipedia extractor."""
        self.session = self._create_session()
        self.cache_dir = Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        
        # Content filters
        self.unwanted_sections = {
            'referencias', 'reference', 'bibliography', 'bibliografía',
            'external links', 'enlaces externos', 'further reading',
            'lectura adicional', 'see also', 'véase también',
            'notes', 'notas', 'citations', 'citas'
        }
        
        # Important sections for Colombia
        self.priority_sections = {
            'historia', 'geografía', 'política', 'economía', 'cultura',
            'demografía', 'gobierno', 'división territorial', 'clima',
            'biodiversidad', 'recursos naturales', 'educación', 'salud'
        }
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy and proper headers."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers to appear as a legitimate browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    def _rate_limit(self):
        """Implement rate limiting to be respectful to Wikipedia."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"wikipedia_cache_{url_hash}.html"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached content is still valid."""
        if not cache_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)
    
    def _fetch_page_content(self, url: str, use_cache: bool = True) -> str:
        """
        Fetch page content from URL with caching and error handling.
        
        Args:
            url: URL to fetch
            use_cache: Whether to use cached content if available
            
        Returns:
            Raw HTML content
            
        Raises:
            NetworkError: If network request fails
            TimeoutError: If request times out
        """
        cache_path = self._get_cache_path(url)
        
        # Check cache first
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Using cached content for {url}")
            return cache_path.read_text(encoding='utf-8')
        
        # Rate limit requests
        self._rate_limit()
        
        try:
            logger.info(f"Fetching content from {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            cache_path.write_text(response.text, encoding='utf-8')
            logger.info(f"Content cached to {cache_path}")
            
            return response.text
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out")
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Failed to fetch {url}: {str(e)}")
    
    def _extract_page_metadata(self, soup: BeautifulSoup, url: str) -> SourceMetadata:
        """Extract metadata from Wikipedia page."""
        
        # Page title
        title_tag = soup.find('h1', {'class': 'firstHeading'})
        title = title_tag.get_text(strip=True) if title_tag else "Unknown"
        
        # Last modified date (if available)
        last_modified = None
        modified_tag = soup.find('li', id='footer-info-lastmod')
        if modified_tag:
            modified_text = modified_tag.get_text()
            # Parse date from "Esta página se editó por última vez el..."
            date_match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', modified_text)
            if date_match:
                try:
                    # This is a simplified date parsing - in production you'd want more robust parsing
                    day, month_name, year = date_match.groups()
                    month_map = {
                        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
                        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
                        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
                    }
                    month = month_map.get(month_name.lower(), 1)
                    last_modified = datetime(int(year), month, int(day))
                except (ValueError, KeyError):
                    pass
        
        return SourceMetadata(
            url=url,
            title=title,
            language="es",
            last_modified=last_modified,
            extraction_date=datetime.utcnow()
        )
    
    def _clean_section_title(self, title: str) -> str:
        """Clean and normalize section title."""
        # Remove edit links and extra whitespace
        title = re.sub(r'\[editar\]', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+', ' ', title.strip())
        return title.lower()
    
    def _is_section_relevant(self, section_title: str) -> bool:
        """Check if a section is relevant for our Colombia knowledge base."""
        clean_title = self._clean_section_title(section_title)
        
        # Skip unwanted sections
        if any(unwanted in clean_title for unwanted in self.unwanted_sections):
            return False
        
        # Prioritize important sections
        if any(priority in clean_title for priority in self.priority_sections):
            return True
        
        # Allow sections that seem informative (basic heuristics)
        if len(clean_title) > 3 and not any(char.isdigit() for char in clean_title):
            return True
        
        return False
    
    def _extract_content_from_element(self, element: Tag) -> Optional[str]:
        """Extract clean text content from a DOM element."""
        if not element:
            return None
        
        # Remove unwanted elements
        unwanted_tags = ['script', 'style', 'sup', 'table', '.mw-editsection']
        for unwanted in unwanted_tags:
            for tag in element.select(unwanted):
                tag.decompose()
        
        # Get text and clean it
        text = element.get_text(separator=' ', strip=True)
        
        # Clean whitespace and formatting
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)  # Remove reference markers
        text = text.strip()
        
        return text if len(text) > 50 else None  # Only return substantial content
    
    def _extract_sections(self, soup: BeautifulSoup, metadata: SourceMetadata) -> List[RawDocument]:
        """Extract content sections from Wikipedia page."""
        documents = []
        
        # Find the main content div - use the largest mw-parser-output div
        content_divs = soup.find_all('div', {'class': 'mw-parser-output'})
        content_div = None

        if content_divs:
            # Take the largest div (main content)
            content_div = max(content_divs, key=lambda d: len(str(d)))
        else:
            # Fallback selectors
            content_div = (soup.find('div', {'id': 'mw-content-text'}) or
                        soup.find('div', {'class': 'mw-body-content'}) or
                        soup.find('div', {'id': 'bodyContent'}))
            
        if not content_div:
            raise DataExtractionError("Could not find main content area in Wikipedia page")
        
        # Extract introduction (content before first h2)
        intro_elements = []
        for element in content_div.children:
            if hasattr(element, 'name'):
                if element.name in ['h2', 'h3', 'h4']:
                    break
                if element.name in ['p', 'div', 'ul', 'ol']:
                    intro_elements.append(element)
        
        if intro_elements:
            intro_content = ' '.join([
                self._extract_content_from_element(elem) or ''
                for elem in intro_elements
            ]).strip()
            
            if intro_content and len(intro_content) > 100:
                intro_metadata = metadata.copy()
                intro_metadata.section = "Introducción"
                
                documents.append(RawDocument(
                    id=f"colombia_intro_{int(time.time())}",
                    content=intro_content,
                    metadata=intro_metadata,
                    content_type=ContentType.PARAGRAPH
                ))
                
                logger.info(f"Extracted introduction: {len(intro_content)} characters")
        
        # Extract sections
        sections = content_div.find_all(['h2', 'h3', 'h4'])
        
        for i, section_header in enumerate(sections):
            section_title = section_header.get_text(strip=True)
            
            if not self._is_section_relevant(section_title):
                logger.debug(f"Skipping irrelevant section: {section_title}")
                continue
            
            # Find content between this header and the next
            section_content = []
            current = section_header.next_sibling
            
            while current:
                if hasattr(current, 'name'):
                    # Stop at next header of same or higher level
                    if current.name in ['h2', 'h3', 'h4']:
                        header_level = int(current.name[1])
                        current_level = int(section_header.name[1])
                        if header_level <= current_level:
                            break
                    
                    # Collect content elements
                    if current.name in ['p', 'div', 'ul', 'ol']:
                        section_content.append(current)
                
                current = current.next_sibling
            
            if section_content:
                content_text = ' '.join([
                    self._extract_content_from_element(elem) or ''
                    for elem in section_content
                ]).strip()
                
                if content_text and len(content_text) > 100:
                    section_metadata = metadata.copy()
                    section_metadata.section = self._clean_section_title(section_title).title()
                    
                    documents.append(RawDocument(
                        id=f"colombia_{self._clean_section_title(section_title).replace(' ', '_')}_{int(time.time())}_{i}",
                        content=content_text,
                        metadata=section_metadata,
                        content_type=ContentType.PARAGRAPH
                    ))
                    
                    logger.info(f"Extracted section '{section_title}': {len(content_text)} characters")
        
        return documents
    
    def extract_colombia_content(
        self, 
        url: Optional[str] = None, 
        use_cache: bool = True,
        sections: Optional[List[str]] = None
    ) -> List[RawDocument]:
        """
        Extract content from Colombia Wikipedia page.
        
        Args:
            url: Wikipedia URL (defaults to settings.wikipedia_url)
            use_cache: Whether to use cached content
            sections: Specific sections to extract (None for all)
            
        Returns:
            List of extracted RawDocument objects
            
        Raises:
            DataExtractionError: If extraction fails
        """
        if not url:
            url = settings.wikipedia_url
        
        if not url:
            raise ConfigurationError("No Wikipedia URL configured")
        
        try:
            logger.info(f"Starting content extraction from {url}")
            start_time = time.time()
            
            # Fetch page content
            html_content = self._fetch_page_content(url, use_cache)
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            metadata = self._extract_page_metadata(soup, url)
            logger.info(f"Extracted metadata: {metadata.title}")
            
            # Extract sections
            documents = self._extract_sections(soup, metadata)
            
            # Filter by requested sections if specified
            if sections:
                sections_lower = [s.lower() for s in sections]
                documents = [
                    doc for doc in documents 
                    if doc.metadata.section and doc.metadata.section.lower() in sections_lower
                ]
            
            extraction_time = time.time() - start_time
            
            logger.info(
                f"Content extraction completed",
                documents_extracted=len(documents),
                total_characters=sum(len(doc.content) for doc in documents),
                extraction_time=extraction_time,
                sections_found=[doc.metadata.section for doc in documents]
            )
            
            return documents
            
        except Exception as e:
            error_msg = f"Failed to extract content from {url}: {str(e)}"
            log_error(e, {"url": url, "use_cache": use_cache})
            raise DataExtractionError(error_msg) from e
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about cached content."""
        cache_files = list(self.cache_dir.glob("wikipedia_cache_*.html"))
        
        cache_info = {
            "cache_directory": str(self.cache_dir),
            "cached_files": len(cache_files),
            "total_cache_size": sum(f.stat().st_size for f in cache_files),
            "oldest_cache": None,
            "newest_cache": None
        }
        
        if cache_files:
            cache_times = [f.stat().st_mtime for f in cache_files]
            cache_info["oldest_cache"] = datetime.fromtimestamp(min(cache_times))
            cache_info["newest_cache"] = datetime.fromtimestamp(max(cache_times))
        
        return cache_info
    
    def clear_cache(self) -> int:
        """Clear all cached content. Returns number of files deleted."""
        cache_files = list(self.cache_dir.glob("wikipedia_cache_*.html"))
        deleted = 0
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                deleted += 1
            except OSError as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {deleted} cache files")
        return deleted