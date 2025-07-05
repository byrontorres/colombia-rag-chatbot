"""
Text processing service for cleaning and normalizing extracted content.
Handles HTML cleaning, text normalization, and content validation.
"""

import re
import time
import unicodedata
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime

from app.config.settings import settings
from app.config.logging import logger, log_error
from app.core.exceptions import (
    DataProcessingError,
    ValidationError,
    ConfigurationError
)
from app.models.documents import (
    RawDocument,
    ProcessedDocument,
    ProcessingStatus,
    ContentType
)
from app.utils.validators import (
    is_colombia_related,
    clean_text,
    validate_content_length
)


class TextProcessor:
    """
    Advanced text processor for Wikipedia content.
    Handles cleaning, normalization, and validation of extracted text.
    """
    
    def __init__(self):
        """Initialize the text processor with cleaning rules."""
        
        # HTML and formatting patterns to remove
        self.html_patterns = [
            (r'<[^>]+>', ''),  # HTML tags
            (r'&nbsp;', ' '),  # Non-breaking spaces
            (r'&[a-zA-Z]+;', ''),  # HTML entities
            (r'&#\d+;', ''),  # Numeric HTML entities
        ]
        
        # Reference and citation patterns
        self.reference_patterns = [
            (r'\[\d+\]', ''),  # [1], [2], etc.
            (r'\[nota \d+\]', ''),  # [nota 1], [nota 2], etc.
            (r'\[cita requerida\]', ''),  # [cita requerida]
            (r'\[¿cuándo\?\]', ''),  # [¿cuándo?]
            (r'\[¿quién\?\]', ''),  # [¿quién?]
            (r'\[aclaración requerida\]', ''),  # [aclaración requerida]
        ]
        
        # Whitespace normalization patterns
        self.whitespace_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'\n\s*\n', '\n\n'),  # Multiple line breaks to double line break
            (r'^\s+', ''),  # Leading whitespace
            (r'\s+$', ''),  # Trailing whitespace
        ]
        
        # Content quality patterns
        self.unwanted_content_patterns = [
            r'^Artículo principal:',
            r'^Véase también:',
            r'^Para otros usos',
            r'^Otros proyectos de Wikimedia',
            r'^Enlaces externos',
            r'^Bibliografía',
            r'^Referencias',
            r'^Notas',
            r'^\d+\s*↑',  # Reference back-links
        ]
        
        # Colombian geographic and cultural keywords for relevance scoring
        self.colombia_keywords = {
            'geography': {
                'colombia', 'bogotá', 'medellín', 'cali', 'barranquilla', 'cartagena',
                'bucaramanga', 'pereira', 'manizales', 'santa marta', 'villavicencio',
                'pasto', 'montería', 'neiva', 'soledad', 'ibagué', 'cúcuta', 'popayán',
                'tunja', 'florencia', 'valledupar', 'quibdó', 'riohacha', 'yopal',
                'arauca', 'casanare', 'amazonas', 'vichada', 'guainía', 'vaupés',
                'andino', 'caribe', 'pacífico', 'amazónico', 'orinoquia', 'magdalena',
                'cauca', 'atrato', 'orinoco', 'cordillera', 'sierra nevada'
            },
            'culture': {
                'muisca', 'wayuu', 'arhuaco', 'embera', 'guambiano', 'nasa',
                'kogui', 'wiwa', 'zenú', 'tikuna', 'cumbia', 'vallenato',
                'salsa', 'bambuco', 'joropo', 'marimba', 'currulao', 'champeta',
                'gabriel garcía márquez', 'fernando botero', 'shakira',
                'juanes', 'carlos vives', 'manu chao'
            },
            'politics': {
                'república de colombia', 'constitución política', 'senado', 'cámara',
                'representantes', 'presidente', 'vicepresidente', 'ministro',
                'gobernador', 'alcalde', 'congreso', 'corte suprema', 'consejo estado',
                'corte constitucional', 'fiscalía', 'procuraduría', 'contraloría'
            },
            'economy': {
                'peso colombiano', 'banco república', 'petróleo', 'carbón', 'café',
                'flores', 'banano', 'cacao', 'caña azúcar', 'arroz', 'maíz',
                'emeraldas', 'oro', 'platino', 'ferroníquel', 'textil', 'confección'
            }
        }
    
    def _apply_patterns(self, text: str, patterns: List[Tuple[str, str]]) -> str:
        """Apply a list of regex patterns to text."""
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.MULTILINE)
        return text
    
    def _remove_unwanted_content(self, text: str) -> str:
        """Remove unwanted content sections."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if line starts with unwanted content
            is_unwanted = any(
                re.match(pattern, line, re.IGNORECASE)
                for pattern in self.unwanted_content_patterns
            )
            
            # Special handling for disambiguation lines - only remove if they are standalone
            if re.match(r'^Para otros usos', line, re.IGNORECASE):
                # Only remove if it's a short disambiguation line
                if len(line) < 200 and ('desambiguación' in line.lower() or 'véase' in line.lower()):
                    continue  # Skip this line
                else:
                    is_unwanted = False  # Keep longer content that starts with "Para otros usos"
            
            if not is_unwanted:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters for consistent processing."""
        # Normalize to NFC (Canonical Decomposition followed by Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        
        # Replace some common problematic characters
        replacements = {
            '\u00a0': ' ',  # Non-breaking space
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2026': '...',  # Horizontal ellipsis
        }
        
        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)
        
        return text
    
    def _calculate_colombia_relevance_score(self, text: str) -> float:
        """Calculate how relevant the text is to Colombia (0.0 to 1.0)."""
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        
        total_keywords = 0
        matched_keywords = 0
        
        for category, keywords in self.colombia_keywords.items():
            total_keywords += len(keywords)
            category_matches = len(keywords.intersection(text_words))
            matched_keywords += category_matches
            
            # Bonus for geography keywords (more important)
            if category == 'geography':
                matched_keywords += category_matches * 0.5
        
        if total_keywords == 0:
            return 0.0
        
        # Base score from keyword matching
        base_score = min(matched_keywords / total_keywords, 1.0)
        
        # Bonus for explicit Colombia mentions
        colombia_mentions = len(re.findall(r'\bcolombi[ao]s?\b', text_lower))
        colombia_bonus = min(colombia_mentions * 0.1, 0.3)
        
        # Length penalty for very short content
        length_factor = min(len(text) / 200, 1.0)
        
        final_score = min((base_score + colombia_bonus) * length_factor, 1.0)
        return final_score
    
    def _extract_processing_stats(
        self, 
        original_text: str, 
        processed_text: str, 
        processing_time: float
    ) -> Dict[str, any]:
        """Extract processing statistics."""
        
        original_lines = original_text.count('\n') + 1
        processed_lines = processed_text.count('\n') + 1
        
        original_words = len(re.findall(r'\b\w+\b', original_text))
        processed_words = len(re.findall(r'\b\w+\b', processed_text))
        
        return {
            'processing_time_seconds': processing_time,
            'original_character_count': len(original_text),
            'processed_character_count': len(processed_text),
            'original_word_count': original_words,
            'processed_word_count': processed_words,
            'original_line_count': original_lines,
            'processed_line_count': processed_lines,
            'character_reduction_percentage': (
                (len(original_text) - len(processed_text)) / len(original_text) * 100
                if len(original_text) > 0 else 0
            ),
            'word_reduction_percentage': (
                (original_words - processed_words) / original_words * 100
                if original_words > 0 else 0
            ),
            'colombia_relevance_score': self._calculate_colombia_relevance_score(processed_text)
        }
    
    def process_document(
        self,
        raw_document: RawDocument,
        clean_html: bool = True,
        normalize_whitespace: bool = True,
        remove_references: bool = True,
        min_paragraph_length: int = 50,
        require_colombia_relevance: bool = True,
        min_relevance_score: float = 0.1
    ) -> Optional[ProcessedDocument]:
        """
        Process a raw document into a cleaned, normalized processed document.
        """
        
        try:
            start_time = time.time()
            
            logger.debug(
                f"Starting text processing for document {raw_document.id}",
                original_length=len(raw_document.content),
                content_type=raw_document.content_type.value
            )
            
            # Start with original content
            processed_text = raw_document.content
            
            # Step 1: Remove HTML if requested
            if clean_html:
                processed_text = self._apply_patterns(processed_text, self.html_patterns)
                logger.debug(f"HTML cleaning completed for {raw_document.id}")
            
            # Step 2: Remove references if requested
            if remove_references:
                processed_text = self._apply_patterns(processed_text, self.reference_patterns)
                logger.debug(f"Reference removal completed for {raw_document.id}")
            
            # Step 3: Remove unwanted content sections
            processed_text = self._remove_unwanted_content(processed_text)
            
            # Step 4: Normalize Unicode
            processed_text = self._normalize_unicode(processed_text)
            
            # Step 5: Normalize whitespace if requested
            if normalize_whitespace:
                processed_text = self._apply_patterns(processed_text, self.whitespace_patterns)
                logger.debug(f"Whitespace normalization completed for {raw_document.id}")
            
            # Step 6: Final cleaning
            processed_text = clean_text(processed_text)
            
            # Validation checks
            if not validate_content_length(processed_text, min_paragraph_length):
                logger.info(
                    f"Document {raw_document.id} rejected: content too short",
                    processed_length=len(processed_text),
                    min_required=min_paragraph_length
                )
                return None
            
            # Colombia relevance check
            if require_colombia_relevance:
                relevance_score = self._calculate_colombia_relevance_score(processed_text)
                if relevance_score < min_relevance_score:
                    logger.info(
                        f"Document {raw_document.id} rejected: low Colombia relevance",
                        relevance_score=relevance_score,
                        min_required=min_relevance_score
                    )
                    return None
            
            # Calculate processing statistics
            processing_time = time.time() - start_time
            processing_stats = self._extract_processing_stats(
                raw_document.content, 
                processed_text, 
                processing_time
            )
            
            # Create processed document
            processed_document = ProcessedDocument(
                id=f"processed_{raw_document.id}",
                raw_document_id=raw_document.id,
                cleaned_content=processed_text,
                metadata=raw_document.metadata.copy(),
                content_type=raw_document.content_type,
                word_count=processing_stats['processed_word_count'],
                character_count=processing_stats['processed_character_count'],
                processing_stats=processing_stats,
                status=ProcessingStatus.COMPLETED
            )
            
            logger.info(
                f"Document processing completed successfully",
                document_id=raw_document.id,
                processed_id=processed_document.id,
                original_length=len(raw_document.content),
                processed_length=len(processed_text),
                word_count=processed_document.word_count,
                relevance_score=processing_stats['colombia_relevance_score'],
                processing_time=processing_time
            )
            
            return processed_document
            
        except Exception as e:
            error_msg = f"Failed to process document {raw_document.id}: {str(e)}"
            log_error(e, {
                "document_id": raw_document.id,
                "content_length": len(raw_document.content),
                "content_type": raw_document.content_type.value
            })
            raise DataProcessingError(error_msg) from e
    
    def batch_process_documents(
        self,
        raw_documents: List[RawDocument],
        **processing_options
    ) -> Tuple[List[ProcessedDocument], Dict[str, any]]:
        """Process multiple documents in batch."""
        
        start_time = time.time()
        processed_documents = []
        failed_documents = []
        skipped_documents = []
        
        logger.info(f"Starting batch processing of {len(raw_documents)} documents")
        
        for i, raw_doc in enumerate(raw_documents):
            try:
                processed_doc = self.process_document(raw_doc, **processing_options)
                
                if processed_doc:
                    processed_documents.append(processed_doc)
                else:
                    skipped_documents.append(raw_doc.id)
                
                # Log progress every 10 documents
                if (i + 1) % 10 == 0:
                    logger.info(f"Batch processing progress: {i + 1}/{len(raw_documents)}")
                    
            except Exception as e:
                failed_documents.append({
                    'document_id': raw_doc.id,
                    'error': str(e)
                })
                logger.warning(f"Failed to process document {raw_doc.id}: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Calculate batch statistics
        batch_stats = {
            'total_input_documents': len(raw_documents),
            'successfully_processed': len(processed_documents),
            'failed_documents': len(failed_documents),
            'skipped_documents': len(skipped_documents),
            'processing_time_seconds': total_time,
            'average_processing_time': total_time / len(raw_documents) if raw_documents else 0,
            'total_processed_characters': sum(doc.character_count for doc in processed_documents),
            'total_processed_words': sum(doc.word_count for doc in processed_documents),
            'failed_document_ids': [f['document_id'] for f in failed_documents],
            'skipped_document_ids': skipped_documents,
            'success_rate': len(processed_documents) / len(raw_documents) if raw_documents else 0
        }
        
        logger.info(f"Batch processing completed", **batch_stats)
        
        return processed_documents, batch_stats
    
    def get_processing_summary(self, processed_documents: List[ProcessedDocument]) -> Dict[str, any]:
        """Get a summary of processed documents."""
        
        if not processed_documents:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'total_words': 0,
                'average_relevance_score': 0.0,
                'content_types': {},
                'sections': {}
            }
        
        # Calculate summary statistics
        total_chars = sum(doc.character_count for doc in processed_documents)
        total_words = sum(doc.word_count for doc in processed_documents)
        
        relevance_scores = [
            doc.processing_stats.get('colombia_relevance_score', 0.0) 
            for doc in processed_documents
        ]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Count content types
        content_types = {}
        for doc in processed_documents:
            content_type = doc.content_type.value
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        # Count sections
        sections = {}
        for doc in processed_documents:
            section = doc.metadata.section or 'Unknown'
            sections[section] = sections.get(section, 0) + 1
        
        return {
            'total_documents': len(processed_documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_characters_per_doc': total_chars / len(processed_documents),
            'average_words_per_doc': total_words / len(processed_documents),
            'average_relevance_score': avg_relevance,
            'content_types': content_types,
            'sections': sections,
            'highest_relevance_score': max(relevance_scores),
            'lowest_relevance_score': min(relevance_scores)
        }