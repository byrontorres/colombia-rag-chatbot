"""
Intelligent chunking service for processed documents.
Handles semantic chunking with sentence boundary preservation and overlap management.
"""

import re
import time
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
    ProcessedDocument,
    DocumentChunk,
    ProcessingStatus
)


class IntelligentChunker:
    """
    Advanced document chunker with semantic awareness and sentence boundary preservation.
    """
    
    def __init__(self):
        """Initialize the chunker with language-specific patterns."""
        
        # Spanish sentence boundary patterns
        self.sentence_endings = [
            r'\.(?=\s+[A-ZÁÉÍÓÚÑ])',  # Period followed by space and capital letter
            r'\!(?=\s+[A-ZÁÉÍÓÚÑ])',  # Exclamation mark
            r'\?(?=\s+[A-ZÁÉÍÓÚÑ])',  # Question mark
            r'\.(?=\s*$)',  # Period at end of text
            r'\!(?=\s*$)',  # Exclamation at end
            r'\?(?=\s*$)',  # Question at end
        ]
        
        # Patterns that should NOT be sentence boundaries
        self.sentence_exceptions = [
            r'Sr\.',  # Señor
            r'Sra\.',  # Señora
            r'Dr\.',  # Doctor
            r'Dra\.',  # Doctora
            r'Prof\.',  # Profesor
            r'Inc\.',  # Incorporated
            r'Ltd\.',  # Limited
            r'S\.A\.',  # Sociedad Anónima
            r'Ltda\.',  # Limitada
            r'etc\.',  # etcetera
            r'vs\.',  # versus
            r'p\.ej\.',  # por ejemplo
            r'i\.e\.',  # id est
            r'e\.g\.',  # exempli gratia
            r'\d+\.',  # Numbers followed by period
        ]
        
        # Logical breaking points (in order of preference)
        self.breaking_preferences = [
            r'\.\s+',  # After sentences
            r'\!\s+',  # After exclamations
            r'\?\s+',  # After questions
            r';\s+',   # After semicolons
            r':\s+',   # After colons
            r',\s+',   # After commas (lower priority)
            r'\s+',    # At word boundaries (last resort)
        ]
        
        # Section and paragraph indicators
        self.section_indicators = [
            r'^##?\s+',  # Markdown headers
            r'^\d+\.\s+',  # Numbered sections
            r'^[A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]+:',  # Title with colon
        ]
    
    def _find_sentence_boundaries(self, text: str) -> List[int]:
        """Find sentence boundaries in text, respecting Spanish grammar rules."""
        
        boundaries = []
        
        # First, mark all potential sentence endings
        potential_boundaries = []
        
        for pattern in self.sentence_endings:
            for match in re.finditer(pattern, text):
                potential_boundaries.append(match.end())
        
        # Remove boundaries that match exceptions
        filtered_boundaries = []
        
        for boundary in potential_boundaries:
            is_exception = False
            
            # Check context around boundary
            start_context = max(0, boundary - 10)
            end_context = min(len(text), boundary + 5)
            context = text[start_context:end_context]
            
            # Check if it matches any exception pattern
            for exception_pattern in self.sentence_exceptions:
                if re.search(exception_pattern, context, re.IGNORECASE):
                    is_exception = True
                    break
            
            if not is_exception:
                filtered_boundaries.append(boundary)
        
        # Sort and deduplicate
        boundaries = sorted(set(filtered_boundaries))
        
        return boundaries
    
    def _find_optimal_break_point(
        self, 
        text: str, 
        target_position: int, 
        window_size: int = 100
    ) -> int:
        """Find the optimal break point near target position."""
        
        # Define search window
        start_pos = max(0, target_position - window_size)
        end_pos = min(len(text), target_position + window_size)
        search_window = text[start_pos:end_pos]
        
        # Try breaking preferences in order
        for i, pattern in enumerate(self.breaking_preferences):
            matches = list(re.finditer(pattern, search_window))
            
            if matches:
                # Find match closest to target position
                target_in_window = target_position - start_pos
                closest_match = min(
                    matches, 
                    key=lambda m: abs(m.end() - target_in_window)
                )
                
                # Convert back to original text position
                break_position = start_pos + closest_match.end()
                
                # Ensure we don't break in the middle of a word unless necessary
                if pattern == r'\s+':  # Word boundary
                    # Look for better break within a smaller window
                    word_window = 20
                    word_start = max(start_pos, break_position - word_window)
                    word_end = min(end_pos, break_position + word_window)
                    word_search = text[word_start:word_end]
                    
                    # Try to find punctuation break
                    for punct_pattern in [r'\.\s+', r';\s+', r':\s+', r',\s+']:
                        punct_matches = list(re.finditer(punct_pattern, word_search))
                        if punct_matches:
                            closest_punct = min(
                                punct_matches,
                                key=lambda m: abs(m.end() - (break_position - word_start))
                            )
                            return word_start + closest_punct.end()
                
                return break_position
        
        # Fallback: return target position
        return target_position
    
    def _calculate_overlap_content(
        self, 
        text: str, 
        chunk_start: int, 
        chunk_end: int, 
        overlap_size: int
    ) -> Tuple[str, str]:
        """Calculate overlap content for previous and next chunks."""
        
        # Content that overlaps with previous chunk
        overlap_start = max(0, chunk_start - overlap_size)
        context_before = text[overlap_start:chunk_start].strip()
        if len(context_before) > 200:
            context_before = "..." + context_before[-197:]
        
        # Content that overlaps with next chunk  
        overlap_end = min(len(text), chunk_end + overlap_size)
        context_after = text[chunk_end:overlap_end].strip()
        if len(context_after) > 200:
            context_after = context_after[:197] + "..."
        
        return context_before, context_after
    
    def _create_chunk(
        self,
        document: ProcessedDocument,
        chunk_content: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        overlap_before: str = "",
        overlap_after: str = "",
        overlap_size: int = 0
    ) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata."""
        
        # Clean chunk content
        chunk_content = chunk_content.strip()
        
        # Calculate word count
        word_count = len(re.findall(r'\b\w+\b', chunk_content))
        
        # Generate unique chunk ID
        chunk_id = f"{document.id}_chunk_{chunk_index:03d}"
        
        return DocumentChunk(
            id=chunk_id,
            document_id=document.id,
            content=chunk_content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            word_count=word_count,
            character_count=len(chunk_content),
            metadata=document.metadata.copy(),
            overlap_with_previous=len(overlap_before),
            overlap_with_next=len(overlap_after),
            context_before=overlap_before if overlap_before else None,
            context_after=overlap_after if overlap_after else None
        )
    
    def chunk_document(
        self,
        document: ProcessedDocument,
        chunk_size: int = None,
        chunk_overlap: int = None,
        preserve_sentences: bool = True,
        min_chunk_size: int = None,
        max_chunk_size: int = None
    ) -> List[DocumentChunk]:
        """
        Chunk a processed document into smaller pieces with intelligent boundary detection.
        """
        
        # Use settings defaults if not provided
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        min_chunk_size = min_chunk_size or max(100, chunk_size // 4)
        max_chunk_size = max_chunk_size or chunk_size * 2
        
        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValidationError("Chunk overlap must be less than chunk size")
        
        if min_chunk_size >= max_chunk_size:
            raise ValidationError("Min chunk size must be less than max chunk size")
        
        try:
            start_time = time.time()
            
            logger.debug(
                f"Starting chunking for document {document.id}",
                content_length=len(document.cleaned_content),
                target_chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                preserve_sentences=preserve_sentences
            )
            
            text = document.cleaned_content
            chunks = []
            
            # Handle very short documents
            if len(text) <= max_chunk_size:
                if len(text) >= min_chunk_size:
                    chunk = self._create_chunk(
                        document=document,
                        chunk_content=text,
                        chunk_index=0,
                        start_char=0,
                        end_char=len(text)
                    )
                    chunks.append(chunk)
                
                logger.info(
                    f"Document {document.id} is short, created {len(chunks)} chunk(s)",
                    content_length=len(text),
                    chunks_created=len(chunks)
                )
                return chunks
            
            # Find sentence boundaries if requested
            sentence_boundaries = []
            if preserve_sentences:
                sentence_boundaries = self._find_sentence_boundaries(text)
                logger.debug(f"Found {len(sentence_boundaries)} sentence boundaries")
            
            # Chunking logic
            current_position = 0
            chunk_index = 0
            
            while current_position < len(text):
                # Calculate target end position
                target_end = current_position + chunk_size
                
                # If this would be the last chunk and it's not too big, take everything
                remaining_text = len(text) - current_position
                if remaining_text <= max_chunk_size:
                    chunk_end = len(text)
                else:
                    # Find optimal break point
                    if preserve_sentences and sentence_boundaries:
                        # Try to break at sentence boundary
                        suitable_boundaries = [
                            b for b in sentence_boundaries 
                            if current_position + min_chunk_size <= b <= current_position + max_chunk_size
                        ]
                        
                        if suitable_boundaries:
                            # Choose boundary closest to target
                            chunk_end = min(suitable_boundaries, key=lambda x: abs(x - target_end))
                        else:
                            # No suitable sentence boundary, find optimal break point
                            chunk_end = self._find_optimal_break_point(text, target_end)
                    else:
                        # Find optimal break point without sentence constraints
                        chunk_end = self._find_optimal_break_point(text, target_end)
                
                # Ensure chunk is not too small (unless it's the last chunk)
                if chunk_end - current_position < min_chunk_size and chunk_end < len(text):
                    chunk_end = min(len(text), current_position + min_chunk_size)
                    chunk_end = self._find_optimal_break_point(text, chunk_end)
                
                # Extract chunk content
                chunk_content = text[current_position:chunk_end]
                
                # Calculate overlap content
                overlap_before, overlap_after = self._calculate_overlap_content(
                    text, current_position, chunk_end, chunk_overlap
                )
                
                # Create chunk
                chunk = self._create_chunk(
                    document=document,
                    chunk_content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=current_position,
                    end_char=chunk_end,
                    overlap_before=overlap_before,
                    overlap_after=overlap_after,
                    overlap_size=chunk_overlap
                )
                
                chunks.append(chunk)
                
                # Move to next position (with overlap)
                next_position = chunk_end - chunk_overlap
                
                # Ensure we make progress
                if next_position <= current_position:
                    next_position = current_position + min_chunk_size
                
                current_position = next_position
                chunk_index += 1
                
                # Safety check to prevent infinite loops
                if chunk_index > 1000:
                    logger.warning(f"Chunking stopped at 1000 chunks for document {document.id}")
                    break
            
            chunking_time = time.time() - start_time
            
            logger.info(
                f"Document chunking completed",
                document_id=document.id,
                chunks_created=len(chunks),
                total_characters=sum(chunk.character_count for chunk in chunks),
                average_chunk_size=sum(chunk.character_count for chunk in chunks) / len(chunks) if chunks else 0,
                chunking_time=chunking_time
            )
            
            return chunks
            
        except Exception as e:
            error_msg = f"Failed to chunk document {document.id}: {str(e)}"
            log_error(e, {
                "document_id": document.id,
                "content_length": len(document.cleaned_content),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            })
            raise DataProcessingError(error_msg) from e
    
    def batch_chunk_documents(
        self,
        documents: List[ProcessedDocument],
        **chunking_options
    ) -> Tuple[List[DocumentChunk], Dict[str, any]]:
        """Chunk multiple documents in batch."""
        
        start_time = time.time()
        all_chunks = []
        failed_documents = []
        
        logger.info(f"Starting batch chunking of {len(documents)} documents")
        
        for i, document in enumerate(documents):
            try:
                document_chunks = self.chunk_document(document, **chunking_options)
                all_chunks.extend(document_chunks)
                
                # Log progress every 5 documents
                if (i + 1) % 5 == 0:
                    logger.info(f"Batch chunking progress: {i + 1}/{len(documents)}")
                    
            except Exception as e:
                failed_documents.append({
                    'document_id': document.id,
                    'error': str(e)
                })
                logger.warning(f"Failed to chunk document {document.id}: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Calculate batch statistics
        batch_stats = {
            'total_input_documents': len(documents),
            'successfully_chunked': len(documents) - len(failed_documents),
            'failed_documents': len(failed_documents),
            'total_chunks_created': len(all_chunks),
            'chunking_time_seconds': total_time,
            'average_chunking_time': total_time / len(documents) if documents else 0,
            'average_chunks_per_document': len(all_chunks) / len(documents) if documents else 0,
            'total_chunk_characters': sum(chunk.character_count for chunk in all_chunks),
            'average_chunk_size': (
                sum(chunk.character_count for chunk in all_chunks) / len(all_chunks) 
                if all_chunks else 0
            ),
            'failed_document_ids': [f['document_id'] for f in failed_documents],
            'success_rate': (len(documents) - len(failed_documents)) / len(documents) if documents else 0
        }
        
        logger.info(f"Batch chunking completed", **batch_stats)
        
        return all_chunks, batch_stats
    
    def get_chunking_summary(self, chunks: List[DocumentChunk]) -> Dict[str, any]:
        """Get a summary of chunked documents."""
        
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'total_words': 0,
                'documents_represented': 0,
                'average_chunk_size': 0.0,
                'size_distribution': {}
            }
        
        # Calculate summary statistics
        total_chars = sum(chunk.character_count for chunk in chunks)
        total_words = sum(chunk.word_count for chunk in chunks)
        
        # Count unique documents
        unique_documents = len(set(chunk.document_id for chunk in chunks))
        
        # Calculate size distribution
        size_ranges = [
            (0, 500, 'Small'),
            (500, 1000, 'Medium'),
            (1000, 1500, 'Large'),
            (1500, float('inf'), 'Extra Large')
        ]
        
        size_distribution = {label: 0 for _, _, label in size_ranges}
        for chunk in chunks:
            for min_size, max_size, label in size_ranges:
                if min_size <= chunk.character_count < max_size:
                    size_distribution[label] += 1
                    break
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_words': total_words,
            'documents_represented': unique_documents,
            'average_chunk_size': total_chars / len(chunks),
            'average_words_per_chunk': total_words / len(chunks),
            'size_distribution': size_distribution,
            'chunks_with_overlap': sum(1 for chunk in chunks if chunk.overlap_with_previous > 0),
            'chunks_with_context': sum(1 for chunk in chunks if chunk.context_before or chunk.context_after)
        }