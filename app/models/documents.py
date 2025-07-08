"""
Pydantic models for document handling and data structures.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


class ContentType(str, Enum):
    """Types of content that can be extracted."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    INFOBOX = "infobox"
    REFERENCE = "reference"


class ProcessingStatus(str, Enum):
    """Processing status for documents and chunks."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SourceMetadata(BaseModel):
    """Metadata about the source of the content."""
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Page title")
    language: str = Field(default="es", description="Content language")
    last_modified: Optional[datetime] = Field(None, description="Last modification date")
    extraction_date: datetime = Field(default_factory=datetime.utcnow, description="When content was extracted")
    section: Optional[str] = Field(None, description="Section within the page")
    subsection: Optional[str] = Field(None, description="Subsection within the page")


class RawDocument(BaseModel):
    """Raw document extracted from Wikipedia before processing."""
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Raw HTML/text content")
    metadata: SourceMetadata = Field(..., description="Source metadata")
    content_type: ContentType = Field(..., description="Type of content")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    @validator('content')
    def validate_content_not_empty(cls, v):
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()
    
    @validator('id')
    def validate_id_format(cls, v):
        """Ensure ID follows expected format."""
        if not v or len(v) < 3:
            raise ValueError("ID must be at least 3 characters long")
        return v


class ProcessedDocument(BaseModel):
    """Document after text processing and cleaning."""
    id: str = Field(..., description="Unique document identifier")
    raw_document_id: str = Field(..., description="Reference to original raw document")
    cleaned_content: str = Field(..., description="Cleaned and processed text")
    metadata: SourceMetadata = Field(..., description="Source metadata")
    content_type: ContentType = Field(..., description="Type of content")
    word_count: int = Field(..., description="Number of words in cleaned content")
    character_count: int = Field(..., description="Number of characters in cleaned content")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    status: ProcessingStatus = Field(default=ProcessingStatus.COMPLETED, description="Processing status")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    
    @validator('cleaned_content')
    def validate_cleaned_content(cls, v):
        """Ensure cleaned content is valid."""
        if not v or not v.strip():
            raise ValueError("Cleaned content cannot be empty")
        if len(v.strip()) < 10:
            raise ValueError("Cleaned content too short (minimum 10 characters)")
        return v.strip()
    
    @validator('word_count')
    def validate_word_count(cls, v, values):
        """Ensure word count matches content."""
        if v <= 0:
            raise ValueError("Word count must be positive")
        return v


class DocumentChunk(BaseModel):
    """Individual chunk of a processed document."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Position in document (0-based)")
    start_char: int = Field(..., description="Start character position in original document")
    end_char: int = Field(..., description="End character position in original document")
    word_count: int = Field(..., description="Number of words in chunk")
    character_count: int = Field(..., description="Number of characters in chunk")
    metadata: SourceMetadata = Field(..., description="Source metadata")
    overlap_with_previous: int = Field(default=0, description="Characters overlapping with previous chunk")
    overlap_with_next: int = Field(default=0, description="Characters overlapping with next chunk")
    context_before: Optional[str] = Field(None, max_length=200, description="Context before this chunk")
    context_after: Optional[str] = Field(None, max_length=200, description="Context after this chunk")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    @validator('content')
    def validate_chunk_content(cls, v):
        """Ensure chunk content is valid."""
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty")
        if len(v.strip()) < 50:
            raise ValueError("Chunk content too short (minimum 50 characters)")
        return v.strip()
    
    @validator('chunk_index')
    def validate_chunk_index(cls, v):
        """Ensure chunk index is non-negative."""
        if v < 0:
            raise ValueError("Chunk index must be non-negative")
        return v
    
    @validator('start_char', 'end_char')
    def validate_char_positions(cls, v):
        """Ensure character positions are non-negative."""
        if v < 0:
            raise ValueError("Character positions must be non-negative")
        return v
    
    @validator('end_char')
    def validate_end_after_start(cls, v, values):
        """Ensure end position is after start position."""
        if 'start_char' in values and v <= values['start_char']:
            raise ValueError("End character position must be after start position")
        return v


class ExtractionJob(BaseModel):
    """Job for extracting content from a source."""
    id: str = Field(..., description="Unique job identifier")
    source_url: str = Field(..., description="URL to extract from")
    job_type: str = Field(default="wikipedia_extraction", description="Type of extraction job")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Job status")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    documents_extracted: int = Field(default=0, description="Number of documents extracted")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    config: Dict[str, Any] = Field(default_factory=dict, description="Job configuration")
    
    @validator('source_url')
    def validate_source_url(cls, v):
        """Ensure source URL is valid."""
        if not v or not v.startswith(('http://', 'https://')):
            raise ValueError("Source URL must be a valid HTTP/HTTPS URL")
        return v


class ProcessingStats(BaseModel):
    """Statistics for processing operations."""
    total_documents: int = Field(default=0, description="Total documents processed")
    total_chunks: int = Field(default=0, description="Total chunks created")
    total_characters: int = Field(default=0, description="Total characters processed")
    total_words: int = Field(default=0, description="Total words processed")
    average_chunk_size: float = Field(default=0.0, description="Average chunk size in characters")
    processing_time_seconds: float = Field(default=0.0, description="Total processing time")
    errors_count: int = Field(default=0, description="Number of errors encountered")
    skipped_count: int = Field(default=0, description="Number of documents skipped")
    
    @validator('total_documents', 'total_chunks', 'errors_count', 'skipped_count')
    def validate_non_negative_counts(cls, v):
        """Ensure counts are non-negative."""
        if v < 0:
            raise ValueError("Counts must be non-negative")
        return v
    
    @validator('processing_time_seconds')
    def validate_processing_time(cls, v):
        """Ensure processing time is non-negative."""
        if v < 0:
            raise ValueError("Processing time must be non-negative")
        return v
    
    def calculate_average_chunk_size(self):
        """Calculate and update average chunk size."""
        if self.total_chunks > 0:
            self.average_chunk_size = self.total_characters / self.total_chunks
        else:
            self.average_chunk_size = 0.0