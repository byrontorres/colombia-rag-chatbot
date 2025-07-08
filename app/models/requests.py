"""
Pydantic models for API requests.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class ExtractionRequest(BaseModel):
    """Request model for data extraction."""
    source_url: str = Field(..., description="URL to extract data from")
    force_refresh: bool = Field(default=False, description="Force refresh even if cached data exists")
    extract_sections: Optional[List[str]] = Field(
        default=None, 
        description="Specific sections to extract (if None, extract all)"
    )
    max_content_length: Optional[int] = Field(
        default=None, 
        description="Maximum content length to extract"
    )
    
    @validator('source_url')
    def validate_source_url(cls, v):
        """Ensure source URL is valid and points to Colombia Wikipedia."""
        if not v or not v.startswith(('http://', 'https://')):
            raise ValueError("Source URL must be a valid HTTP/HTTPS URL")
        
        # Validate it's the Colombia Wikipedia page
        expected_paths = [
            'es.wikipedia.org/wiki/Colombia',
            'wikipedia.org/wiki/Colombia'
        ]
        
        if not any(path in v for path in expected_paths):
            raise ValueError("Source URL must point to Colombia's Wikipedia page")
        
        return v
    
    @validator('max_content_length')
    def validate_max_content_length(cls, v):
        """Ensure max content length is reasonable."""
        if v is not None and v <= 0:
            raise ValueError("Max content length must be positive")
        if v is not None and v > 10_000_000:  # 10MB
            raise ValueError("Max content length too large (max 10MB)")
        return v


class ProcessingRequest(BaseModel):
    """Request model for text processing."""
    document_id: str = Field(..., description="Document ID to process")
    clean_html: bool = Field(default=True, description="Remove HTML tags")
    normalize_whitespace: bool = Field(default=True, description="Normalize whitespace")
    remove_references: bool = Field(default=True, description="Remove reference markers")
    min_paragraph_length: int = Field(default=50, description="Minimum paragraph length to keep")
    language: str = Field(default="es", description="Expected language of content")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        """Ensure document ID is valid."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Document ID must be at least 3 characters long")
        return v.strip()
    
    @validator('min_paragraph_length')
    def validate_min_paragraph_length(cls, v):
        """Ensure minimum paragraph length is reasonable."""
        if v < 10:
            raise ValueError("Minimum paragraph length must be at least 10 characters")
        if v > 1000:
            raise ValueError("Minimum paragraph length too large (max 1000)")
        return v


class ChunkingRequest(BaseModel):
    """Request model for document chunking."""
    document_id: str = Field(..., description="Document ID to chunk")
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in characters")
    preserve_sentences: bool = Field(default=True, description="Try to preserve sentence boundaries")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in characters")
    max_chunk_size: int = Field(default=2000, description="Maximum chunk size in characters")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        """Ensure document ID is valid."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Document ID must be at least 3 characters long")
        return v.strip()
    
    @validator('chunk_size', 'chunk_overlap', 'min_chunk_size', 'max_chunk_size')
    def validate_positive_sizes(cls, v):
        """Ensure sizes are positive."""
        if v <= 0:
            raise ValueError("Chunk sizes must be positive")
        return v
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is less than chunk size."""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v
    
    @validator('max_chunk_size')
    def validate_max_chunk_size(cls, v, values):
        """Ensure max chunk size is larger than min and target sizes."""
        if 'min_chunk_size' in values and v < values['min_chunk_size']:
            raise ValueError("Max chunk size must be larger than min chunk size")
        if 'chunk_size' in values and v < values['chunk_size']:
            raise ValueError("Max chunk size must be larger than target chunk size")
        return v


class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str = Field(..., description="User query about Colombia")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    
    @validator('query')
    def validate_query(cls, v):
        """Ensure query is valid."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        query_clean = v.strip()
        if len(query_clean) < 3:
            raise ValueError("Query must be at least 3 characters long")
        if len(query_clean) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        
        return query_clean
    
    @validator('max_results')
    def validate_max_results(cls, v):
        """Ensure max results is reasonable."""
        if v <= 0:
            raise ValueError("Max results must be positive")
        if v > 50:
            raise ValueError("Max results too large (max 50)")
        return v
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        """Ensure similarity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v


class UpdateDataRequest(BaseModel):
    """Request model for updating data sources."""
    force_full_refresh: bool = Field(default=False, description="Force complete data refresh")
    update_sections: Optional[List[str]] = Field(
        default=None, 
        description="Specific sections to update"
    )
    backup_existing: bool = Field(default=True, description="Backup existing data before update")
    notify_completion: bool = Field(default=False, description="Send notification when complete")


class BulkProcessingRequest(BaseModel):
    """Request model for bulk processing operations."""
    operation_type: str = Field(..., description="Type of bulk operation")
    document_ids: List[str] = Field(..., description="List of document IDs to process")
    config: Dict[str, Any] = Field(default_factory=dict, description="Operation configuration")
    
    @validator('operation_type')
    def validate_operation_type(cls, v):
        """Ensure operation type is valid."""
        valid_operations = ['extract', 'process', 'chunk', 'reindex', 'cleanup']
        if v not in valid_operations:
            raise ValueError(f"Operation type must be one of: {valid_operations}")
        return v
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        """Ensure document IDs list is valid."""
        if not v:
            raise ValueError("Document IDs list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Too many document IDs (max 1000)")
        
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Document IDs list contains duplicates")
        
        # Validate each ID
        for doc_id in v:
            if not doc_id or len(doc_id.strip()) < 3:
                raise ValueError("Each document ID must be at least 3 characters long")
        
        return v