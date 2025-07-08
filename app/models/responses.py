"""
Pydantic models for API responses.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

from .documents import ProcessedDocument, DocumentChunk, ProcessingStats, ProcessingStatus


class BaseResponse(BaseModel):
    """Base response model for all API endpoints."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ExtractionResponse(BaseResponse):
    """Response model for data extraction operations."""
    job_id: str = Field(..., description="Extraction job ID")
    documents_extracted: int = Field(..., description="Number of documents extracted")
    sections_processed: List[str] = Field(..., description="List of sections processed")
    total_characters: int = Field(..., description="Total characters extracted")
    extraction_time_seconds: float = Field(..., description="Time taken for extraction")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully extracted content from Colombia Wikipedia page",
                "timestamp": "2024-01-15T10:30:00Z",
                "job_id": "ext_20240115_103000",
                "documents_extracted": 25,
                "sections_processed": ["Historia", "Geografía", "Política", "Economía"],
                "total_characters": 125000,
                "extraction_time_seconds": 5.2
            }
        }


class ProcessingResponse(BaseResponse):
    """Response model for text processing operations."""
    document_id: str = Field(..., description="Processed document ID")
    original_length: int = Field(..., description="Original content length")
    processed_length: int = Field(..., description="Processed content length")
    words_count: int = Field(..., description="Number of words in processed content")
    processing_time_seconds: float = Field(..., description="Time taken for processing")
    changes_made: List[str] = Field(..., description="List of changes made during processing")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Document processed successfully",
                "timestamp": "2024-01-15T10:35:00Z",
                "document_id": "doc_colombia_historia_001",
                "original_length": 5000,
                "processed_length": 4200,
                "words_count": 750,
                "processing_time_seconds": 1.2,
                "changes_made": ["removed_html_tags", "normalized_whitespace", "removed_references"]
            }
        }


class ChunkingResponse(BaseResponse):
    """Response model for document chunking operations."""
    document_id: str = Field(..., description="Source document ID")
    chunks_created: int = Field(..., description="Number of chunks created")
    total_characters: int = Field(..., description="Total characters in all chunks")
    average_chunk_size: float = Field(..., description="Average chunk size in characters")
    chunking_time_seconds: float = Field(..., description="Time taken for chunking")
    chunk_ids: List[str] = Field(..., description="List of created chunk IDs")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Document chunked successfully",
                "timestamp": "2024-01-15T10:40:00Z",
                "document_id": "doc_colombia_historia_001",
                "chunks_created": 4,
                "total_characters": 4200,
                "average_chunk_size": 1050.0,
                "chunking_time_seconds": 0.5,
                "chunk_ids": ["chunk_001", "chunk_002", "chunk_003", "chunk_004"]
            }
        }


class DocumentResponse(BaseResponse):
    """Response model for single document operations."""
    document: Union[ProcessedDocument, DocumentChunk] = Field(..., description="Document or chunk data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Document retrieved successfully",
                "timestamp": "2024-01-15T10:45:00Z",
                "document": {
                    "id": "doc_colombia_historia_001",
                    "cleaned_content": "Colombia es un país ubicado en...",
                    "word_count": 750,
                    "character_count": 4200
                }
            }
        }


class DocumentListResponse(BaseResponse):
    """Response model for multiple documents."""
    documents: List[Union[ProcessedDocument, DocumentChunk]] = Field(..., description="List of documents or chunks")
    total_count: int = Field(..., description="Total number of documents available")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=20, description="Number of items per page")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Documents retrieved successfully",
                "timestamp": "2024-01-15T10:50:00Z",
                "documents": [],
                "total_count": 25,
                "page": 1,
                "page_size": 20
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""
    chunk_id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Chunk content")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "chunk_colombia_geografia_001",
                "content": "Colombia está situada en la esquina noroeste de América del Sur...",
                "similarity_score": 0.92,
                "metadata": {
                    "section": "Geografía",
                    "source_url": "https://es.wikipedia.org/wiki/Colombia"
                }
            }
        }


class SearchResponse(BaseResponse):
    """Response model for search operations."""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time_seconds: float = Field(..., description="Time taken for search")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Search completed successfully",
                "timestamp": "2024-01-15T10:55:00Z",
                "query": "geografía de Colombia",
                "results": [],
                "total_results": 5,
                "search_time_seconds": 0.1
            }
        }


class JobStatusResponse(BaseResponse):
    """Response model for job status inquiries."""
    job_id: str = Field(..., description="Job ID")
    status: ProcessingStatus = Field(..., description="Current job status")
    progress_percentage: float = Field(..., description="Job progress (0-100)")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    current_operation: Optional[str] = Field(None, description="Current operation being performed")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Job status retrieved successfully",
                "timestamp": "2024-01-15T11:00:00Z",
                "job_id": "ext_20240115_103000",
                "status": "processing",
                "progress_percentage": 65.0,
                "started_at": "2024-01-15T10:30:00Z",
                "estimated_completion": "2024-01-15T11:05:00Z",
                "current_operation": "Processing section: Economía"
            }
        }


class StatsResponse(BaseResponse):
    """Response model for system statistics."""
    stats: ProcessingStats = Field(..., description="Processing statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Statistics retrieved successfully",
                "timestamp": "2024-01-15T11:05:00Z",
                "stats": {
                    "total_documents": 25,
                    "total_chunks": 120,
                    "total_characters": 150000,
                    "total_words": 25000,
                    "average_chunk_size": 1250.0,
                    "processing_time_seconds": 45.0,
                    "errors_count": 0,
                    "skipped_count": 2
                }
            }
        }


class HealthResponse(BaseResponse):
    """Response model for health check."""
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    database_status: str = Field(..., description="Database connection status")
    last_data_update: Optional[datetime] = Field(None, description="Last data update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "System is healthy",
                "timestamp": "2024-01-15T11:10:00Z",
                "version": "1.0.0",
                "environment": "development",
                "uptime_seconds": 3600.0,
                "database_status": "connected",
                "last_data_update": "2024-01-15T09:00:00Z"
            }
        }


class ErrorResponse(BaseResponse):
    """Response model for errors."""
    error_code: str = Field(..., description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "message": "Document not found",
                "timestamp": "2024-01-15T11:15:00Z",
                "error_code": "DOCUMENT_NOT_FOUND",
                "error_details": {
                    "document_id": "doc_invalid_001",
                    "suggestion": "Check if document ID is correct"
                }
            }
        }

class QueryResponse(BaseModel):
    """Response model for query requests."""
    
    query: str = Field(..., description="Original query from user")
    answer: str = Field(..., description="Generated answer about Colombia") 
    sources: List[str] = Field(default=[], description="List of source URLs")
    retrieval_results: int = Field(..., description="Number of documents retrieved")
    retrieval_time_ms: float = Field(..., description="Time spent on document retrieval")
    generation_time_ms: float = Field(..., description="Time spent on answer generation")
    total_time_ms: float = Field(..., description="Total processing time")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

