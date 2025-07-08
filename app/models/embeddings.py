"""
Pydantic models for embeddings and vector operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    MINILM_L12_V2 = "sentence-transformers/all-MiniLM-L12-v2"
    MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"


class VectorSimilarityMetric(str, Enum):
    """Vector similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "l2"
    INNER_PRODUCT = "ip"


class EmbeddingStatus(str, Enum):
    """Status of embedding operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentEmbedding(BaseModel):
    """Document embedding with metadata."""
    
    id: str = Field(..., description="Unique embedding ID")
    document_id: str = Field(..., description="Source document ID")
    chunk_id: str = Field(..., description="Source chunk ID")
    content: str = Field(..., description="Original text content")
    embedding_vector: List[float] = Field(..., description="Embedding vector")
    model_name: str = Field(..., description="Embedding model used")
    vector_dimension: int = Field(..., description="Dimension of embedding vector")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('embedding_vector')
    def validate_embedding_vector(cls, v):
        """Validate embedding vector is not empty and contains valid floats."""
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding vector must contain only numeric values")
        
        return v
    
    @validator('vector_dimension')
    def validate_vector_dimension(cls, v, values):
        """Validate vector dimension matches embedding vector length."""
        if 'embedding_vector' in values and len(values['embedding_vector']) != v:
            raise ValueError("Vector dimension must match embedding vector length")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EmbeddingJob(BaseModel):
    """Embedding generation job."""
    
    id: str = Field(..., description="Unique job ID")
    document_ids: List[str] = Field(..., description="Document IDs to process")
    chunk_ids: List[str] = Field(..., description="Chunk IDs to process")
    model_name: str = Field(..., description="Embedding model to use")
    status: EmbeddingStatus = Field(default=EmbeddingStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Job progress (0.0-1.0)")
    total_chunks: int = Field(default=0, description="Total chunks to process")
    processed_chunks: int = Field(default=0, description="Chunks processed")
    failed_chunks: int = Field(default=0, description="Chunks that failed")
    embeddings_created: List[str] = Field(default_factory=list, description="Created embedding IDs")
    
    @validator('processed_chunks')
    def validate_processed_chunks(cls, v, values):
        """Validate processed chunks doesn't exceed total."""
        if 'total_chunks' in values and v > values['total_chunks']:
            raise ValueError("Processed chunks cannot exceed total chunks")
        return v
    
    def update_progress(self):
        """Update progress based on processed chunks."""
        if self.total_chunks > 0:
            self.progress = self.processed_chunks / self.total_chunks
        else:
            self.progress = 0.0


class VectorSearchQuery(BaseModel):
    """Vector similarity search query."""
    
    query_text: str = Field(..., min_length=1, description="Query text")
    query_embedding: Optional[List[float]] = Field(None, description="Pre-computed query embedding")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    include_content: bool = Field(default=True, description="Include original content in results")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    collection_name: Optional[str] = Field(None, description="Specific collection to search")


class VectorSearchResult(BaseModel):
    """Single vector search result."""
    
    id: str = Field(..., description="Embedding ID")
    document_id: str = Field(..., description="Source document ID")
    chunk_id: str = Field(..., description="Source chunk ID")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    content: Optional[str] = Field(None, description="Original text content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")
    embedding_vector: Optional[List[float]] = Field(None, description="Embedding vector")


class VectorSearchResponse(BaseModel):
    """Vector search response with multiple results."""
    
    query: str = Field(..., description="Original query text")
    results: List[VectorSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    collection_name: str = Field(..., description="Collection searched")
    model_name: str = Field(..., description="Embedding model used")
    
    @validator('total_results')
    def validate_total_results(cls, v, values):
        """Validate total results matches actual results count."""
        if 'results' in values and v != len(values['results']):
            raise ValueError("Total results must match actual results count")
        return v


class EmbeddingBatchRequest(BaseModel):
    """Request for batch embedding generation."""
    
    chunk_ids: List[str] = Field(..., min_items=1, description="Chunk IDs to process")
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch processing size")
    overwrite_existing: bool = Field(default=False, description="Overwrite existing embeddings")
    collection_name: Optional[str] = Field(None, description="Target collection name")


class EmbeddingBatchResponse(BaseModel):
    """Response for batch embedding generation."""
    
    job_id: str = Field(..., description="Job ID for tracking")
    total_chunks: int = Field(..., description="Total chunks to process")
    estimated_time_minutes: float = Field(..., description="Estimated processing time")
    status: EmbeddingStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")


class VectorStoreStats(BaseModel):
    """Vector store statistics."""
    
    collection_name: str = Field(..., description="Collection name")
    total_embeddings: int = Field(..., description="Total embeddings in collection")
    unique_documents: int = Field(..., description="Unique documents represented")
    total_chunks: int = Field(..., description="Total chunks embedded")
    vector_dimension: int = Field(..., description="Vector dimension")
    model_name: str = Field(..., description="Embedding model used")
    last_updated: datetime = Field(..., description="Last update timestamp")
    storage_size_mb: float = Field(..., description="Storage size in MB")
    index_size_mb: float = Field(..., description="Index size in MB")


class EmbeddingHealth(BaseModel):
    """Embedding service health status."""
    
    service_status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether embedding model is loaded")
    model_name: str = Field(..., description="Current embedding model")
    vector_store_connected: bool = Field(..., description="Vector store connection status")
    last_embedding_time: Optional[datetime] = Field(None, description="Last successful embedding")
    total_embeddings_generated: int = Field(default=0, description="Total embeddings generated")
    average_embedding_time_ms: float = Field(default=0.0, description="Average embedding time")
    errors_last_hour: int = Field(default=0, description="Errors in last hour")