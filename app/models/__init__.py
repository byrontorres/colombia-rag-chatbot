"""
Pydantic models for the Colombia RAG Chatbot application.
"""

from .documents import (
    ContentType,
    ProcessingStatus,
    SourceMetadata,
    RawDocument,
    ProcessedDocument,
    DocumentChunk,
    ExtractionJob,
    ProcessingStats,
)

from .requests import (
    ExtractionRequest,
    ProcessingRequest,
    ChunkingRequest,
    QueryRequest,
    UpdateDataRequest,
    BulkProcessingRequest,
)

from .responses import (
    BaseResponse,
    ExtractionResponse,
    ProcessingResponse,
    ChunkingResponse,
    DocumentResponse,
    DocumentListResponse,
    SearchResult,
    SearchResponse,
    JobStatusResponse,
    StatsResponse,
    HealthResponse,
    ErrorResponse,
)

from .embeddings import (
    EmbeddingModel,
    VectorSimilarityMetric,
    EmbeddingStatus,
    DocumentEmbedding,
    EmbeddingJob,
    VectorSearchQuery,
    VectorSearchResult,
    VectorSearchResponse,
    EmbeddingBatchRequest,
    EmbeddingBatchResponse,
    VectorStoreStats,
    EmbeddingHealth,
)

__all__ = [
    # Document models
    "ContentType",
    "ProcessingStatus", 
    "SourceMetadata",
    "RawDocument",
    "ProcessedDocument",
    "DocumentChunk",
    "ExtractionJob",
    "ProcessingStats",
    
    # Request models
    "ExtractionRequest",
    "ProcessingRequest", 
    "ChunkingRequest",
    "QueryRequest",
    "UpdateDataRequest",
    "BulkProcessingRequest",
    
    # Response models
    "BaseResponse",
    "ExtractionResponse",
    "ProcessingResponse",
    "ChunkingResponse", 
    "DocumentResponse",
    "DocumentListResponse",
    "SearchResult",
    "SearchResponse",
    "JobStatusResponse",
    "StatsResponse",
    "HealthResponse",
    "ErrorResponse",

    # Embedding models
    "EmbeddingModel",
    "VectorSimilarityMetric",
    "EmbeddingStatus",
    "DocumentEmbedding",
    "EmbeddingJob",
    "VectorSearchQuery",
    "VectorSearchResult",
    "VectorSearchResponse",
    "EmbeddingBatchRequest",
    "EmbeddingBatchResponse",
    "VectorStoreStats",
    "EmbeddingHealth",
]