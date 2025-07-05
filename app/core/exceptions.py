"""
Custom exceptions for the Colombia RAG Chatbot application.
"""

from typing import Any, Dict, Optional


class ColombiaRAGException(Exception):
    """Base exception for all application-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = None, 
        context: Dict[str, Any] = None
    ) -> None:
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


# Configuration Exceptions
class ConfigurationError(ColombiaRAGException):
    """Raised when there's a configuration issue."""
    pass


# Data Processing Exceptions
class DataExtractionError(ColombiaRAGException):
    """Raised when data extraction from Wikipedia fails."""
    pass


class DataProcessingError(ColombiaRAGException):
    """Raised when data processing operations fail."""
    pass


# Vector Database Exceptions
class VectorStoreError(ColombiaRAGException):
    """Base exception for vector store operations."""
    pass


class DocumentNotFoundError(VectorStoreError):
    """Raised when requested documents are not found."""
    pass


# Model Exceptions
class ModelError(ColombiaRAGException):
    """Base exception for model-related errors."""
    pass


# RAG System Exceptions
class RAGError(ColombiaRAGException):
    """Base exception for RAG system errors."""
    pass


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""
    pass


class GenerationError(RAGError):
    """Raised when response generation fails."""
    pass


# Query Processing Exceptions
class QueryError(ColombiaRAGException):
    """Base exception for query processing errors."""
    pass


class InvalidQueryError(QueryError):
    """Raised when query is invalid or malformed."""
    pass


class QueryNotColombiaRelatedError(QueryError):
    """Raised when query is not related to Colombia."""
    pass


# API Exceptions
class APIError(ColombiaRAGException):
    """Base exception for API-related errors."""
    pass


class ValidationError(APIError):
    """Raised when request validation fails."""
    pass


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAPPING = {
    ValidationError: 400,
    InvalidQueryError: 400,
    QueryNotColombiaRelatedError: 400,
    DocumentNotFoundError: 404,
    ConfigurationError: 500,
    DataExtractionError: 500,
    DataProcessingError: 500,
    VectorStoreError: 500,
    ModelError: 500,
    RAGError: 500,
}


def get_status_code(exception: ColombiaRAGException) -> int:
    """Get appropriate HTTP status code for an exception."""
    return EXCEPTION_STATUS_MAPPING.get(type(exception), 500)


def create_error_response(
    error: ColombiaRAGException, 
    status_code: int = 500
) -> Dict[str, Any]:
    """Create standardized error response for API endpoints."""
    return {
        "success": False,
        "error": error.to_dict(),
        "status_code": status_code
    }

# External Service Exceptions
class ExternalServiceError(ColombiaRAGException):
    """Base exception for external service errors."""
    pass


class WikipediaError(ExternalServiceError):
    """Raised when Wikipedia API operations fail."""
    pass


class NetworkError(ExternalServiceError):
    """Raised when network operations fail."""
    pass


class TimeoutError(ExternalServiceError):
    """Raised when operations timeout."""
    pass