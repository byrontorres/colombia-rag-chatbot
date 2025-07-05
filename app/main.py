"""
Main FastAPI application entry point.
Colombia RAG Chatbot - Production-ready API server.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config.settings import settings
from app.config.logging import setup_logging, log_error, logger
from app.core.exceptions import (
    ColombiaRAGException, 
    get_status_code, 
    create_error_response
)


# Initialize logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown operations.
    """
    # Startup
    logger.info(
        "Starting Colombia RAG Chatbot",
        version=settings.app_version,
        environment=settings.environment
    )
    
    try:
        # TODO: Initialize RAG system, vector store, models, etc.
        # This will be implemented in subsequent phases
        logger.info("Application startup completed successfully")
        yield
        
    except Exception as e:
        log_error(e, {"phase": "startup"})
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Colombia RAG Chatbot")
        # TODO: Cleanup resources (close DB connections, etc.)


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="RAG-based chatbot providing information about Colombia from Wikipedia",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None,
)


# --- Middleware: fuerza charset UTF-8 en JSON ---------------------------------
from fastapi import Request
from fastapi.responses import Response

@app.middleware("http")
async def ensure_utf8_json(request: Request, call_next):
    response: Response = await call_next(request)
    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response
# ------------------------------------------------------------------------------

from app.api.endpoints import chat, health   # ← importa los routers

app.include_router(chat)          
app.include_router(health)           


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Custom exception handlers
@app.exception_handler(ColombiaRAGException)
async def colombia_rag_exception_handler(
    request: Request, 
    exc: ColombiaRAGException
) -> JSONResponse:
    """Handle custom Colombia RAG exceptions."""
    status_code = get_status_code(exc)
    
    log_error(
        exc, 
        {
            "request_path": str(request.url.path),
            "request_method": request.method,
            "status_code": status_code
        }
    )
    
    return JSONResponse(
        status_code=status_code,
        content=create_error_response(exc, status_code)
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    logger.warning(
        "Request validation failed",
        path=str(request.url.path),
        method=request.method,
        errors=exc.errors()
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "error_code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "context": {"details": exc.errors()}
            },
            "status_code": 422
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, 
    exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(
        "HTTP exception occurred",
        path=str(request.url.path),
        method=request.method,
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "error_code": "HTTP_ERROR",
                "message": exc.detail,
                "context": {}
            },
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, 
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    log_error(
        exc,
        {
            "request_path": str(request.url.path),
            "request_method": request.method,
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "context": {} if settings.is_production else {"detail": str(exc)}
            },
            "status_code": 500
        }
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancers.
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
        "service": "colombia-rag-chatbot"
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing basic application information.
    """
    return {
        "message": "Welcome to Colombia RAG Chatbot",
        "description": "RAG-based chatbot providing information about Colombia",
        "version": settings.app_version,
        "docs_url": "/docs" if settings.is_development else "Documentation disabled in production",
        "health_check": "/health"
    }

# Include API router
from app.api import api_router
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=settings.is_development,
    )