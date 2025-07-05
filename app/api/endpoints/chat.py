from fastapi import APIRouter, HTTPException, Depends
import time

from app.models.requests import QueryRequest
from app.models.responses import QueryResponse
from app.services import ResponseGenerationService          
from app.core.exceptions import GenerationError            
from app.config.logging import logger
from app.config.settings import get_settings, Settings
from app.core.exceptions import GenerationError, QueryNotColombiaRelatedError

router = APIRouter(prefix="/query", tags=["query"])
rgs = ResponseGenerationService()


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    settings: Settings = Depends(get_settings)
) -> QueryResponse:
    """
    Query documents about Colombia and get relevant information.
    Genera respuesta con RAG (retrieval + LLM) usando información exclusiva de Wikipedia Colombia.
    """
    start_time = time.time()

    try:
        logger.info("Processing query request", extra={"query": request.query[:50]})

        rag_result = rgs.generate_response(
            query=request.query,
            top_k=settings.top_k_documents,
            include_sources=True
        )

        # --- normaliza las fuentes a lista de cadenas ---------------------------------
        source_urls: list[str] = []
        for src in rag_result["sources"]:
            if isinstance(src, str):
                source_urls.append(src)

            elif isinstance(src, dict):
                # intenta distintas claves donde podríamos tener la URL
                url = (
                    src.get("url")
                    or src.get("source")
                    or src.get("page")
                    or src.get("link")
                    or src.get("metadata", {}).get("wiki_url")
                )
                source_urls.append(url) if url else source_urls.append(str(src))

            else:
                source_urls.append(str(src))
        # -----------------------------------------------------------------------------

        total_time = rag_result["total_time_ms"]

        response = QueryResponse(
            query              = rag_result["query"],
            answer             = rag_result["answer"],
            sources            = source_urls,
            retrieval_results  = rag_result["retrieval_results"],
            retrieval_time_ms  = rag_result["retrieval_time_ms"],
            generation_time_ms = rag_result["generation_time_ms"],
            total_time_ms      = total_time,
            metadata           = {
                "model_used"     : rag_result["metadata"]["model_used"],
                "context_length" : rag_result["metadata"]["context_length"],
                "top_k_documents": rag_result["metadata"]["top_k_documents"]
            }
        )

        logger.info(
            "Query processed successfully",
            extra={
                "query"            : request.query[:50],
                "retrieval_results": rag_result["retrieval_results"],
                "total_time_ms"    : total_time
            }
        )

        return response
    
    except QueryNotColombiaRelatedError:
        # 200 con JSON válido
        return QueryResponse(
            query=request.query,
            answer="Lo siento, solo puedo responder preguntas relacionadas con Colombia.",
            sources=[],
            retrieval_results=0,
            retrieval_time_ms=0.0,
            generation_time_ms=0.0,
            total_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "model_used": None,
                "context_length": 0,
                "top_k_documents": 0,
                "in_scope": False
            }
        )

    except GenerationError as e:
        logger.error("Generation error", extra={"error": str(e), "query": request.query[:50]})
        raise HTTPException(status_code=500, detail=f"Error en generación: {str(e)}")

    except Exception as e:
        logger.error("Unexpected error", extra={"error": str(e), "query": request.query[:50]})
        raise HTTPException(status_code=500, detail="Error interno del servidor")