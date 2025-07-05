"""
Response generation service using LangChain and Ollama for RAG implementation.
"""

import time
from typing import List, Dict, Optional, Any
from datetime import datetime

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.config.settings import settings
from app.config.logging import logger, log_error
from app.core.exceptions import (
    GenerationError,
    ModelError,
    InvalidQueryError,
    QueryNotColombiaRelatedError
)
from app.models.embeddings import VectorSearchResult
from app.services.retrieval_service import RetrievalService



class ResponseGenerationService:

    """
    Service for generating contextual responses using retrieved documents and LLM.
    """
    _llm_instance = None
    
    @classmethod
    def get_llm(cls):
        if cls._llm_instance is None:
                cls._llm_instance = ChatOllama(
                    model=settings.llm_model,
                    base_url=settings.ollama_base_url,
                    temperature=0.2,
                    max_tokens=120,
                    timeout=60,
                    num_predict=120,
                    repeat_penalty=1.15,
                    top_p=0.9,
                    top_k=20
                )
        return cls._llm_instance
    
    @classmethod
    def get_chain(cls):
        if not hasattr(cls, '_chain_instance') or cls._chain_instance is None:
            cls._chain_instance = LLMChain(
                llm=cls.get_llm(),
                prompt=cls._get_prompt_template(),
                verbose=False
            )
        return cls._chain_instance
    
    @classmethod
    def _get_prompt_template(cls):
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""Eres un asistente especializado en Colombia. Responde SOLO con información del contexto.

CONTEXTO:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
- Usa ÚNICAMENTE información del contexto
- Si no hay información, di "No encuentro esa información en el contexto"
- Máximo 80 palabras
- Español claro y directo

RESPUESTA:"""
        )
    
    def __init__(self, retrieval_service: RetrievalService = None):
        """Initialize the response generation service."""
        
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm = self.get_llm()
        self.chain = self.get_chain()
        self.model_loaded = True
        

        
        logger.info("ResponseGenerationService initialized with singleton LLM")
    

    
    def generate_response(
        self,
        query: str,
        max_context_length: int = None,
        top_k: int = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Generate a response for the given query using RAG."""
        
        try:
            start_time = time.time()
            max_context_length = max_context_length or settings.max_context_length
            top_k = top_k or settings.top_k_documents
            
            logger.info(f"Generating response for query", query=query[:50])
            
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            retrieval_response = self.retrieval_service.retrieve_documents(
                query=query,
                top_k=top_k,
                validate_colombia_relevance=True,
                include_context=True
            )
            retrieval_time = time.time() - retrieval_start
            
            if not retrieval_response.results:
                return {
                    "query": query,
                    "answer": "No encontré información relevante sobre Colombia para responder tu pregunta.",
                    "sources": [],
                    "retrieval_results": 0,
                    "retrieval_time_ms": retrieval_time * 1000,
                    "generation_time_ms": 0,
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "metadata": {
                        "model_used": settings.llm_model,
                        "context_length": 0,
                        "top_k_documents": 0
                    }
                }
            
            # Step 2: Prepare context from retrieved documents
            context = self._prepare_context(
                retrieval_response.results,
                max_context_length
            )
            
            # Step 3: Load model if not already loaded
            #self._load_model()

            
            
            # Step 4: Generate response
            generation_start = time.time()
            response = self.chain.invoke({
                "context": context,
                "question": query
            })
            
            # Extract text from response
            if isinstance(response, dict):
                answer_text = response.get("text", str(response))
            else:
                answer_text = str(response)
                
            generation_time = time.time() - generation_start

            # Step 4.5: Validate response is based on retrieved context
            if not self._validate_response_relevance(answer_text, retrieval_response.results):
                return {
                    "query": query,
                    "answer": "No encuentro información suficiente sobre esa consulta en el contexto de Colombia disponible.",
                    "sources": [],
                    "retrieval_results": len(retrieval_response.results),
                    "retrieval_time_ms": retrieval_time * 1000,
                    "generation_time_ms": generation_time * 1000,
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "metadata": {
                        "model_used": settings.llm_model,
                        "context_length": len(context),
                        "top_k_documents": len(retrieval_response.results),
                        "validation_failed": True
                    }
                }
            
            # Step 5: Prepare sources information
            sources = []
            if include_sources:
                sources = self._prepare_sources(retrieval_response.results)
            
            result = {
                "query": query,
                "answer": answer_text.strip(),
                "sources": sources,
                "retrieval_results": len(retrieval_response.results),
                "retrieval_time_ms": retrieval_time * 1000,
                "generation_time_ms": generation_time * 1000,
                "total_time_ms": (time.time() - start_time) * 1000,
                "metadata": {
                    "model_used": settings.llm_model,
                    "context_length": len(context),
                    "top_k_documents": len(retrieval_response.results)
                }
            }
            
            logger.info(
                f"Response generated successfully",
                query=query[:50],
                retrieval_results=len(retrieval_response.results),
                generation_time_ms=generation_time * 1000,
                total_time_ms=result["total_time_ms"]
            )
            
            return result
        
        except QueryNotColombiaRelatedError:
            # Deja que el endpoint maneje esta excepción y responda 400
            raise
            
        except Exception as e:
            error_msg = f"Failed to generate response for query '{query}': {str(e)}"
            log_error(e, {"query": query})
            raise GenerationError(error_msg) from e
    
    def _prepare_context(
        self, 
        results: List[VectorSearchResult], 
        max_length: int
    ) -> str:
        """Prepare context from retrieval results."""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            if not result.content:
                continue
            
            # Add source info
            section = result.metadata.get('source_section', 'Sección desconocida') if result.metadata else 'Sección desconocida'
            similarity = result.similarity_score
            
            content_with_source = f"[Fuente {i+1} - {section} (relevancia: {similarity:.2f})]\n{result.content}\n"
            
            # Check if adding this content would exceed max length
            if current_length + len(content_with_source) > max_length:
                # Try to add a truncated version
                remaining_space = max_length - current_length - 100  # Leave some buffer
                if remaining_space > 200:  # Only if there's meaningful space left
                    truncated_content = result.content[:remaining_space] + "..."
                    content_with_source = f"[Fuente {i+1} - {section} (relevancia: {similarity:.2f})]\n{truncated_content}\n"
                    context_parts.append(content_with_source)
                break
            
            context_parts.append(content_with_source)
            current_length += len(content_with_source)
        
        return "\n".join(context_parts)
    
    def _prepare_sources(self, results: List[VectorSearchResult]) -> List[str]:
        """
        Devuelve únicamente las URLs (str) de cada VectorSearchResult.
        Si no encuentra URL reconocible, hace str(result) para no perder la referencia.
        """

        urls: list[str] = []

        for r in results:
            meta = r.metadata or {}

            url = (
                meta.get("wiki_url")
                or meta.get("source_url")
                or meta.get("url")
                or meta.get("source")
                or meta.get("page")
                or meta.get("link")
            )

            urls.append(url if url else str(r))

        unique_ordered = list(dict.fromkeys(urls))
        return unique_ordered
    
    def _validate_response_relevance(
        self, 
        response_text: str, 
        retrieval_results: List[VectorSearchResult]
    ) -> bool:
        """
        Validate that the response is based on retrieved context with sufficient relevance.
        
        Args:
            response_text: Generated response text
            retrieval_results: List of retrieved documents
            
        Returns:
            bool: True if response meets relevance criteria, False otherwise
        """
        if not retrieval_results:
            return False
        
        # Check minimum similarity threshold
        max_similarity = max(
            result.similarity_score for result in retrieval_results
        )
        
        # Require at least one document above similarity threshold
        if max_similarity < settings.similarity_threshold:
            logger.warning(
                "Response validation failed: insufficient document relevance",
                max_similarity=max_similarity,
                threshold=settings.similarity_threshold,
                response_preview=response_text[:100]
            )
            return False
        
        # Check for generic/evasive responses that might indicate hallucination
        evasive_patterns = [
            "no encuentro esa información",
            "no hay información",
            "no puedo responder",
            "información no disponible"
        ]
        
        response_lower = response_text.lower()
        if any(pattern in response_lower for pattern in evasive_patterns):
            # If the model itself says it can't find info, respect that
            return False
        
        # Additional check: response should be reasonably substantial
        if len(response_text.strip()) < 20:
            logger.warning(
                "Response validation failed: response too short",
                response_length=len(response_text),
                response_text=response_text
            )
            return False
        
        logger.debug(
            "Response validation passed",
            max_similarity=max_similarity,
            response_length=len(response_text)
        )
        
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the response generation service."""
        
        health_status = {
            "service_status": "healthy",
            "model_loaded": self.model_loaded,
            "model_name": settings.llm_model,
            "last_check": datetime.utcnow().isoformat()
        }
        
        try:
            # Test retrieval service
            retrieval_health = self.retrieval_service.health_check()
            health_status["retrieval_service"] = retrieval_health.get("service_status", "unknown")
            
            # Test LLM if loaded
            if self.model_loaded:
                try:
                    test_start = time.time()
                    test_response = self.generate_response(
                        "¿Qué es Colombia?",
                        top_k=1
                    )
                    test_time = time.time() - test_start
                    
                    health_status.update({
                        "generation_test": "success",
                        "test_response_length": len(test_response.get("answer", "")),
                        "test_time_ms": test_time * 1000
                    })
                    
                except Exception as e:
                    health_status.update({
                        "generation_test": "failed",
                        "test_error": str(e),
                        "service_status": "degraded"
                    })
            else:
                health_status["generation_test"] = "skipped - model not loaded"
        
        except Exception as e:
            health_status.update({
                "service_status": "degraded",
                "error": str(e)
            })
        
        return health_status