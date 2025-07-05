"""
Embedding service for generating and managing document embeddings.
"""

import time
import uuid
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import settings
from app.config.logging import logger, log_error
from app.core.exceptions import (
    ModelError,
    DataProcessingError,
    ConfigurationError
)
from app.models.documents import DocumentChunk
from app.models.embeddings import (
    DocumentEmbedding,
    EmbeddingJob,
    EmbeddingStatus,
    EmbeddingModel,
    EmbeddingBatchRequest,
    EmbeddingBatchResponse
)


class EmbeddingService:
    """
    Service for generating embeddings from document chunks using SentenceTransformers.
    """
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding service."""
        
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self.model_loaded = False
        self.vector_dimension = None
        self.total_embeddings_generated = 0
        self.embedding_times = []
        
        logger.info(f"Initializing EmbeddingService with model: {self.model_name}")
        
        # Validate model name
        if self.model_name not in [model.value for model in EmbeddingModel]:
            logger.warning(f"Model {self.model_name} not in predefined list, proceeding anyway")
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        
        if self.model_loaded:
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()
            
            self.model = SentenceTransformer(self.model_name)
            
            # Get vector dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            self.vector_dimension = len(test_embedding[0])
            
            load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.info(
                f"Model loaded successfully",
                model_name=self.model_name,
                vector_dimension=self.vector_dimension,
                load_time=load_time
            )
            
        except Exception as e:
            error_msg = f"Failed to load embedding model {self.model_name}: {str(e)}"
            log_error(e, {"model_name": self.model_name})
            raise ModelError(error_msg) from e
    
    def _generate_embedding_id(self, chunk_id: str, model_name: str) -> str:
        """Generate a unique embedding ID."""
        
        # Create deterministic ID based on chunk ID and model
        content = f"{chunk_id}_{model_name}_{datetime.utcnow().isoformat()}"
        hash_object = hashlib.md5(content.encode())
        return f"emb_{hash_object.hexdigest()[:16]}"
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation."""
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Basic preprocessing
        text = text.strip()
        
        # Limit text length to avoid model issues
        max_length = 8192  # Most sentence transformers have this limit
        if len(text) > max_length:
            logger.warning(
                f"Text truncated from {len(text)} to {max_length} characters",
                original_length=len(text)
            )
            text = text[:max_length]
        
        return text
    
    def generate_embedding(self, chunk: DocumentChunk) -> DocumentEmbedding:
        """Generate embedding for a single document chunk."""
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            start_time = time.time()
            
            # Preprocess text
            processed_text = self._preprocess_text(chunk.content)
            
            logger.debug(
                f"Generating embedding for chunk {chunk.id}",
                content_length=len(processed_text),
                chunk_id=chunk.id
            )
            
            # Generate embedding
            embedding_vector = self.model.encode(
                [processed_text],
                show_progress_bar=False,
                normalize_embeddings=True
            )[0]
            
            # Convert to Python list
            embedding_list = embedding_vector.tolist()
            
            embedding_time = time.time() - start_time
            self.embedding_times.append(embedding_time)
            self.total_embeddings_generated += 1
            
            # Create embedding object
            embedding_id = self._generate_embedding_id(chunk.id, self.model_name)
            
            # Prepare metadata
            metadata = {
                "source_section": chunk.metadata.section,
                "source_url": chunk.metadata.url,
                "chunk_index": chunk.chunk_index,
                "word_count": chunk.word_count,
                "character_count": chunk.character_count,
                "embedding_time_ms": embedding_time * 1000,
                "model_version": getattr(self.model, '_model_name', self.model_name)
            }
            
            embedding = DocumentEmbedding(
                id=embedding_id,
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                content=processed_text,
                embedding_vector=embedding_list,
                model_name=self.model_name,
                vector_dimension=self.vector_dimension,
                metadata=metadata
            )
            
            logger.debug(
                f"Embedding generated successfully",
                embedding_id=embedding_id,
                chunk_id=chunk.id,
                vector_dimension=len(embedding_list),
                embedding_time=embedding_time
            )
            
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to generate embedding for chunk {chunk.id}: {str(e)}"
            log_error(e, {
                "chunk_id": chunk.id,
                "content_length": len(chunk.content),
                "model_name": self.model_name
            })
            raise DataProcessingError(error_msg) from e
    
    def generate_embeddings_batch(
        self, 
        chunks: List[DocumentChunk], 
        batch_size: int = 32
    ) -> Tuple[List[DocumentEmbedding], Dict[str, Any]]:
        """Generate embeddings for multiple chunks in batches."""
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            start_time = time.time()
            embeddings = []
            failed_chunks = []
            
            logger.info(f"Starting batch embedding generation for {len(chunks)} chunks")
            
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = []
                valid_chunks = []
                
                # Preprocess texts
                for chunk in batch_chunks:
                    try:
                        processed_text = self._preprocess_text(chunk.content)
                        batch_texts.append(processed_text)
                        valid_chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to preprocess chunk {chunk.id}: {str(e)}")
                        failed_chunks.append({
                            'chunk_id': chunk.id,
                            'error': str(e)
                        })
                
                if not batch_texts:
                    continue
                
                # Generate embeddings for batch
                try:
                    batch_start = time.time()
                    
                    embedding_vectors = self.model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        batch_size=min(batch_size, 32)  # Limit to avoid memory issues
                    )
                    
                    batch_time = time.time() - batch_start
                    
                    # Create embedding objects
                    for chunk, text, vector in zip(valid_chunks, batch_texts, embedding_vectors):
                        embedding_id = self._generate_embedding_id(chunk.id, self.model_name)
                        
                        metadata = {
                            "source_section": chunk.metadata.section,
                            "source_url": chunk.metadata.url,
                            "chunk_index": chunk.chunk_index,
                            "word_count": chunk.word_count,
                            "character_count": chunk.character_count,
                            "batch_processing": True,
                            "batch_size": len(batch_texts),
                            "model_version": getattr(self.model, '_model_name', self.model_name)
                        }
                        
                        embedding = DocumentEmbedding(
                            id=embedding_id,
                            document_id=chunk.document_id,
                            chunk_id=chunk.id,
                            content=text,
                            embedding_vector=vector.tolist(),
                            model_name=self.model_name,
                            vector_dimension=self.vector_dimension,
                            metadata=metadata
                        )
                        
                        embeddings.append(embedding)
                    
                    self.total_embeddings_generated += len(valid_chunks)
                    
                    logger.debug(
                        f"Batch {i//batch_size + 1} completed",
                        batch_size=len(valid_chunks),
                        batch_time=batch_time,
                        embeddings_created=len(valid_chunks)
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {i//batch_size + 1}: {str(e)}")
                    for chunk in valid_chunks:
                        failed_chunks.append({
                            'chunk_id': chunk.id,
                            'error': f"Batch processing failed: {str(e)}"
                        })
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            stats = {
                'total_chunks_input': len(chunks),
                'embeddings_created': len(embeddings),
                'failed_chunks': len(failed_chunks),
                'success_rate': len(embeddings) / len(chunks) if chunks else 0,
                'total_processing_time': total_time,
                'average_time_per_chunk': total_time / len(chunks) if chunks else 0,
                'average_time_per_embedding': total_time / len(embeddings) if embeddings else 0,
                'failed_chunk_details': failed_chunks,
                'vector_dimension': self.vector_dimension,
                'model_name': self.model_name
            }
            
            logger.info(
                f"Batch embedding generation completed",
                **{k: v for k, v in stats.items() if k != 'failed_chunk_details'}
            )
            
            return embeddings, stats
            
        except Exception as e:
            error_msg = f"Batch embedding generation failed: {str(e)}"
            log_error(e, {
                "total_chunks": len(chunks),
                "batch_size": batch_size,
                "model_name": self.model_name
            })
            raise DataProcessingError(error_msg) from e
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for a search query."""
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            # Preprocess query
            processed_query = self._preprocess_text(query_text)
            
            logger.debug(f"Generating query embedding", query_length=len(processed_query))
            
            # Generate embedding
            start_time = time.time()
            query_vector = self.model.encode(
                [processed_query],
                show_progress_bar=False,
                normalize_embeddings=True
            )[0]
            
            embedding_time = time.time() - start_time
            
            logger.debug(
                f"Query embedding generated",
                vector_dimension=len(query_vector),
                embedding_time=embedding_time
            )
            
            return query_vector.tolist()
            
        except Exception as e:
            error_msg = f"Failed to generate query embedding: {str(e)}"
            log_error(e, {
                "query_text": query_text[:100],  # Limit logged query length
                "model_name": self.model_name
            })
            raise DataProcessingError(error_msg) from e
    
    def create_embedding_job(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32
    ) -> EmbeddingJob:
        """Create an embedding job for processing chunks."""
        
        job_id = f"emb_job_{uuid.uuid4().hex[:12]}"
        chunk_ids = [chunk.id for chunk in chunks]
        document_ids = list(set(chunk.document_id for chunk in chunks))
        
        job = EmbeddingJob(
            id=job_id,
            document_ids=document_ids,
            chunk_ids=chunk_ids,
            model_name=self.model_name,
            total_chunks=len(chunks),
            status=EmbeddingStatus.PENDING
        )
        
        logger.info(
            f"Created embedding job {job_id}",
            total_chunks=len(chunks),
            unique_documents=len(document_ids),
            model_name=self.model_name
        )
        
        return job
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        
        avg_embedding_time = (
            sum(self.embedding_times) / len(self.embedding_times)
            if self.embedding_times else 0.0
        )
        
        return {
            'model_name': self.model_name,
            'model_loaded': self.model_loaded,
            'vector_dimension': self.vector_dimension,
            'total_embeddings_generated': self.total_embeddings_generated,
            'average_embedding_time_ms': avg_embedding_time * 1000,
            'embedding_history_count': len(self.embedding_times),
            'service_uptime': time.time() - getattr(self, '_start_time', time.time()),
            'memory_usage_mb': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> float:
        """Get approximate memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the embedding service."""
        
        health_status = {
            'service_status': 'healthy',
            'model_loaded': self.model_loaded,
            'model_name': self.model_name,
            'vector_dimension': self.vector_dimension,
            'total_embeddings_generated': self.total_embeddings_generated,
            'last_check': datetime.utcnow().isoformat()
        }
        
        # Test embedding generation
        try:
            if self.model_loaded:
                test_start = time.time()
                test_embedding = self.generate_query_embedding("test health check")
                test_time = time.time() - test_start
                
                health_status.update({
                    'test_embedding_success': True,
                    'test_embedding_time_ms': test_time * 1000,
                    'test_vector_dimension': len(test_embedding)
                })
            else:
                health_status['test_embedding_success'] = False
                health_status['test_embedding_error'] = 'Model not loaded'
                
        except Exception as e:
            health_status.update({
                'service_status': 'degraded',
                'test_embedding_success': False,
                'test_embedding_error': str(e)
            })
        
        return health_status