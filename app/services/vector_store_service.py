"""
Vector store service using ChromaDB for document embeddings storage and retrieval.
"""

import os
import time
import uuid
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

import chromadb

from app.config.settings import settings
from app.config.logging import logger, log_error
from app.core.exceptions import (
    VectorStoreError,
    DocumentNotFoundError,
    ConfigurationError,
    DataProcessingError
)
from app.models.embeddings import (
    DocumentEmbedding,
    VectorSearchQuery,
    VectorSearchResult,
    VectorSearchResponse,
    VectorStoreStats
)


class VectorStoreService:
    """
    Service for managing vector storage and retrieval using ChromaDB.
    """
    
    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """Initialize the vector store service."""
        
        self.collection_name = collection_name or settings.vector_db_collection_name
        self.persist_directory = persist_directory or settings.vector_db_path
        self.client = None
        self.collection = None
        self._is_initialized = False
        
        logger.info(
            f"Initializing VectorStoreService",
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client."""
        
        if self._is_initialized:
            return
        
        try:
            logger.info("Initializing ChromaDB client")
            
            # Create client with new API (persistent client)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
                
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "created_at": datetime.utcnow().isoformat(),
                        "description": "Colombia RAG document embeddings",
                        "model_name": settings.embedding_model,
                        "chunk_size": settings.chunk_size,
                        "chunk_overlap": settings.chunk_overlap
                    }
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            self._is_initialized = True
            
            logger.info(
                f"ChromaDB client initialized successfully",
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB client: {str(e)}"
            log_error(e, {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            })
            raise VectorStoreError(error_msg) from e
    
    def add_embeddings(
        self, 
        embeddings: List[DocumentEmbedding],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Add embeddings to the vector store."""
        
        try:
            self._initialize_client()
            
            start_time = time.time()
            added_count = 0
            failed_count = 0
            failed_embeddings = []
            
            logger.info(f"Adding {len(embeddings)} embeddings to vector store")
            
            # Process in batches
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                
                try:
                    # Prepare batch data
                    ids = []
                    vectors = []
                    metadatas = []
                    documents = []
                    
                    for embedding in batch_embeddings:
                        ids.append(embedding.id)
                        vectors.append(embedding.embedding_vector)
                        
                        # Prepare metadata
                        metadata = {
                            "document_id": embedding.document_id,
                            "chunk_id": embedding.chunk_id,
                            "model_name": embedding.model_name,
                            "vector_dimension": embedding.vector_dimension,
                            "created_at": embedding.created_at.isoformat(),
                            "content_length": len(embedding.content),
                            **embedding.metadata
                        }
                        metadatas.append(metadata)
                        documents.append(embedding.content)
                    
                    # Add to collection
                    self.collection.add(
                        ids=ids,
                        embeddings=vectors,
                        metadatas=metadatas,
                        documents=documents
                    )
                    
                    added_count += len(batch_embeddings)
                    
                    logger.debug(
                        f"Batch {i//batch_size + 1} added successfully",
                        batch_size=len(batch_embeddings),
                        total_added=added_count
                    )
                    
                except Exception as e:
                    failed_count += len(batch_embeddings)
                    failed_embeddings.extend([emb.id for emb in batch_embeddings])
                    logger.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
            

            
            total_time = time.time() - start_time
            
            result = {
                "embeddings_added": added_count,
                "embeddings_failed": failed_count,
                "failed_embedding_ids": failed_embeddings,
                "success_rate": added_count / len(embeddings) if embeddings else 0,
                "processing_time": total_time,
                "collection_name": self.collection_name
            }
            
            logger.info(
                f"Embeddings added to vector store",
                **{k: v for k, v in result.items() if k != "failed_embedding_ids"}
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to add embeddings to vector store: {str(e)}"
            log_error(e, {
                "embeddings_count": len(embeddings),
                "collection_name": self.collection_name
            })
            raise VectorStoreError(error_msg) from e
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filters: Dict[str, Any] = None,
        include_content: bool = True,
        include_metadata: bool = True
    ) -> List[VectorSearchResult]:
        """Search for similar embeddings in the vector store."""
        
        try:
            self._initialize_client()
            
            start_time = time.time()
            
            logger.debug(
                f"Searching for similar embeddings",
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters
            )
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k
            }
            
            # Add filters if provided
            if filters:
                # Convert filters to ChromaDB format
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = {"$eq": value}
                
                if where_clause:
                    query_params["where"] = where_clause
            
            # Include content and metadata
            include_list = []
            if include_content:
                include_list.append("documents")
            if include_metadata:
                include_list.append("metadatas")
            include_list.extend(["distances"])
            
            query_params["include"] = include_list
            
            # Perform search
            results = self.collection.query(**query_params)
            
            search_time = time.time() - start_time
            
            # Process results
            search_results = []
            
            if results and results.get("ids") and len(results["ids"]) > 0:
                ids = results["ids"][0]
                distances = results["distances"][0]
                metadatas = results.get("metadatas", [[]])[0]
                documents = results.get("documents", [[]])[0]
                
                for i, (result_id, distance) in enumerate(zip(ids, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1.0 - distance
                    
                    # Filter by similarity threshold
                    if similarity_score < similarity_threshold:
                        continue
                    
                    # Get metadata
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    content = documents[i] if i < len(documents) else None
                    
                    result = VectorSearchResult(
                        id=result_id,
                        document_id=metadata.get("document_id", ""),
                        chunk_id=metadata.get("chunk_id", ""),
                        similarity_score=similarity_score,
                        content=content if include_content else None,
                        metadata=metadata if include_metadata else None
                    )
                    
                    search_results.append(result)
            
            logger.debug(
                f"Vector search completed",
                results_found=len(search_results),
                search_time=search_time,
                top_k=top_k
            )
            
            return search_results
            
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            log_error(e, {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "collection_name": self.collection_name
            })
            raise VectorStoreError(error_msg) from e
    
    def search_by_query(self, search_query: VectorSearchQuery) -> VectorSearchResponse:
        """Search using a structured query object."""
        
        try:
            start_time = time.time()
            
            # If query embedding is not provided, we need to generate it
            # This would typically be done by calling the embedding service
            if not search_query.query_embedding:
                raise ValueError("Query embedding must be provided")
            
            # Perform search
            results = self.search_similar(
                query_embedding=search_query.query_embedding,
                top_k=search_query.top_k,
                similarity_threshold=search_query.similarity_threshold,
                filters=search_query.filters,
                include_content=search_query.include_content,
                include_metadata=search_query.include_metadata
            )
            
            search_time_ms = (time.time() - start_time) * 1000
            
            response = VectorSearchResponse(
                query=search_query.query_text,
                results=results,
                total_results=len(results),
                search_time_ms=search_time_ms,
                collection_name=self.collection_name,
                model_name=settings.embedding_model
            )
            
            logger.info(
                f"Query search completed",
                query=search_query.query_text[:50],
                results_count=len(results),
                search_time_ms=search_time_ms
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Query search failed: {str(e)}"
            log_error(e, {
                "query": search_query.query_text,
                "collection_name": self.collection_name
            })
            raise VectorStoreError(error_msg) from e
    
    def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific embedding by ID."""
        
        try:
            self._initialize_client()
            
            results = self.collection.get(
                ids=[embedding_id],
                include=["metadatas", "documents", "embeddings"]
            )
            
            if results and results.get("ids") and len(results["ids"]) > 0:
                metadatas_data = results.get("metadatas")
                documents_data = results.get("documents") 
                embeddings_data = results.get("embeddings")
                
                metadata = metadatas_data[0] if metadatas_data is not None and len(metadatas_data) > 0 else {}
                content = documents_data[0] if documents_data is not None and len(documents_data) > 0 else ""
                embedding = embeddings_data[0] if embeddings_data is not None and len(embeddings_data) > 0 else []
                
                return {
                    "id": embedding_id,
                    "content": content,
                    "metadata": metadata,
                    "embedding": embedding
                }
            
            return None
            
        except Exception as e:
            error_msg = f"Failed to get embedding by ID {embedding_id}: {str(e)}"
            log_error(e, {"embedding_id": embedding_id})
            raise VectorStoreError(error_msg) from e
    
    def delete_embeddings(self, embedding_ids: List[str]) -> Dict[str, Any]:
        """Delete embeddings by IDs."""
        
        try:
            self._initialize_client()
            
            logger.info(f"Deleting {len(embedding_ids)} embeddings")
            
            self.collection.delete(ids=embedding_ids)

            
            result = {
                "deleted_count": len(embedding_ids),
                "collection_name": self.collection_name
            }
            
            logger.info(f"Embeddings deleted successfully", **result)
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to delete embeddings: {str(e)}"
            log_error(e, {"embedding_ids": embedding_ids})
            raise VectorStoreError(error_msg) from e
    
    def get_collection_stats(self) -> VectorStoreStats:
        """Get statistics about the vector store collection."""
        
        try:
            self._initialize_client()
            
            # Get collection info
            collection_count = self.collection.count()
            
            # Get sample to determine vector dimension
            sample_results = self.collection.get(
                limit=1,
                include=["metadatas", "embeddings"]
            )
            
            vector_dimension = 0
            model_name = settings.embedding_model
            unique_documents = 0
            
            embeddings_data = sample_results.get("embeddings")
            if sample_results and embeddings_data is not None and len(embeddings_data) > 0:
                vector_dimension = len(embeddings_data[0])
                
                if sample_results.get("metadatas"):
                    sample_metadata = sample_results["metadatas"][0]
                    model_name = sample_metadata.get("model_name", model_name)
            
            # Get unique document count (approximate)
            if collection_count > 0:
                all_metadatas = self.collection.get(include=["metadatas"])
                if all_metadatas and all_metadatas.get("metadatas"):
                    document_ids = set()
                    for metadata in all_metadatas["metadatas"]:
                        if metadata.get("document_id"):
                            document_ids.add(metadata["document_id"])
                    unique_documents = len(document_ids)
            
            # Estimate storage size (rough approximation)
            storage_size_mb = collection_count * vector_dimension * 4 / (1024 * 1024)  # 4 bytes per float
            index_size_mb = storage_size_mb * 0.1  # Rough estimate
            
            stats = VectorStoreStats(
                collection_name=self.collection_name,
                total_embeddings=collection_count,
                unique_documents=unique_documents,
                total_chunks=collection_count,
                vector_dimension=vector_dimension,
                model_name=model_name,
                last_updated=datetime.utcnow(),
                storage_size_mb=storage_size_mb,
                index_size_mb=index_size_mb
            )
            
            logger.debug(f"Collection stats retrieved", total_embeddings=collection_count)
            
            return stats
            
        except Exception as e:
            error_msg = f"Failed to get collection stats: {str(e)}"
            log_error(e, {"collection_name": self.collection_name})
            raise VectorStoreError(error_msg) from e
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the vector store."""
        
        health_status = {
            "service_status": "healthy",
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "client_initialized": self._is_initialized,
            "last_check": datetime.utcnow().isoformat()
        }
        
        try:
            # Test connection
            self._initialize_client()
            
            # Test basic operations
            collection_count = self.collection.count()
            health_status.update({
                "connection_test": "success",
                "total_embeddings": collection_count,
                "collection_accessible": True
            })
            
            # Test search if we have data
            if collection_count > 0:
                try:
                    # Get a sample embedding for testing
                    sample = self.collection.get(limit=1, include=["embeddings"])
                    embeddings_data = sample.get("embeddings")
                    if sample and embeddings_data is not None and len(embeddings_data) > 0:
                        test_embedding = embeddings_data[0]
                        test_start = time.time()
                        
                        test_results = self.search_similar(
                            query_embedding=test_embedding,
                            top_k=1,
                            similarity_threshold=0.0
                        )
                        
                        test_time = time.time() - test_start
                        
                        health_status.update({
                            "search_test": "success",
                            "search_time_ms": test_time * 1000,
                            "search_results_count": len(test_results)
                        })
                    else:
                        health_status["search_test"] = "skipped - no embeddings available"
                except Exception as search_error:
                    health_status["search_test"] = f"failed - {str(search_error)}"
            else:
                health_status["search_test"] = "skipped - collection empty"
                
        except Exception as e:
            health_status.update({
                "service_status": "degraded",
                "connection_test": "failed",
                "error": str(e)
            })
        
        return health_status
    
    def reset_collection(self) -> Dict[str, Any]:
        """Reset (delete and recreate) the collection. USE WITH CAUTION!"""
        
        try:
            self._initialize_client()
            
            logger.warning(f"Resetting collection: {self.collection_name}")
            
            # Delete existing collection
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                logger.info(f"Collection {self.collection_name} did not exist")
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "created_at": datetime.utcnow().isoformat(),
                    "description": "Colombia RAG document embeddings",
                    "model_name": settings.embedding_model,
                    "reset_at": datetime.utcnow().isoformat()
                }
            )
            

            
            result = {
                "collection_reset": True,
                "collection_name": self.collection_name,
                "reset_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Collection reset successfully", **result)
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to reset collection: {str(e)}"
            log_error(e, {"collection_name": self.collection_name})
            raise VectorStoreError(error_msg) from e