"""
Test script for vector store functionality.
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.config.logging import setup_logging
from app.services.data_extractor import WikipediaExtractor
from app.services.text_processor import TextProcessor
from app.services.chunking_service import IntelligentChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


def test_vector_store_complete():
    """Test the complete vector store workflow."""
    
    print("TESTING VECTOR STORE - COMPLETE WORKFLOW")
    print("=" * 60)
    
    setup_logging()
    
    try:
        # Step 1: Get embeddings (using existing pipeline)
        print("\n1. PREPARING EMBEDDINGS")
        print("-" * 40)
        
        # Extract and process data
        extractor = WikipediaExtractor()
        raw_documents = extractor.extract_colombia_content(use_cache=True, sections=None)
        
        processor = TextProcessor()
        processed_documents, _ = processor.batch_process_documents(raw_documents)
        
        chunker = IntelligentChunker()
        chunks, _ = chunker.batch_chunk_documents(processed_documents, chunk_size=500, chunk_overlap=100)
        
        print(f"   Chunks available: {len(chunks)}")
        
        # Generate embeddings for first 10 chunks (for testing)
        embedding_service = EmbeddingService()
        test_chunks = chunks[:10]
        embeddings, stats = embedding_service.generate_embeddings_batch(test_chunks)
        
        print(f"   Embeddings generated: {len(embeddings)}")
        print(f"   Vector dimension: {embeddings[0].vector_dimension if embeddings else 'N/A'}")
        
        # Step 2: Initialize Vector Store
        print("\n2. INITIALIZING VECTOR STORE")
        print("-" * 40)
        
        vector_store = VectorStoreService()
        
        # Health check
        health = vector_store.health_check()
        print(f"   Service status: {health['service_status']}")
        print(f"   Collection: {health['collection_name']}")
        print(f"   Client initialized: {health['client_initialized']}")
        
        # Step 3: Add embeddings to vector store
        print("\n3. ADDING EMBEDDINGS TO VECTOR STORE")
        print("-" * 40)
        
        add_result = vector_store.add_embeddings(embeddings, batch_size=5)
        print(f"   Embeddings added: {add_result['embeddings_added']}")
        print(f"   Success rate: {add_result['success_rate']:.1%}")
        print(f"   Processing time: {add_result['processing_time']:.2f} seconds")
        
        # Step 4: Get collection statistics
        print("\n4. COLLECTION STATISTICS")
        print("-" * 40)
        
        stats = vector_store.get_collection_stats()
        print(f"   Total embeddings: {stats.total_embeddings}")
        print(f"   Unique documents: {stats.unique_documents}")
        print(f"   Vector dimension: {stats.vector_dimension}")
        print(f"   Model name: {stats.model_name}")
        print(f"   Storage size: {stats.storage_size_mb:.2f} MB")
        
        # Step 5: Test search functionality
        print("\n5. TESTING SEARCH FUNCTIONALITY")
        print("-" * 40)
        
        # Generate query embedding
        test_queries = [
            "¿Cuáles son los principales ríos de Colombia?",
            "Háblame sobre los platos típicos de Colombia.",
            "¿Cuál es la moneda oficial de Colombia?"
        ]
        
        for query_text in test_queries:
            print(f"\n   Query: '{query_text}'")
            
            # Generate query embedding
            query_embedding = embedding_service.generate_query_embedding(query_text)
            
            # Search similar embeddings
            search_results = vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=3,
                similarity_threshold=0.1,
                include_content=True,
                include_metadata=True
            )
            
            print(f"   Results found: {len(search_results)}")
            
            for i, result in enumerate(search_results):
                print(f"     {i+1}. Similarity: {result.similarity_score:.3f}")
                print(f"        Content: {result.content[:80]}..." if result.content else "")
                print(f"        Chunk ID: {result.chunk_id}")
        
        # Step 6: Test specific embedding retrieval
        print("\n6. TESTING EMBEDDING RETRIEVAL")
        print("-" * 40)
        
        if embeddings:
            test_embedding_id = embeddings[0].id
            retrieved = vector_store.get_embedding_by_id(test_embedding_id)
            
            if retrieved:
                print(f"   Retrieved embedding: {test_embedding_id}")
                print(f"   Content length: {len(retrieved['content'])}")
                print(f"   Metadata keys: {list(retrieved['metadata'].keys())}")
                print(f"   Vector length: {len(retrieved['embedding'])}")
            else:
                print(f"   Embedding not found: {test_embedding_id}")
        
        # Step 7: Final health check
        print("\n7. FINAL HEALTH CHECK")
        print("-" * 40)
        
        final_health = vector_store.health_check()
        print(f"   Service status: {final_health['service_status']}")
        print(f"   Total embeddings: {final_health.get('total_embeddings', 'N/A')}")
        print(f"   Search test: {final_health.get('search_test', 'N/A')}")
        if 'search_time_ms' in final_health:
            print(f"   Search time: {final_health['search_time_ms']:.1f} ms")
        
        print("\n" + "=" * 60)
        print("VECTOR STORE TEST COMPLETED SUCCESSFULLY")
        print("Vector store is ready for RAG implementation")
        
    except Exception as e:
        print(f"\nERROR: Vector store test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vector_store_complete()