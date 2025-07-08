#!/usr/bin/env python3
"""
Test script for embedding generation.
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


def test_embedding_generation():
    """Test the complete pipeline including embedding generation."""
    
    print("TESTING EMBEDDING GENERATION")
    print("=" * 60)
    
    setup_logging()
    
    try:
        # Step 1: Get processed chunks (using existing pipeline)
        print("\n1. EXTRACTING AND PROCESSING DATA")
        print("-" * 40)
        
        extractor = WikipediaExtractor()
        raw_documents = extractor.extract_colombia_content(use_cache=True, sections=None)
        print(f"   Documents extracted: {len(raw_documents)}")
        
        processor = TextProcessor()
        processed_documents, _ = processor.batch_process_documents(raw_documents)
        print(f"   Documents processed: {len(processed_documents)}")
        
        chunker = IntelligentChunker()
        chunks, _ = chunker.batch_chunk_documents(processed_documents, chunk_size=500, chunk_overlap=100)
        print(f"   Chunks created: {len(chunks)}")
        
        if not chunks:
            print("ERROR: No chunks available for embedding generation")
            return
        
        # Step 2: Test single embedding generation
        print("\n2. TESTING SINGLE EMBEDDING GENERATION")
        print("-" * 40)
        
        embedding_service = EmbeddingService()
        
        # Test with first chunk
        test_chunk = chunks[0]
        print(f"   Test chunk ID: {test_chunk.id}")
        print(f"   Content length: {len(test_chunk.content)} chars")
        print(f"   Content preview: {test_chunk.content[:100]}...")
        
        # Generate single embedding
        embedding = embedding_service.generate_embedding(test_chunk)
        print(f"   Embedding ID: {embedding.id}")
        print(f"   Vector dimension: {embedding.vector_dimension}")
        print(f"   Model used: {embedding.model_name}")
        print(f"   Embedding preview: {embedding.embedding_vector[:5]}...")
        
        # Step 3: Test batch embedding generation
        print("\n3. TESTING BATCH EMBEDDING GENERATION")
        print("-" * 40)
        
        # Use first 5 chunks for testing
        test_chunks = chunks[:5]
        print(f"   Processing {len(test_chunks)} chunks in batch")
        
        embeddings, stats = embedding_service.generate_embeddings_batch(test_chunks, batch_size=3)
        
        print(f"   Embeddings created: {stats['embeddings_created']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Total time: {stats['total_processing_time']:.2f} seconds")
        print(f"   Average time per chunk: {stats['average_time_per_chunk']:.3f} seconds")
        
        if embeddings:
            print(f"   Sample embedding:")
            sample = embeddings[0]
            print(f"     ID: {sample.id}")
            print(f"     Source chunk: {sample.chunk_id}")
            print(f"     Vector dimension: {sample.vector_dimension}")
            print(f"     Content length: {len(sample.content)} chars")
        
        # Step 4: Test query embedding
        print("\n4. TESTING QUERY EMBEDDING GENERATION")
        print("-" * 40)
        
        test_queries = [
            "¿Cuál es la capital de Colombia?",
            "Historia de Colombia",
            "Geografía colombiana"
        ]
        
        for query in test_queries:
            query_embedding = embedding_service.generate_query_embedding(query)
            print(f"   Query: '{query}'")
            print(f"   Embedding dimension: {len(query_embedding)}")
            print(f"   Embedding preview: {query_embedding[:3]}...")
        
        # Step 5: Service statistics
        print("\n5. SERVICE STATISTICS")
        print("-" * 40)
        
        stats = embedding_service.get_service_stats()
        print(f"   Model loaded: {stats['model_loaded']}")
        print(f"   Vector dimension: {stats['vector_dimension']}")
        print(f"   Total embeddings generated: {stats['total_embeddings_generated']}")
        print(f"   Average embedding time: {stats['average_embedding_time_ms']:.1f} ms")
        
        # Step 6: Health check
        print("\n6. HEALTH CHECK")
        print("-" * 40)
        
        health = embedding_service.health_check()
        print(f"   Service status: {health['service_status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   Test embedding success: {health['test_embedding_success']}")
        if 'test_embedding_time_ms' in health:
            print(f"   Test embedding time: {health['test_embedding_time_ms']:.1f} ms")
        
        print("\n" + "=" * 60)
        print("EMBEDDING GENERATION TEST COMPLETED SUCCESSFULLY")
        print("Ready to proceed with Vector Store implementation")
        
    except Exception as e:
        print(f"\nERROR: Embedding generation test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_embedding_generation()