#!/usr/bin/env python3
"""
Test script for retrieval system functionality.
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.config.logging import setup_logging
from app.services.retrieval_service import RetrievalService


def test_retrieval_system():
    """Test the complete retrieval system."""
    
    print("TESTING RETRIEVAL SYSTEM")
    print("=" * 60)
    
    setup_logging()
    
    try:
        # Initialize retrieval service
        print("\n1. INITIALIZING RETRIEVAL SERVICE")
        print("-" * 40)
        
        retrieval_service = RetrievalService()
        print("   RetrievalService initialized successfully")
        
        # Health check
        print("\n2. HEALTH CHECK")
        print("-" * 40)
        
        health = retrieval_service.health_check()
        print(f"   Service status: {health['service_status']}")
        print(f"   Embedding service: {health['embedding_service']['status']}")
        print(f"   Vector store: {health['vector_store']['status']}")
        
        if 'retrieval_test' in health:
            print(f"   Retrieval test: {health['retrieval_test']['status']}")
            if health['retrieval_test']['status'] == 'success':
                print(f"   Test results: {health['retrieval_test']['results_found']}")
                print(f"   Search time: {health['retrieval_test']['search_time_ms']:.1f} ms")
        
        # Test different types of queries
        print("\n3. TESTING DIFFERENT QUERY TYPES")
        print("-" * 40)
        
        test_queries = [
            "¿Cuáles son los principales ríos de Colombia?",
            "Háblame brevemente de la historia de Colombia en la época colonial",
            "¿Qué departamentos conforman Colombia?",
            "Descríbeme brevemente el escudo de armas de Colombia.",
            "¿Qué país comparte frontera con Colombia?",
            "Háblame sobre la biodiversidad en los parques nacionales de Colombia.",
            "Háblame sobre los platos típicos de Colombia.",
            "¿Cuál es la moneda oficial de Colombia?",
            "¿Quién es el compositor del himno de Colombia?",
            "¿Cuáles son los principales productos de exportación de Colombia?",
            "Describe brevemente el sistema político de Colombia.",
            "¿Cuáles son algunas de las montañas más altas de Colombia y dónde se ubican?",
            "¿Qué es el Carnaval de Barranquilla y en qué mes se celebra?",
            "¿Quién es Simón Bolivar?",
            "¿Quién es Gustavo Petro?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            
            try:
                response = retrieval_service.retrieve_documents(
                    query=query,
                    top_k=3,
                    similarity_threshold=0.1,
                    expand_query=True,
                    validate_colombia_relevance=True,
                    include_context=True
                )
                
                print(f"   Results found: {response.total_results}")
                print(f"   Search time: {response.search_time_ms:.1f} ms")
                
                # Show top result details
                if response.results:
                    top_result = response.results[0]
                    print(f"   Top result similarity: {top_result.similarity_score:.3f}")
                    print(f"   Content preview: {top_result.content[:80]}..." if top_result.content else "")
                    
                    if top_result.metadata:
                        section = top_result.metadata.get('source_section', 'Unknown')
                        rerank_score = top_result.metadata.get('rerank_score', 0)
                        print(f"   Section: {section}")
                        print(f"   Re-rank score: {rerank_score:.3f}")
                
            except Exception as e:
                print(f"   ERROR: {str(e)}")
        
        # Test query validation
        print("\n4. TESTING QUERY VALIDATION")
        print("-" * 40)
        
        invalid_queries = [
            "¿Cuál es la capital de Perú?",
            "Explícame cómo implementar una lista enlazada en C#.",
            "¿Quién ganó la Copa Mundial de la FIFA 2022?",
            "Dime la temperatura actual en París."
        ]
        
        for query in invalid_queries:
            print(f"\n   Testing invalid query: '{query}'")
            try:
                response = retrieval_service.retrieve_documents(
                    query=query,
                    top_k=2,
                    validate_colombia_relevance=True
                )
                print(f"   Unexpected success: {response.total_results} results")
            except Exception as e:
                print(f"   Correctly rejected: {type(e).__name__}")
        
        # Test query expansion
        print("\n5. TESTING QUERY EXPANSION")
        print("-" * 40)
        
        expansion_test_query = "ríos"
        print(f"   Testing expansion for: '{expansion_test_query}'")
        
        # With expansion
        response_with = retrieval_service.retrieve_documents(
            query=expansion_test_query,
            top_k=3,
            expand_query=True,
            validate_colombia_relevance=False
        )
        
        # Without expansion  
        response_without = retrieval_service.retrieve_documents(
            query=expansion_test_query,
            top_k=3,
            expand_query=False,
            validate_colombia_relevance=False
        )
        
        print(f"   With expansion: {response_with.total_results} results")
        print(f"   Without expansion: {response_without.total_results} results")
        print(f"   Search time with expansion: {response_with.search_time_ms:.1f} ms")
        
        # Service statistics
        print("\n6. SERVICE STATISTICS")
        print("-" * 40)
        
        stats = retrieval_service.get_retrieval_stats()
        print(f"   Embedding service:")
        print(f"     Model loaded: {stats['embedding_service']['model_loaded']}")
        print(f"     Total embeddings generated: {stats['embedding_service']['total_embeddings_generated']}")
        print(f"     Avg embedding time: {stats['embedding_service']['average_embedding_time_ms']:.1f} ms")
        
        print(f"   Vector store:")
        print(f"     Total embeddings: {stats['vector_store']['total_embeddings']}")
        print(f"     Unique documents: {stats['vector_store']['unique_documents']}")
        print(f"     Storage size: {stats['vector_store']['storage_size_mb']:.2f} MB")
        
        print(f"   Query processor:")
        print(f"     Colombia keywords: {stats['query_processor']['colombia_keywords_count']}")
        print(f"     Expansion terms: {stats['query_processor']['expansion_terms_count']}")
        
        print("\n" + "=" * 60)
        print("RETRIEVAL SYSTEM TEST COMPLETED SUCCESSFULLY")
        print("Ready to proceed with LLM Integration (PASO 4)")
        
    except Exception as e:
        print(f"\nERROR: Retrieval system test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_retrieval_system()