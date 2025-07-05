#!/usr/bin/env python3
"""
Smoke test para verificar que el pipeline RAG completo funciona end-to-end.
Prueba: pregunta -> retrieval -> generación -> respuesta
"""

import sys
import os
from pathlib import Path

# Añadir el directorio raíz al path para importar app
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.services import ResponseGenerationService
    from app.config.settings import get_settings
    print("PASS: Imports exitosos")
except Exception as e:
    print(f"FAIL: Error en imports: {e}")
    sys.exit(1)


def test_rag_pipeline():  # Quitar async
    """Test completo del pipeline RAG"""
    
    print("\nIniciando smoke test del pipeline RAG...")
    
    try:
        # Inicializar el servicio
        print("\n1. Inicializando ResponseGenerationService...")
        rgs = ResponseGenerationService()
        print("PASS: ResponseGenerationService inicializado")
        
        # Health check
        print("\n2. Ejecutando health check...")
        health_result = rgs.health_check()  # Quitar await
        print(f"PASS: Health check: {health_result}")
        
        # Prueba con pregunta sobre Colombia
        test_queries = [
            "¿Cuál es la capital de Colombia?",
            "¿Qué océanos rodean a Colombia?", 
            "¿Cuáles son las principales cordilleras de Colombia?"
        ]
        
        print("\n3. Probando generación de respuestas...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Pregunta {i}: {query} ---")
            
            try:
                response = rgs.generate_response(query)  # Quitar await
                
                print(f"Respuesta: {response['answer'][:200]}...")
                print(f"Fuentes encontradas: {len(response['sources'])}")
                
                if response['sources']:
                    print(f"Primera fuente: {response['sources'][0][:100]}...")
                
            except Exception as e:
                print(f"FAIL: Error en pregunta {i}: {e}")
        
        print("\nSmoke test completado exitosamente")
        return True
        
    except Exception as e:
        print(f"FAIL: Error en smoke test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ollama_connection():
    """Verificar que Ollama está corriendo"""
    try:
        from langchain_community.llms import Ollama
        settings = get_settings()
        
        print("\nVerificando conexión con Ollama...")
        llm = Ollama(model=settings.llm_model)
        
        # Test simple
        response = llm.invoke("Responde solo: 'OK'")
        print(f"PASS: Ollama responde: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"FAIL: Error conectando con Ollama: {e}")
        print("HINT: Asegúrate de que Ollama esté corriendo: ollama serve")
        return False


if __name__ == "__main__":
    print("Colombia RAG Chatbot - Smoke Test")
    print("="*50)
    
    # Verificar Ollama primero
    if not test_ollama_connection():
        print("\nFAIL: Ollama no disponible. Abortando test.")
        sys.exit(1)
    
    # Ejecutar test RAG
    success = test_rag_pipeline()  # Quitar asyncio.run()
    
    if success:
        print("\nPASS: Todos los tests pasaron")
        print("INFO: El pipeline RAG está listo para producción")
        sys.exit(0)
    else:
        print("\nFAIL: Algunos tests fallaron")
        sys.exit(1)