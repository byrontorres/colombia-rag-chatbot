"""
Unit tests para ResponseGenerationService.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.response_generation_service import ResponseGenerationService
from app.core.exceptions import GenerationError, ModelError


class TestResponseGenerationService:
    """Test suite para ResponseGenerationService."""

    @pytest.fixture
    def mock_settings(self):
        """Mock de configuración de la aplicación."""
        with patch('app.config.settings.get_settings') as mock:
            mock.return_value = Mock(
                llm_model="llama3.1",
                llm_temperature=0.0,
                max_context_length=4000,
                top_k_documents=5
            )
            yield mock.return_value

    @pytest.fixture
    def mock_retrieval_service(self):
        """Mock del servicio de recuperación de documentos."""
        with patch('app.services.response_generation_service.RetrievalService') as mock:
            mock_instance = Mock()
            mock_instance.retrieve_documents.return_value = [
                {
                    'content': 'Colombia es un país ubicado en América del Sur.',
                    'metadata': {'source': 'https://es.wikipedia.org/wiki/Colombia'}
                },
                {
                    'content': 'La capital de Colombia es Bogotá.',
                    'metadata': {'source': 'https://es.wikipedia.org/wiki/Colombia'}
                }
            ]
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_ollama(self):
        """Mock del modelo de lenguaje Ollama."""
        with patch('app.services.response_generation_service.Ollama') as mock:
            mock_instance = Mock()
            mock_instance.invoke.return_value = (
                "Colombia es un país ubicado en el noroeste de América del Sur. "
                "Su capital es Bogotá, que también es la ciudad más poblada del país."
            )
            mock.return_value = mock_instance
            yield mock_instance

    def test_init_service(self, mock_settings, mock_retrieval_service, mock_ollama):
        """Test de inicialización correcta del servicio."""
        service = ResponseGenerationService()
        
        assert service.settings is not None
        assert service.retrieval_service is not None
        assert service.llm is not None
        assert service.chain is not None

    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_settings, mock_retrieval_service, mock_ollama):
        """Test de generación exitosa de respuesta."""
        service = ResponseGenerationService()
        
        # Mock del método async apredict de la chain
        with patch.object(service.chain, 'apredict', new_callable=AsyncMock) as mock_apredict:
            mock_apredict.return_value = (
                "Colombia es un país ubicado en América del Sur. "
                "Su capital es Bogotá."
            )
            
            result = await service.generate_response("¿Cuál es la capital de Colombia?")
            
            assert "answer" in result
            assert "sources" in result
            assert "Colombia" in result["answer"]
            assert "Bogotá" in result["answer"]
            assert len(result["sources"]) == 2
            assert "https://es.wikipedia.org/wiki/Colombia" in result["sources"]

    @pytest.mark.asyncio
    async def test_generate_response_no_documents(self, mock_settings, mock_ollama):
        """Test cuando no se encuentran documentos relevantes."""
        
        # Mock RetrievalService que no retorna documentos
        with patch('app.services.response_generation_service.RetrievalService') as mock_retrieval:
            mock_instance = Mock()
            mock_instance.retrieve_documents.return_value = []
            mock_retrieval.return_value = mock_instance
            
            service = ResponseGenerationService()
            
            with pytest.raises(GenerationError, match="No se encontró información"):
                await service.generate_response("¿Qué es Francia?")

    @pytest.mark.asyncio
    async def test_generate_response_llm_error(self, mock_settings, mock_retrieval_service):
        """Test cuando el modelo de lenguaje falla."""
        
        # Mock Ollama que lanza excepción
        with patch('app.services.response_generation_service.Ollama') as mock_ollama:
            mock_instance = Mock()
            mock_instance.invoke.side_effect = Exception("Connection error")
            mock_ollama.return_value = mock_instance
            
            service = ResponseGenerationService()
            
            # Mock del método async apredict que falla
            with patch.object(service.chain, 'apredict', new_callable=AsyncMock) as mock_apredict:
                mock_apredict.side_effect = Exception("LLM connection failed")
                
                with pytest.raises(ModelError, match="Error del modelo"):
                    await service.generate_response("¿Cuál es la capital de Colombia?")

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_settings, mock_retrieval_service, mock_ollama):
        """Test de health check exitoso."""
        service = ResponseGenerationService()
        
        with patch.object(service.chain, 'apredict', new_callable=AsyncMock) as mock_apredict:
            mock_apredict.return_value = "Colombia es un país en América del Sur."
            
            result = await service.health_check()
            
            assert result["status"] == "healthy"
            assert result["model"] == "llama3.1"
            assert "response_time" in result
            assert result["response_time"] >= 0

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_settings, mock_retrieval_service):
        """Test de health check que falla."""
        
        # Mock Ollama que falla
        with patch('app.services.response_generation_service.Ollama') as mock_ollama:
            mock_instance = Mock()
            mock_ollama.return_value = mock_instance
            
            service = ResponseGenerationService()
            
            with patch.object(service.chain, 'apredict', new_callable=AsyncMock) as mock_apredict:
                mock_apredict.side_effect = Exception("Health check failed")
                
                result = await service.health_check()
                
                assert result["status"] == "unhealthy"
                assert "error" in result

    def test_format_sources(self, mock_settings, mock_retrieval_service, mock_ollama):
        """Test de formateo de fuentes."""
        service = ResponseGenerationService()
        
        documents = [
            {'metadata': {'source': 'https://es.wikipedia.org/wiki/Colombia'}},
            {'metadata': {'source': 'https://es.wikipedia.org/wiki/Colombia'}},  # Duplicado
            {'metadata': {'source': 'https://es.wikipedia.org/wiki/Bogotá'}},
        ]
        
        sources = service._format_sources(documents)
        
        # Debe eliminar duplicados
        assert len(sources) == 2
        assert 'https://es.wikipedia.org/wiki/Colombia' in sources
        assert 'https://es.wikipedia.org/wiki/Bogotá' in sources

    def test_validate_query(self, mock_settings, mock_retrieval_service, mock_ollama):
        """Test de validación de consultas."""
        service = ResponseGenerationService()
        
        # Query válido
        assert service._validate_query("¿Cuál es la capital de Colombia?") is True
        
        # Query vacío
        with pytest.raises(GenerationError, match="Query no puede estar vacío"):
            service._validate_query("")
        
        # Query muy largo
        long_query = "a" * 1001
        with pytest.raises(GenerationError, match="Query demasiado largo"):
            service._validate_query(long_query)


# Configuración de pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])