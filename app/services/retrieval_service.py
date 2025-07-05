"""
Retrieval service for intelligent document search and context management.
"""
import unicodedata
import time
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from app.config.settings import settings
from app.config.logging import logger, log_error
from app.core.exceptions import (
    RetrievalError,
    InvalidQueryError,
    QueryNotColombiaRelatedError,
    ConfigurationError
)
from app.models.embeddings import (
    VectorSearchQuery,
    VectorSearchResult,
    VectorSearchResponse
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


class QueryProcessor:
    """Helper class for query preprocessing and validation."""
    
    def __init__(self):
        """Initialize query processor."""


        
        # Colombia-related keywords for relevance validation
        self.colombia_keywords = {
            'places': {
                'colombia', 'bogotá', 'medellín', 'cali', 'barranquilla', 'cartagena',
                'bucaramanga', 'pereira', 'manizales', 'santa marta', 'villavicencio',
                'pasto', 'montería', 'neiva', 'soledad', 'ibagué', 'cúcuta', 'popayán',
                'antioquia', 'cundinamarca', 'valle del cauca', 'santander', 'boyacá',
                'tolima', 'magdalena', 'meta', 'la guajira', 'nariño', 'chocó',
                'amazonas', 'orinoquía', 'andes', 'sierra nevada de santa marta',
                'san andrés', 'providencia', 'caldas', 'risaralda', 'quindío',
                'huila', 'cauca', 'córdoba', 'sucre', 'cesar', 'atlántico',
                'norte de santander', 'casanare', 'arauca', 'vichada', 'guainía',
                'vaupés', 'guaviare', 'putumayo', 'caquetá', 'río magdalena',
                'río cauca', 'río orinoco', 'río amazonas', 'cordillera oriental',
                'cordillera central', 'cordillera occidental', 'llanos orientales',
                'costa caribe', 'costa pacífica', 'eje cafetero', 'triángulo de oro'
            },
            'topics': {
                'colombiano', 'colombiana', 'historia', 'geografía', 'cultura',
                'bolívar', 'simón bolívar', 'libertador', 'economía', 'política', 
                'gobierno', 'presidente', 'constitución', 'independencia', 'república', 
                'territorio', 'población', 'moneda', 'peso', 'cop', 'bandera', 'himno', 
                'gastronomía', 'comida', 'turismo', 'deportes', 'fútbol', 'cafetero', 
                'café', 'flora', 'fauna', 'clima', 'pib', 'exportaciones', 'festivales', 
                'ferias', 'costumbres', 'peso', 'pesos', 'divisa',
                
                # Personajes históricos y políticos
                'jorge eliécer gaitán', 'francisco de paula santander', 'antonio nariño',
                'camilo torres', 'policarpa salavarrieta', 'rafael núñez', 'rafael uribe uribe',
                'álvaro uribe', 'juan manuel santos', 'iván duque', 'gustavo petro',
                'andrés pastrana', 'césar gaviria', 'ernesto samper', 'virgilio barco',
                
                # Cultura y arte
                'gabriel garcía márquez', 'fernando botero', 'shakira', 'juanes',
                'carlos vives', 'manu chao', 'joe arroyo', 'vallenato', 'cumbia',
                'salsa', 'champeta', 'bambuco', 'joropo', 'carranga', 'reggaeton',
                'marimba', 'gaita', 'tambor', 'acordeón',
                
                # Gastronomía
                'arepas', 'bandeja paisa', 'sancocho', 'ajiaco', 'tamales', 'empanadas',
                'patacón', 'casuela', 'mondongo', 'lechona', 'arepa de huevo',
                'buñuelos', 'natilla', 'aguardiente', 'chicha', 'guarapo', 'tinto',
                'mazamorra', 'changua', 'caldo de pajarilla', 'fritanga',
                
                # Deportes y deportistas
                'james rodríguez', 'radamel falcao', 'carlos valderrama', 'rené higuita',
                'nairo quintana', 'egan bernal', 'rigoberto urán', 'mariana pajón',
                'caterine ibargüen', 'yerry mina', 'juan cuadrado', 'selección colombia',
                'millonarios', 'américa de cali', 'atlético nacional', 'independiente santa fe',
                'ciclismo', 'bmx', 'atletismo', 'boxeo', 'natación',
                
                # Economía y productos
                'petróleo', 'carbón', 'oro', 'esmeraldas', 'platino', 'níquel',
                'flores', 'banano', 'cacao', 'azúcar', 'arroz', 'maíz', 'papa',
                'aguacate', 'mango', 'piña', 'coca', 'palma africana', 'textiles',
                'ecopetrol', 'avianca', 'grupo aval', 'bancolombia', 'isagen',
                
                # Política y social
                'farc', 'eln', 'auc', 'paramilitares', 'guerrilla', 'violencia',
                'narcotráfico', 'pablo escobar', 'cartel de medellín', 'cartel de cali',
                'proceso de paz', 'acuerdos de paz', 'justicia transicional',
                'desplazamiento', 'víctimas', 'reintegración', 'ddr',
                
                # Geografía específica
                'pico cristóbal colón', 'nevado del ruiz', 'volcán galeras',
                'desierto de la tatacoa', 'caño cristales', 'ciudad perdida',
                'parque tayrona', 'islas del rosario', 'laguna de guatavita',
                'cocuy', 'chingaza', 'farallones', 'utría', 'gorgona',
                
                # Educación
                'universidad nacional', 'universidad de los andes', 'javeriana',
                'universidad del valle', 'universidad de antioquia', 'icesi',
                'eafit', 'universidad del norte', 'externado', 'rosario',
                
                # Flora y fauna específica
                'orquídeas', 'ceiba', 'palma de cera', 'frailejón', 'yarumo',
                'cóndor', 'jaguar', 'oso de anteojos', 'delfín rosado', 'manatí',
                'mono tití', 'perezoso', 'iguana', 'colibrí', 'quetzal',
                'anaconda', 'caimán', 'tortuga', 'ballena jorobada',
                
                # Festivales y tradiciones
                'carnaval de barranquilla', 'feria de las flores', 'festival vallenato',
                'rock al parque', 'festival de teatro', 'hay festival',
                'semana santa', 'navidad', 'año nuevo', 'día de las velitas',
                'corpus christi', 'fiesta de reyes', 'fiestas patrias',
                
                # Términos adicionales
                'biodiversidad', 'megadiverso', 'páramo', 'humedal', 'manglar',
                'selva tropical', 'bosque seco', 'sabana', 'altiplano',
                'constitución de 1991', 'estado social de derecho', 'corte constitucional',
                'congreso', 'senado', 'cámara de representantes', 'alcalde', 'gobernador',
                'departamento', 'municipio', 'corregimiento', 'vereda', 'resguardo',
                'afrodescendientes', 'indígenas', 'mestizos', 'palenque', 'wayuu',
                'emberá', 'nasa', 'arhuaco', 'kogui', 'muisca'
            },
            'general': {
                'qué', 'cuál', 'cómo', 'dónde', 'cuándo', 'por qué', 'quién',
                'puedes', 'podrías', 'dame', 'muéstrame', 'háblame', 'quisiera',
                'me gustaría', 'explícame', 'detállame', 'cuéntame', 'señálame',
                'conocer', 'saber', 'información', 'datos', 'características',
                'descripción', 'historia de', 'origen de', 'capital de',
                'población de', 'ubicación de', 'clima de', 'cultura de'
            }
        }
        
        # Query expansion terms
        self.expansion_terms = {
            'capital':   ['bogotá', 'ciudad capital', 'sede gobierno'],
            'historia':  ['independencia', 'colonial', 'precolombino', 'república'],
            'geografía': ['territorio', 'límites', 'regiones', 'clima', 'relieve'],
            'cultura':   ['tradiciones', 'música', 'arte', 'literatura', 'festivales'],
            'economía':  ['pib', 'comercio', 'industria', 'agricultura', 'minería'],
            'población': ['habitantes', 'demografía', 'etnias', 'idiomas'],
            'moneda':    ['peso', 'pesos', 'peso colombiano', 'COP', 'divisa'],
            'deportes':  ['fútbol', 'ciclismo', 'olímpicos', 'selección colombiana'],
            'turismo':   ['lugares turísticos', 'atracciones', 'sitios de interés'],
            'café':      ['cafetero', 'eje cafetero', 'producción de café'],
            'flora':     ['biodiversidad', 'orquídeas', 'parques naturales'],
            'fauna':     ['especies endémicas', 'biodiversidad', 'fauna silvestre']
        }

    def normalize_text(self, text: str) -> str:
        """Remove accents and normalize text for better matching"""
        if not text:
            return ""
        # Remove accents and convert to lowercase
        normalized = unicodedata.normalize('NFD', text)
        ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
        return ascii_text.lower()
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess and clean the query."""
        
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")
        
        # Clean and normalize
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)  # Multiple spaces to single
        query = query.lower()
        
        # Remove unnecessary punctuation but keep question marks
        query = re.sub(r'[^\w\s¿?¡!áéíóúñü]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def validate_colombia_relevance(self, query: str) -> bool:
        """Devuelve True si la consulta está claramente relacionada con Colombia."""

        query_lower = query.lower()
        query_normalized = self.normalize_text(query)
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        query_words_normalized = set(re.findall(r'\b\w+\b', query_normalized))

        # 1 — rechazar rápido si menciona otro país popular
        foreign_countries = {
            'francia', 'france', 'españa', 'spain', 'eeuu', 'estados unidos',
            'méxico', 'mexico', 'argentina', 'chile', 'perú', 'peru',
            'alemania', 'germany', 'italia', 'italy', 'japón', 'japon', 'china',
            'brasil', 'brasil', 'canadá', 'canada', 'inglaterra', 'reino unido',
            'australia', 'india', 'rusia', 'turquía', 'turquia'
        }
        foreign_normalized = {self.normalize_text(country) for country in foreign_countries}
        if any(country in query_words_normalized for country in foreign_normalized):
            return False

        # 2 — si menciona Colombia o una ciudad/departamento → válido
        places_normalized = {self.normalize_text(place) for place in self.colombia_keywords['places']}
        if any(self.normalize_text(keyword) in query_normalized for keyword in self.colombia_keywords['places']):
            return True
        if any(place in query_words_normalized for place in places_normalized):
            return True

        # 3 — si menciona 'colombia' explícitamente → válido
        colombia_variants = {'colombia', 'colombiano', 'colombiana'}
        colombia_normalized = {self.normalize_text(variant) for variant in colombia_variants}
        if any(variant in query_normalized for variant in colombia_normalized):
            return True
        
        # 3.5 — si menciona personajes/temas históricos de Colombia → válido
        topics_normalized = {self.normalize_text(topic) for topic in self.colombia_keywords['topics']}
        if any(topic in query_normalized for topic in topics_normalized):
            return True

        # 4 — pregunta temática → no válido sin mención explícita de Colombia
        return False

    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with related terms for better retrieval."""
        
        query_lower = query.lower()
        expanded_queries = [query]
        
        # Add expansions based on detected topics
        for topic, expansions in self.expansion_terms.items():
            if topic in query_lower:
                for expansion in expansions:
                    if expansion not in query_lower:
                        expanded_query = f"{query} {expansion}"
                        expanded_queries.append(expanded_query)
        
        return expanded_queries[:3]  # Limit to 3 variations


class RetrievalService:
    """
    Service for intelligent document retrieval with query processing and re-ranking.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        vector_store_service: VectorStoreService = None
    ):
        """Initialize the retrieval service."""
        
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.query_processor = QueryProcessor()
        
        logger.info("RetrievalService initialized successfully")
    
    def retrieve_documents(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None,
        filters: Dict[str, Any] = None,
        expand_query: bool = True,
        validate_colombia_relevance: bool = True,
        include_context: bool = True
    ) -> VectorSearchResponse:
        """
        Retrieve relevant documents for a given query.
        """
        
        try:
            start_time = time.time()
            
            # Use default values from settings
            top_k = top_k or settings.top_k_documents
            similarity_threshold = similarity_threshold or settings.similarity_threshold
            
            logger.info(
                f"Starting document retrieval",
                query=query[:50],
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # Step 1: Preprocess query
            processed_query = self.query_processor.preprocess_query(query)
            
            # Step 2: Validate Colombia relevance
            if validate_colombia_relevance:
                if not self.query_processor.validate_colombia_relevance(processed_query):
                    raise QueryNotColombiaRelatedError(
                        f"Query does not appear to be related to Colombia: {query}"
                    )
            
            # Step 3: Query expansion
            query_variants = [processed_query]
            if expand_query:
                query_variants = self.query_processor.expand_query(processed_query)
            
            # Step 4: Generate embeddings for all query variants
            all_results = []
            
            for i, variant in enumerate(query_variants):
                try:
                    # Generate query embedding
                    query_embedding = self.embedding_service.generate_query_embedding(variant)
                    
                    # Search similar documents
                    search_results = self.vector_store_service.search_similar(
                        query_embedding=query_embedding,
                        top_k=top_k * 2,  # Get more results for re-ranking
                        similarity_threshold=similarity_threshold * 0.8,  # Lower threshold for variants
                        filters=filters,
                        include_content=True,
                        include_metadata=True
                    )
                    
                    # Add variant info to results
                    for result in search_results:
                        result.metadata = result.metadata or {}
                        result.metadata['query_variant'] = variant
                        result.metadata['variant_index'] = i
                    
                    all_results.extend(search_results)
                    
                    logger.debug(
                        f"Query variant {i+1} retrieved {len(search_results)} results",
                        variant=variant,
                        results_count=len(search_results)
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to process query variant '{variant}': {str(e)}")
                    continue
            
            # Step 5: Re-rank and deduplicate results
            final_results = self._rerank_and_deduplicate(
                all_results, 
                processed_query, 
                top_k
            )
            
            # Step 6: Add context if requested
            if include_context:
                final_results = self._add_context_information(final_results)
            
            retrieval_time = time.time() - start_time
            
            # Create response
            response = VectorSearchResponse(
                query=query,
                results=final_results,
                total_results=len(final_results),
                search_time_ms=retrieval_time * 1000,
                collection_name=self.vector_store_service.collection_name,
                model_name=self.embedding_service.model_name
            )
            
            logger.info(
                f"Document retrieval completed",
                query=query[:50],
                results_found=len(final_results),
                retrieval_time_ms=retrieval_time * 1000,
                query_variants_used=len(query_variants)
            )
            
            return response
            
        except (InvalidQueryError, QueryNotColombiaRelatedError) as e:
            # Re-raise validation errors
            raise e
        except Exception as e:
            error_msg = f"Document retrieval failed for query '{query}': {str(e)}"
            log_error(e, {
                "query": query,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            })
            raise RetrievalError(error_msg) from e
    
    def _rerank_and_deduplicate(
        self, 
        results: List[VectorSearchResult], 
        original_query: str, 
        top_k: int
    ) -> List[VectorSearchResult]:
        """Re-rank results and remove duplicates."""
        
        if not results:
            return []
        
        # Step 1: Remove exact duplicates based on chunk_id
        unique_results = {}
        for result in results:
            chunk_id = result.chunk_id
            if chunk_id not in unique_results or result.similarity_score > unique_results[chunk_id].similarity_score:
                unique_results[chunk_id] = result
        
        results = list(unique_results.values())
        
        # Step 2: Calculate re-ranking scores
        for result in results:
            result.metadata = result.metadata or {}
            
            # Base score is similarity score
            base_score = result.similarity_score
            
            # Bonus for exact query matches in content
            content_match_bonus = self._calculate_content_match_bonus(
                result.content or "", 
                original_query
            )
            
            # Bonus for main query variant (not expanded)
            variant_bonus = 0.1 if result.metadata.get('variant_index', 0) == 0 else 0.0
            
            # Section relevance bonus
            section_bonus = self._calculate_section_bonus(result.metadata)
            
            # Calculate final score
            final_score = base_score + content_match_bonus + variant_bonus + section_bonus
            result.metadata['rerank_score'] = final_score
            result.metadata['content_match_bonus'] = content_match_bonus
            result.metadata['section_bonus'] = section_bonus
        
        # Step 3: Sort by re-ranking score and take top_k
        results.sort(key=lambda x: x.metadata.get('rerank_score', x.similarity_score), reverse=True)
        
        return results[:top_k]
    
    def _calculate_content_match_bonus(self, content: str, query: str) -> float:
        """Calculate bonus score for direct content matches."""
        
        if not content or not query:
            return 0.0
        
        content_lower = content.lower()
        query_words = re.findall(r'\b\w+\b', query.lower())
        
        # Count exact word matches
        word_matches = sum(1 for word in query_words if word in content_lower)
        match_ratio = word_matches / len(query_words) if query_words else 0
        
        # Bonus for phrase matches
        query_phrases = [phrase.strip() for phrase in query.split() if len(phrase.strip()) > 3]
        phrase_matches = sum(1 for phrase in query_phrases if phrase.lower() in content_lower)
        phrase_bonus = phrase_matches * 0.1
        
        return min(match_ratio * 0.2 + phrase_bonus, 0.3)  # Cap at 0.3
    
    def _calculate_section_bonus(self, metadata: Dict[str, Any]) -> float:
        """Calculate bonus based on section relevance."""
        
        if not metadata:
            return 0.0
        
        section = metadata.get('source_section', '').lower()
        
        # Higher bonus for introductory sections
        if section in ['introducción', 'resumen', 'overview']:
            return 0.1
        
        # Medium bonus for main content sections
        if section in ['historia', 'geografía', 'cultura', 'economía', 'política']:
            return 0.05
        
        return 0.0
    
    def _add_context_information(
        self, 
        results: List[VectorSearchResult]
    ) -> List[VectorSearchResult]:
        """Add context information to results."""
        
        for result in results:
            if result.metadata:
                # Add readable section info
                section = result.metadata.get('source_section', 'Unknown')
                chunk_index = result.metadata.get('chunk_index', 0)
                
                result.metadata['display_section'] = section
                result.metadata['display_position'] = f"Sección {chunk_index + 1}"

                # Asegura que cada resultado exponga la URL de Wikipedia
                url = (
                    result.metadata.get("wiki_url")
                    or result.metadata.get("url")
                    or result.metadata.get("source")
                    or result.metadata.get("page")
                    or result.metadata.get("link")
                )
                if url:
                    result.metadata["wiki_url"] = url
               # ―――――――――――――――――――――――――――――――――――――――――
                
                # Add content summary
                if result.content:
                    content_words = len(result.content.split())
                    result.metadata['content_summary'] = f"{content_words} palabras"
        
        return results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval service statistics."""
        
        # Get stats from underlying services
        embedding_stats = self.embedding_service.get_service_stats()
        vector_store_stats = self.vector_store_service.get_collection_stats()
        
        return {
            'embedding_service': {
                'model_loaded': embedding_stats['model_loaded'],
                'total_embeddings_generated': embedding_stats['total_embeddings_generated'],
                'average_embedding_time_ms': embedding_stats['average_embedding_time_ms']
            },
            'vector_store': {
                'total_embeddings': vector_store_stats.total_embeddings,
                'unique_documents': vector_store_stats.unique_documents,
                'storage_size_mb': vector_store_stats.storage_size_mb
            },
            'query_processor': {
                'colombia_keywords_count': sum(len(keywords) for keywords in self.query_processor.colombia_keywords.values()),
                'expansion_terms_count': len(self.query_processor.expansion_terms)
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the retrieval service."""
        
        health_status = {
            'service_status': 'healthy',
            'last_check': datetime.utcnow().isoformat()
        }
        
        try:
            # Test embedding service
            embedding_health = self.embedding_service.health_check()
            health_status['embedding_service'] = {
                'status': 'healthy' if embedding_health.get('test_embedding_success', False) else 'degraded',
                'model_loaded': embedding_health.get('model_loaded', False)
            }
            
            # Test vector store
            vector_store_health = self.vector_store_service.health_check()
            health_status['vector_store'] = {
                'status': vector_store_health.get('service_status', 'unknown'),
                'total_embeddings': vector_store_health.get('total_embeddings', 0)
            }
            
            # Test end-to-end retrieval
            try:
                test_query = "¿Qué es Colombia?"
                test_response = self.retrieve_documents(
                    test_query, 
                    top_k=1, 
                    validate_colombia_relevance=False
                )
                
                health_status['retrieval_test'] = {
                    'status': 'success',
                    'test_query': test_query,
                    'results_found': test_response.total_results,
                    'search_time_ms': test_response.search_time_ms
                }
                
            except Exception as e:
                health_status['retrieval_test'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                health_status['service_status'] = 'degraded'
            
        except Exception as e:
            health_status.update({
                'service_status': 'degraded',
                'error': str(e)
            })
        
        return health_status