# Architecture Documentation

## System Overview

The Colombia RAG Chatbot implements a retrieval-augmented generation architecture designed for domain-specific question answering. The system processes natural language queries about Colombia and generates contextual responses using information exclusively from Wikipedia Colombia.

## High-Level Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │───▶│  FastAPI     │───▶│   Ollama    │
│ Application │    │   Gateway    │    │ LLM Server  │
└─────────────┘    └──────────────┘    └─────────────┘
                           │
                           ▼
                   ┌──────────────┐    ┌─────────────┐
                   │   RAG Core   │───▶│  ChromaDB   │
                   │   Services   │    │ Vector Store│
                   └──────────────┘    └─────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │  Wikipedia   │
                   │ Data Source  │
                   └──────────────┘
```

## Core Components

### 1. API Gateway Layer

**FastAPI Application (`app/main.py`)**
- Request routing and validation
- Error handling and response formatting
- CORS middleware and security headers
- Request/response logging

**Endpoints (`app/api/endpoints/`)**
- `chat.py`: Primary query processing endpoint
- `health.py`: System health monitoring

### 2. Service Layer

The system implements six core services following single responsibility principle:

#### WikipediaExtractor (`app/services/wikipedia_extractor.py`)
- Extracts content from Wikipedia Colombia page
- Validates source domain and URL structure
- Handles HTTP requests with retry logic and caching
- Implements rate limiting for responsible API usage

#### TextProcessor (`app/services/text_processor.py`)
- UTF-8 normalization and encoding handling
- HTML tag removal and text cleaning
- Special character and accent normalization
- Content deduplication and filtering

#### IntelligentChunker (`app/services/intelligent_chunker.py`)
- Document segmentation with semantic awareness
- Configurable chunk size (1000 characters) and overlap (200 characters)
- Section boundary preservation
- Metadata extraction and tagging

#### EmbeddingService (`app/services/embedding_service.py`)
- Vector embedding generation using sentence-transformers
- Model: `all-MiniLM-L6-v2` for multilingual support
- Batch processing for efficiency
- Embedding caching and reuse

#### VectorStoreService (`app/services/vector_store_service.py`)
- ChromaDB integration for vector storage
- Similarity search with configurable parameters
- Metadata filtering and retrieval
- Collection management and persistence

#### ResponseGenerationService (`app/services/response_generation_service.py`)
- LangChain integration for prompt management
- Ollama LLM coordination (llama3.2:1b)
- Context preparation and length management
- Response validation and quality control

### 3. Data Processing Pipeline

#### Stage 1: Data Extraction
```python
# WikipediaExtractor
raw_content = extractor.extract_content(
    url="https://es.wikipedia.org/wiki/Colombia"
)
```

#### Stage 2: Text Processing
```python
# TextProcessor
clean_content = processor.clean_text(raw_content)
normalized_content = processor.normalize_encoding(clean_content)
```

#### Stage 3: Document Chunking
```python
# IntelligentChunker
chunks = chunker.create_chunks(
    text=normalized_content,
    chunk_size=1000,
    overlap=200
)
```

#### Stage 4: Embedding Generation
```python
# EmbeddingService
embeddings = embedding_service.generate_embeddings(chunks)
```

#### Stage 5: Vector Storage
```python
# VectorStoreService
vector_store.store_embeddings(chunks, embeddings, metadata)
```

#### Stage 6: Query Processing and Generation
```python
# Query flow
query → validation → retrieval → context_prep → generation → validation → response
```

## Data Flow

### Query Processing Flow

1. **Request Reception**
   - FastAPI receives HTTP POST request
   - Request validation (Pydantic models)
   - Query extraction and preprocessing

2. **Pre-Query Validation**
   - Colombia relevance checking
   - Foreign country rejection
   - Keyword analysis and normalization

3. **Document Retrieval**
   - Query embedding generation
   - Vector similarity search
   - Top-K document selection
   - Metadata enrichment

4. **Context Preparation**
   - Document ranking and scoring
   - Context length optimization
   - Source attribution preparation

5. **Response Generation**
   - LangChain prompt construction
   - Ollama LLM invocation
   - Response extraction and formatting

6. **Post-Generation Validation**
   - Similarity threshold verification
   - Content quality assessment
   - Hallucination prevention

7. **Response Delivery**
   - Source URL compilation
   - Performance metrics collection
   - JSON response formatting

## Design Patterns

### Service Layer Pattern
- Clear separation of concerns
- Dependency injection for testability
- Interface-based design for flexibility

### Repository Pattern
- VectorStoreService abstracts data persistence
- Consistent data access interface
- Swap-able storage backends

### Chain of Responsibility
- Query validation pipeline
- Error handling chain
- Response processing stages

### Singleton Pattern
- LLM instance management (resource optimization)
- Configuration settings
- Logging infrastructure

## Configuration Management

### Settings Architecture (`app/config/settings.py`)
```python
class Settings(BaseSettings):
    # Application settings
    app_name: str = "Colombia RAG Chatbot"
    app_version: str = "1.0.0"
    
    # LLM configuration
    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "llama3.2:1b"
    
    # Vector store settings
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    
    # Performance tuning
    top_k_documents: int = 8
    similarity_threshold: float = 0.3
    max_context_length: int = 4000
```

### Environment-Specific Configuration
- Development: Verbose logging, debug endpoints
- Production: Error logging only, security headers
- Testing: In-memory storage, mock services

## Error Handling Architecture

### Exception Hierarchy (`app/core/exceptions.py`)
```python
ColombiaRAGException (base)
├── ValidationError
│   ├── InvalidQueryError
│   └── QueryNotColombiaRelatedError
├── ProcessingError
│   ├── ExtractionError
│   ├── ChunkingError
│   └── EmbeddingError
├── ServiceError
│   ├── ModelError
│   ├── VectorStoreError
│   └── GenerationError
└── ConfigurationError
```

### Error Propagation Strategy
- Service-level error capture and context enrichment
- Structured error logging with correlation IDs
- Client-friendly error message translation
- Performance impact minimization

## Security Architecture

### Current Implementation
- Input validation and sanitization
- CORS configuration for cross-origin requests
- Request size limits and timeout handling
- UTF-8 encoding validation

### Production Recommendations
- API key authentication
- Rate limiting per IP/user
- Request/response encryption (HTTPS)
- Input sanitization against injection attacks
- Audit logging for security events

## Performance Considerations

### Optimization Strategies
- **Model Caching**: Singleton LLM instance
- **Embedding Reuse**: Vector store persistence
- **Connection Pooling**: HTTP client optimization
- **Batch Processing**: Multiple embedding generation
- **Context Compression**: Intelligent truncation

### Bottleneck Analysis
- **Primary**: LLM inference time (10-20 seconds)
- **Secondary**: Vector similarity search (50-100ms)
- **Tertiary**: Network latency for Wikipedia extraction

### Scaling Considerations
- **Horizontal**: Multiple API instances behind load balancer
- **Vertical**: Larger models require more memory/compute
- **Caching**: Redis for embedding and response caching
- **CDN**: Static asset delivery optimization

## Monitoring and Observability

### Logging Strategy (`app/config/logging.py`)
- Structured JSON logging
- Correlation ID tracking
- Performance metrics collection
- Error rate monitoring

### Health Checks
- Service dependency verification
- Model availability confirmation
- Database connectivity testing
- Performance threshold monitoring

### Metrics Collection
- Request/response latency
- Error rates by type
- Model inference performance
- Resource utilization tracking

## Development Architecture

### Code Organization
```
app/
├── api/           # API layer (routes, middleware)
├── config/        # Configuration and logging
├── core/          # Core utilities and exceptions
├── models/        # Data models and schemas
├── services/      # Business logic services
└── utils/         # Helper functions and utilities
```

### Testing Strategy
- **Unit Tests**: Individual service testing
- **Integration Tests**: Service interaction testing
- **End-to-End Tests**: Full pipeline validation
- **Performance Tests**: Load and stress testing

### Deployment Architecture
- **Containerization**: Docker multi-stage builds
- **Orchestration**: Docker Compose for local development
- **Service Mesh**: Individual service containers
- **Data Persistence**: Volume mounting for vector store

## Technology Decisions

### Language Model Selection
- **Choice**: Ollama llama3.2:1b
- **Rationale**: 
  - Open source compliance
  - Local deployment capability
  - Multilingual support (Spanish/English)
  - Reasonable resource requirements
  - Good performance for domain-specific tasks

### Vector Database Selection
- **Choice**: ChromaDB
- **Rationale**:
  - Python-native integration
  - Simple deployment and management
  - Good performance for medium-scale datasets
  - Metadata filtering capabilities
  - Local persistence support

### Framework Selection
- **Choice**: FastAPI
- **Rationale**:
  - High performance async capabilities
  - Automatic API documentation
  - Type safety with Pydantic
  - Modern Python features support
  - Excellent testing capabilities

## Future Architecture Considerations

### Scalability Enhancements
- Microservices decomposition
- Event-driven architecture for async processing
- Caching layer for common queries
- CDN integration for static assets

### Feature Extensions
- Multi-language support beyond Spanish/English
- Real-time data source updates
- Advanced query understanding (intent recognition)
- Conversation context management

### Infrastructure Evolution
- Kubernetes deployment for production
- Service mesh for advanced networking
- Distributed vector storage
- ML pipeline automation