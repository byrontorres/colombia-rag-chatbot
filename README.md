# Colombia RAG Chatbot

A specialized chatbot system that provides information about Colombia using Retrieval-Augmented Generation (RAG) with Wikipedia as the exclusive data source.

## Overview

This project implements a complete RAG pipeline designed to answer questions exclusively about Colombia. The system extracts information from the Spanish Wikipedia page for Colombia, processes it through multiple stages, and generates contextual responses using an open-source language model.

The implementation focuses on production-ready practices including proper error handling, comprehensive logging, containerization, and strict scope validation to ensure responses remain within the Colombia domain.

## Architecture

The system implements a five-stage RAG pipeline:

1. **Data Extraction** - Retrieves content from Wikipedia Colombia page
2. **Text Processing** - Handles UTF-8 normalization and HTML cleanup
3. **Document Chunking** - Intelligent text segmentation (1000 characters with 200 character overlap)
4. **Embedding Generation** - Creates vector representations using sentence-transformers
5. **Response Generation** - Produces answers using LangChain and Ollama integration

### Technology Stack

- **Application Framework**: FastAPI with Python 3.12
- **Language Model**: Ollama llama3.2:1b (open source)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: ChromaDB for persistent storage
- **Orchestration**: Docker Compose for multi-service deployment

## Quick Start

### Prerequisites
- Docker and docker-compose installed
- Minimum 8GB RAM
- 10GB available disk space

### Setup Rápido
```bash
# Clone repository
git clone https://github.com/your-username/colombia-rag-chatbot.git
cd colombia-rag-chatbot

# Start services
docker-compose up --build

# Wait for services to initialize (~2-3 minutes)
```

### Verify Installation
```bash
# Check system health
curl http://localhost:8000/health

# Test query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Cuál es la capital de Colombia?"}'
```

## Evaluation Guide for Technical Reviewers

This section provides step-by-step instructions for evaluating all technical requirements from the Finaipro technical assessment.

### Requirement 1: Exclusive Wikipedia Colombia Data Source

**Verification:**
```bash
# Test Colombia-related query (should work)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Cuál es la capital de Colombia?"}'

# Expected: Valid response with Wikipedia Colombia sources

# Test foreign country query (should be rejected)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?"}'

# Expected: "Lo siento, solo puedo responder preguntas relacionadas con Colombia"
```

### Requirement 2: Five Identifiable RAG Stages

**Code Location:** All stages implemented as separate services in `app/services/`

1. **Stage 1 - Data Extraction**: `app/services/data_extractor.py` (WikipediaExtractor)
2. **Stage 2 - Text Processing**: `app/services/text_processor.py` (TextProcessor)  
3. **Stage 3 - Document Chunking**: `app/services/chunking_service.py` (IntelligentChunker)
4. **Stage 4 - Embedding Generation**: `app/services/embedding_service.py` (EmbeddingService)
5. **Stage 5 - Vector Storage & Generation**: `app/services/vector_store_service.py` + `app/services/response_generation_service.py`

**Demonstration Script:**
```bash
# Run complete pipeline demonstration
python scripts/demonstrate_pipeline.py

# Expected: Shows all 5 stages executing with metrics and results
```

### Requirement 3: LangChain Integration

**Code Location:** `app/services/response_generation_service.py`

**Verification:**
```python
# Check LangChain usage in code:
grep -r "langchain" app/services/response_generation_service.py
# Shows: from langchain_community.chat_models import ChatOllama
#        from langchain.chains import LLMChain
```

### Requirement 4: Open Source Model

**Verification:**
```bash
# Check model configuration
docker-compose exec api python -c "
from app.config.settings import settings
print(f'Model: {settings.llm_model}')
print(f'Base URL: {settings.ollama_base_url}')
"

# Expected: Model: llama3.2:1b (open source)
```

### Requirement 5: Object-Oriented Programming

**Code Location:** All services implemented as classes with proper OOP architecture

**Verification:**
```bash
# Check class definitions
find app/services -name "*.py" -exec grep "class " {} \;

# Expected: Shows multiple class definitions (WikipediaExtractor, TextProcessor, etc.)
```

### Requirement 6: FastAPI Endpoint

**Verification:**
```bash
# Test main endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Qué océanos bordean a Colombia?"}'

# Test health endpoint  
curl http://localhost:8000/health

# Test API documentation
curl http://localhost:8000/docs
```

### Requirement 7: Docker Implementation

**Verification:**
```bash
# Check Docker files exist
ls Dockerfile docker-compose.yml

# Check services running
docker-compose ps

# Expected: Shows api, ollama, and chroma services running
```

### Requirement 8: GitHub Repository

**Current Status:** Repository URL to be provided upon completion

### Complete System Verification

**Run comprehensive test suite:**
```bash
# Automated system verification
python scripts/verify_system.py

# Expected: All tests pass, showing complete system functionality
```

### Performance Testing

**Test response times and system stability:**
```bash
# Multiple test queries
for query in "¿Cuál es la capital de Colombia?" "¿Qué océanos bordean Colombia?" "Háblame sobre la cultura colombiana"
do
  echo "Testing: $query"
  time curl -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$query\"}"
  echo "---"
done
```

### Expected Results Summary

- **Query Processing Time**: ~13 seconds average
- **Colombia Queries**: Processed with Wikipedia sources cited
- **Foreign Queries**: Rejected with appropriate message
- **Source Attribution**: All responses include Wikipedia Colombia URLs
- **System Health**: All services operational
- **RAG Pipeline**: All 5 stages demonstrable via script

### Troubleshooting for Evaluators

**If system doesn't start:**
```bash
# Check Docker resources
docker system df

# Restart with fresh build
docker-compose down
docker-compose up --build --force-recreate
```

**If queries fail:**
```bash
# Check service logs
docker-compose logs api
docker-compose logs ollama

# Verify model is loaded
docker-compose exec ollama ollama list
```

**If tests fail:**
```bash
# Check individual components
curl http://localhost:8000/health
python scripts/demonstrate_pipeline.py
```

### Prerequisites

- Docker and docker-compose installed
- Minimum 8GB RAM
- 10GB available disk space

### Installation

```bash
git clone https://github.com/YOUR-USERNAME/colombia-rag-chatbot.git
cd colombia-rag-chatbot
docker-compose up --build
```

The system will automatically download required models and initialize the vector database. Initial startup may take 2-3 minutes.

### Verification

```bash
# Check system health
curl http://localhost:8000/health

# Test query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Cuál es la capital de Colombia?"}'
```

## Usage Examples

### Valid Queries

The system accepts queries related to Colombia and responds with information from Wikipedia:

```bash
# General information
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me about Colombia"}'

# Geography
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What oceans border Colombia?"}'

# History
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"When did Colombia gain independence?"}'
```

### Rejected Queries

Questions about other countries or topics outside Colombia's scope are automatically rejected:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?"}'
```

Response: "Sorry, I can only answer questions related to Colombia."

## Project Structure

```
colombia-rag-chatbot/
├── app/
│   ├── api/endpoints/         # FastAPI route handlers
│   ├── config/               # Application settings and logging
│   ├── core/exceptions.py    # Custom exception hierarchy
│   ├── models/               # Pydantic data models
│   ├── services/             # Core RAG service implementations
│   └── main.py               # FastAPI application factory
├── data/                     # Vector store and cache directory
├── tests/                    # Test suites (unit, integration, e2e)
├── requirements/             # Environment-specific dependencies
├── Dockerfile               # Multi-stage container build
├── docker-compose.yml       # Service orchestration
└── README.md               # Project documentation
```

## Configuration

### Environment Variables

Key configuration parameters can be adjusted via environment variables:

```bash
# Core service settings
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL_NAME=llama3.2:1b
CHROMA_HOST=chroma
CHROMA_PORT=8000

# Performance tuning
TOP_K_DOCUMENTS=8
SIMILARITY_THRESHOLD=0.3
MAX_CONTEXT_LENGTH=4000
```

### Customization Options

- **Language Model**: Modify `OLLAMA_MODEL_NAME` to use different Ollama models
- **Similarity Threshold**: Adjust `SIMILARITY_THRESHOLD` for retrieval sensitivity
- **Colombia Keywords**: Extend keyword lists in `app/services/retrieval_service.py`

## API Documentation

### POST /query

Primary endpoint for chatbot interactions.

**Request Body:**
```json
{
  "query": "string"
}
```

**Response:**
```json
{
  "query": "¿Cuál es la capital de Colombia?",
  "answer": "The capital of Colombia is Bogotá...",
  "sources": ["https://es.wikipedia.org/wiki/Colombia"],
  "retrieval_results": 6,
  "retrieval_time_ms": 47.2,
  "generation_time_ms": 13015.8,
  "total_time_ms": 13063.0,
  "metadata": {
    "model_used": "llama3.2:1b",
    "context_length": 1952,
    "top_k_documents": 8
  }
}
```

### GET /health

System health check endpoint returning service status and performance metrics.

### GET /docs

Interactive API documentation (Swagger UI) available in development mode.

## System Features

### Query Filtering

The system implements dual-layer filtering to ensure responses remain within the Colombia domain:

**Pre-Query Validation:**
- Rejects queries about other countries using keyword analysis
- Validates Colombia-related terms using an extensive keyword database
- Handles UTF-8 normalization for proper accent and special character processing

**Post-Generation Validation:**
- Verifies response relevance based on document similarity scores
- Rejects responses when retrieved context falls below similarity threshold
- Prevents model hallucination by requiring sufficient contextual support

### Performance Characteristics

- Average response time: 13 seconds
- Keyword coverage: 300+ Colombia-specific terms
- Foreign country rejection rate: 100%
- Data source: Exclusively Wikipedia Colombia

## Service Architecture

The application consists of six primary service components:

1. **WikipediaExtractor** - Handles content extraction from Wikipedia Colombia
2. **TextProcessor** - Manages text cleaning and UTF-8 normalization
3. **IntelligentChunker** - Performs document segmentation with overlap management
4. **EmbeddingService** - Generates vector embeddings for semantic search
5. **VectorStoreService** - Manages ChromaDB operations and retrieval
6. **ResponseGenerationService** - Coordinates LLM response generation with validation

## Testing

```bash
# Run unit tests
docker-compose exec api python -m pytest tests/unit/

# Run integration tests
docker-compose exec api python -m pytest tests/integration/

# Verify system health
curl http://localhost:8000/health
```

## Deployment

The system is containerized using Docker with multi-stage builds for optimized image size. Three services are orchestrated via docker-compose:

- **api**: FastAPI application server
- **ollama**: Local LLM inference server
- **chroma**: Vector database service

## Troubleshooting

### Common Issues

**Ollama Connection Errors:**
Check service logs and ensure the model is properly downloaded:
```bash
docker-compose logs ollama
```

**Response Timeouts:**
Adjust timeout settings in docker-compose.yml or use a smaller model variant.

**Memory Constraints:**
The system requires minimum 8GB RAM. For lower-memory environments, consider using the 1b parameter model variant.

## Technical Requirements

This implementation satisfies the following technical specifications:

- Exclusive use of Wikipedia Colombia as data source
- Complete RAG pipeline with five identifiable stages
- RESTful API endpoint following production best practices
- LangChain integration for prompt management and LLM orchestration
- Open-source language model (llama3.2:1b)
- Object-oriented programming architecture with custom exception hierarchy
- Full containerization with Docker and docker-compose
- Scope validation ensuring Colombia-only responses

## License

This project was developed as a technical assessment for Finaipro.