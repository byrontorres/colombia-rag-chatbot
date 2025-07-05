# API Documentation

## Overview

The Colombia RAG Chatbot exposes a RESTful API built with FastAPI that provides information about Colombia through a retrieval-augmented generation system. The API implements strict scope validation to ensure all responses are based exclusively on Wikipedia Colombia content.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API operates without authentication for development purposes. In production deployments, consider implementing:

- API key authentication
- Rate limiting per IP/user
- Request logging and monitoring

## Endpoints

### Query Endpoint

**POST /query**

Primary endpoint for chatbot interactions. Accepts natural language queries about Colombia and returns contextual answers with source attribution.

#### Request

```http
POST /query
Content-Type: application/json

{
  "query": "string"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Natural language question about Colombia |

**Query Examples:**
- "¿Cuál es la capital de Colombia?"
- "What is the geography of Colombia like?"
- "Tell me about Colombian culture"
- "¿En qué año se independizó Colombia?"

#### Response

```json
{
  "query": "string",
  "answer": "string", 
  "sources": ["string"],
  "retrieval_results": "integer",
  "retrieval_time_ms": "float",
  "generation_time_ms": "float", 
  "total_time_ms": "float",
  "metadata": {
    "model_used": "string",
    "context_length": "integer",
    "top_k_documents": "integer",
    "validation_failed": "boolean (optional)"
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| query | string | Original user query |
| answer | string | Generated response about Colombia |
| sources | array | Wikipedia URLs used as sources |
| retrieval_results | integer | Number of documents retrieved |
| retrieval_time_ms | float | Time spent on document retrieval |
| generation_time_ms | float | Time spent on answer generation |
| total_time_ms | float | Total processing time |
| metadata.model_used | string | LLM model identifier |
| metadata.context_length | integer | Characters in context sent to LLM |
| metadata.top_k_documents | integer | Documents considered for response |
| metadata.validation_failed | boolean | Present when post-generation validation fails |

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Successful response |
| 400 | Invalid query or query not related to Colombia |
| 422 | Request validation error |
| 500 | Internal server error |

#### Error Response

```json
{
  "success": false,
  "error": {
    "error_code": "string",
    "message": "string", 
    "context": {}
  },
  "status_code": "integer"
}
```

### Health Check Endpoint

**GET /health**

System health and status monitoring endpoint.

#### Response

```json
{
  "status": "healthy",
  "version": "string",
  "environment": "string", 
  "service": "colombia-rag-chatbot"
}
```

### System Information

**GET /**

Root endpoint providing basic application information.

#### Response

```json
{
  "message": "Welcome to Colombia RAG Chatbot",
  "description": "RAG-based chatbot providing information about Colombia",
  "version": "string",
  "docs_url": "string",
  "health_check": "/health"
}
```

## Query Filtering

The API implements two-layer filtering to ensure Colombia-specific responses:

### Pre-Query Validation

Queries are validated before processing:

- **Scope validation**: Rejects queries about other countries
- **Keyword matching**: Requires Colombia-related terms
- **Text normalization**: Handles UTF-8 and accent variations

**Rejected query example:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?"}'
```

**Response:**
```json
{
  "query": "What is the capital of France?",
  "answer": "Lo siento, solo puedo responder preguntas relacionadas con Colombia.",
  "sources": [],
  "retrieval_results": 0,
  "metadata": {
    "in_scope": false
  }
}
```

### Post-Generation Validation

Generated responses undergo validation:

- **Similarity threshold**: Requires minimum relevance score
- **Context verification**: Ensures response is grounded in retrieved documents
- **Content quality**: Validates response substance and length

**Insufficient context example:**
```json
{
  "query": "How many traffic lights are in Bogotá exactly?",
  "answer": "No encuentro información suficiente sobre esa consulta en el contexto de Colombia disponible.",
  "sources": [],
  "metadata": {
    "validation_failed": true
  }
}
```

## Performance Characteristics

- **Average response time**: 13 seconds
- **Typical retrieval time**: 50-100ms
- **Generation time**: 10-20 seconds (model dependent)
- **Concurrent requests**: Limited by Ollama model capacity

## Error Handling

The API implements comprehensive error handling:

### Client Errors (4xx)

- **400 Bad Request**: Query not related to Colombia
- **422 Unprocessable Entity**: Invalid request format

### Server Errors (5xx)

- **500 Internal Server Error**: Model unavailable, timeout, or processing failure

### Timeout Handling

Default timeouts:
- **Ollama model**: 20 seconds
- **Embedding generation**: 30 seconds
- **Vector search**: 10 seconds

## Rate Limiting

Current implementation supports:
- Development: No limits
- Production: Implement rate limiting per IP

Recommended production limits:
- 10 requests per minute per IP
- 100 requests per hour per IP

## API Versioning

Current version: v1
Future versions will maintain backward compatibility with deprecation notices.

## Examples

### Successful Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Qué océanos bordean a Colombia?"}'
```

**Response:**
```json
{
  "query": "¿Qué océanos bordean a Colombia?",
  "answer": "Colombia tiene costas en dos océanos: el Océano Pacífico al oeste y el Océano Atlántico (Mar Caribe) al norte.",
  "sources": [
    "https://es.wikipedia.org/wiki/Geograf%C3%ADa_de_Colombia",
    "https://es.wikipedia.org/wiki/Colombia"
  ],
  "retrieval_results": 8,
  "retrieval_time_ms": 71.2,
  "generation_time_ms": 15420.8,
  "total_time_ms": 15492.0,
  "metadata": {
    "model_used": "llama3.2:1b",
    "context_length": 1951,
    "top_k_documents": 8
  }
}
```

### Integration Example

```python
import requests
import json

def query_colombia_chatbot(question):
    url = "http://localhost:8000/query"
    payload = {"query": question}
    
    response = requests.post(
        url, 
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}")

# Usage
result = query_colombia_chatbot("¿Cuál es la capital de Colombia?")
print(result["answer"])
```