# Technical Documentation

This directory contains comprehensive technical documentation for the Colombia RAG Chatbot system, providing detailed information for developers, system administrators, and technical evaluators.

## Documentation Overview

The Colombia RAG Chatbot is a production-ready retrieval-augmented generation system that provides information exclusively about Colombia using Wikipedia as its data source. The system implements a complete five-stage RAG pipeline with strict scope validation and professional deployment practices.

## Documentation Structure

### [API Documentation](api/)
Complete reference for the REST API endpoints, including request/response formats, error handling, authentication, and integration examples.

**Key Sections:**
- Endpoint specifications with examples
- Query filtering and validation rules
- Response format documentation
- Performance characteristics
- Integration guidelines

### [Architecture Documentation](architecture/)
Comprehensive system design documentation covering the technical implementation, design patterns, and component interactions.

**Key Sections:**
- High-level system architecture
- RAG pipeline implementation details
- Service layer design patterns
- Data flow and processing stages
- Technology selection rationale

### [Deployment Documentation](deployment/)
Complete deployment guides for different environments, from local development to production cloud deployments.

**Key Sections:**
- Local development setup
- Production deployment procedures
- Cloud platform configurations
- Monitoring and maintenance
- Troubleshooting guides

## Quick Navigation

### For Developers
- **Getting Started**: [Deployment Guide - Local Development](deployment/README.md#local-development-deployment)
- **API Integration**: [API Documentation - Endpoints](api/README.md#endpoints)
- **System Architecture**: [Architecture Documentation - Core Components](architecture/README.md#core-components)

### For System Administrators
- **Production Deployment**: [Deployment Guide - Production](deployment/README.md#production-deployment)
- **Monitoring Setup**: [Deployment Guide - Monitoring](deployment/README.md#monitoring-and-logging)
- **Troubleshooting**: [Deployment Guide - Troubleshooting](deployment/README.md#troubleshooting)

### For Technical Evaluators
- **RAG Pipeline Details**: [Architecture Documentation - Data Processing Pipeline](architecture/README.md#data-processing-pipeline)
- **Query Validation**: [API Documentation - Query Filtering](api/README.md#query-filtering)
- **Performance Metrics**: [API Documentation - Performance](api/README.md#performance-characteristics)

## System Requirements

### Technical Implementation
- **Framework**: FastAPI with Python 3.12
- **Language Model**: Ollama llama3.2:1b (open source)
- **Vector Database**: ChromaDB for persistent storage
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Containerization**: Docker with multi-service orchestration

### Key Features
- Five-stage RAG pipeline with clear separation
- Exclusive Wikipedia Colombia data source
- Pre and post-query validation for scope control
- Object-oriented architecture with custom exceptions
- Production-ready error handling and logging
- Comprehensive API documentation and testing

## RAG Pipeline Stages

The system implements five clearly identifiable stages:

1. **Data Extraction** - Wikipedia Colombia content retrieval with domain validation
2. **Text Processing** - UTF-8 normalization, HTML cleanup, and content sanitization
3. **Document Chunking** - Intelligent segmentation with configurable overlap
4. **Embedding Generation** - Vector representation using sentence transformers
5. **Response Generation** - LLM-powered answer synthesis with source attribution

## Scope Validation

The system ensures all responses are Colombia-specific through:

- **Pre-query filtering** - Keyword analysis and foreign country rejection
- **Content validation** - Wikipedia Colombia source verification
- **Post-generation checks** - Response relevance and similarity thresholds
- **Source attribution** - Proper Wikipedia Colombia URL references

## Getting Started

### Quick Setup
```bash
# Clone and start the system
git clone https://github.com/your-username/colombia-rag-chatbot.git
cd colombia-rag-chatbot
docker-compose up --build

# Verify functionality
curl http://localhost:8000/health
```

### Documentation Reading Order

1. **Start Here**: [Architecture Documentation](architecture/) - Understand the system design
2. **Deploy Locally**: [Deployment Guide](deployment/) - Get the system running
3. **Integrate**: [API Documentation](api/) - Learn the endpoints and responses

### Example Usage

```bash
# Test the complete RAG pipeline
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Cuál es la capital de Colombia?"}'

# Verify query filtering
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?"}'
```

## Support and Troubleshooting

### Common Issues
- **Model Loading**: See [Deployment Troubleshooting](deployment/README.md#troubleshooting)
- **API Errors**: See [API Documentation - Error Handling](api/README.md#error-handling)
- **Performance**: See [Architecture - Performance Considerations](architecture/README.md#performance-considerations)

### System Verification
```bash
# Run comprehensive system tests
python scripts/verify_system.py

# Demonstrate RAG pipeline stages
python scripts/demonstrate_pipeline.py
```

## Technical Compliance

This implementation satisfies all technical requirements:

- Exclusive Wikipedia Colombia data source
- Five identifiable RAG stages with clear separation
- LangChain integration for prompt management
- Open-source language model (llama3.2:1b)
- Object-oriented programming architecture
- Production-ready FastAPI endpoints
- Complete Docker containerization
- Comprehensive scope validation for Colombia-only responses

## Contributing to Documentation

When updating documentation:

1. Maintain consistency across all documents
2. Update cross-references when adding new sections
3. Test all code examples and commands
4. Follow the established documentation structure
5. Update this index when adding new major sections

For technical questions or clarifications about the system architecture, refer to the detailed documentation in each specialized section.