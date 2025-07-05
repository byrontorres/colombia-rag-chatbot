# Deployment Documentation

## Overview

This document provides comprehensive deployment guidelines for the Colombia RAG Chatbot across different environments. The system supports local development, staging, and production deployments using Docker containerization.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores (x86_64)
- RAM: 8GB
- Storage: 10GB available space
- Network: Stable internet connection for model downloads

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+ SSD
- Network: High-bandwidth connection

### Software Dependencies

- Docker Engine 24.0+
- Docker Compose 2.20+
- Git (for source code access)

**Operating System Support:**
- Linux (Ubuntu 20.04+, RHEL 8+, CentOS 8+)
- macOS 12+
- Windows 10+ (with WSL2)

## Local Development Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/colombia-rag-chatbot.git
cd colombia-rag-chatbot

# Start development environment
docker-compose up --build

# Verify deployment
curl http://localhost:8000/health
```

### Development Configuration

The development environment uses `docker-compose.yml` with the following services:

**API Service:**
- Port: 8000
- Environment: development
- Hot reload: enabled
- Debug logging: enabled

**Ollama Service:**
- Port: 11434
- Model: llama3.2:1b
- GPU acceleration: disabled (CPU only)

**ChromaDB Service:**
- Port: 8001
- Persistence: ./data/chroma_db
- Memory limit: 2GB

### Environment Variables

Create `.env` file for local development:

```bash
# Application settings
APP_NAME=Colombia RAG Chatbot
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Ollama configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL_NAME=llama3.2:1b
OLLAMA_TIMEOUT=30

# ChromaDB configuration
CHROMA_HOST=chroma
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=colombia_documents

# RAG parameters
TOP_K_DOCUMENTS=8
SIMILARITY_THRESHOLD=0.3
MAX_CONTEXT_LENGTH=4000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# CORS settings
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

### Development Workflow

1. **Initial Setup:**
   ```bash
   docker-compose up --build -d
   docker-compose logs -f api
   ```

2. **Model Download (first run):**
   ```bash
   # Wait for Ollama to download llama3.2:1b model
   # This may take 5-10 minutes depending on connection
   docker-compose logs ollama
   ```

3. **Data Ingestion:**
   ```bash
   # Trigger Wikipedia data extraction
   curl -X POST "http://localhost:8000/admin/ingest" \
     -H "Content-Type: application/json"
   ```

4. **Testing:**
   ```bash
   # Functional test
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query":"¿Cuál es la capital de Colombia?"}'
   ```

## Production Deployment

### Production Configuration

Create `.env.production` for production settings:

```bash
# Application settings
APP_NAME=Colombia RAG Chatbot
APP_VERSION=1.0.0
ENVIRONMENT=production
DEBUG=false

# Security settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=WARNING

# Ollama configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL_NAME=llama3.2:1b
OLLAMA_TIMEOUT=60

# ChromaDB configuration with persistence
CHROMA_HOST=chroma
CHROMA_PORT=8000
CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db

# Production RAG parameters
TOP_K_DOCUMENTS=10
SIMILARITY_THRESHOLD=0.35
MAX_CONTEXT_LENGTH=6000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# CORS for production
CORS_ORIGINS=["https://yourdomain.com"]

# Resource limits
OLLAMA_NUM_THREAD=4
OLLAMA_NUM_GPU=0
```

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env.production
    depends_on:
      - ollama
      - chroma
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

  chroma:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/version"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

volumes:
  ollama_models:
  chroma_data:

networks:
  default:
    driver: bridge
```

### Production Deployment Steps

1. **Server Preparation:**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install Docker Compose
   sudo apt install docker-compose-plugin
   ```

2. **Application Deployment:**
   ```bash
   # Clone repository
   git clone https://github.com/your-username/colombia-rag-chatbot.git
   cd colombia-rag-chatbot
   
   # Configure environment
   cp .env.production .env
   
   # Deploy with production configuration
   docker-compose -f docker-compose.prod.yml up -d --build
   ```

3. **Verification:**
   ```bash
   # Check service status
   docker-compose -f docker-compose.prod.yml ps
   
   # Verify health
   curl http://localhost:8000/health
   
   # Monitor logs
   docker-compose -f docker-compose.prod.yml logs -f
   ```

## Cloud Deployment Options

### AWS Deployment

**EC2 Instance Setup:**

1. **Launch EC2 Instance:**
   - Instance Type: t3.large (2 vCPU, 8GB RAM) minimum
   - AMI: Ubuntu 22.04 LTS
   - Security Group: Allow ports 22, 80, 443, 8000

2. **ECS Deployment (Alternative):**
   ```yaml
   # task-definition.json
   {
     "family": "colombia-rag-chatbot",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "2048",
     "memory": "8192",
     "containerDefinitions": [
       {
         "name": "api",
         "image": "your-account.dkr.ecr.region.amazonaws.com/colombia-rag:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "ENVIRONMENT",
             "value": "production"
           }
         ]
       }
     ]
   }
   ```

### Google Cloud Platform

**Cloud Run Deployment:**

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/colombia-rag:latest', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/colombia-rag:latest']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'colombia-rag-chatbot'
      - '--image'
      - 'gcr.io/$PROJECT_ID/colombia-rag:latest'
      - '--platform'
      - 'managed'
      - '--region'
      - 'us-central1'
      - '--memory'
      - '8Gi'
      - '--cpu'
      - '4'
      - '--timeout'
      - '300'
```

### Railway Deployment

```yaml
# railway.toml
[build]
dockerfile = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 30
restartPolicy = "on-failure"

[[services]]
name = "api"
source = "."

[services.env]
PORT = "8000"
ENVIRONMENT = "production"
```

## Container Registry

### Building and Pushing Images

```bash
# Build production image
docker build -t colombia-rag:latest .

# Tag for registry
docker tag colombia-rag:latest your-registry/colombia-rag:latest

# Push to registry
docker push your-registry/colombia-rag:latest
```

### Multi-architecture Builds

```bash
# Build for multiple architectures
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t your-registry/colombia-rag:latest --push .
```

## Monitoring and Logging

### Production Monitoring

**Health Check Endpoints:**
- `/health` - Overall system health
- `/health/detailed` - Component-specific health
- `/metrics` - Prometheus metrics (if implemented)

**Log Aggregation:**
```yaml
# docker-compose.prod.yml addition
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**External Monitoring:**
```bash
# Prometheus monitoring
curl http://localhost:8000/metrics

# Custom health check
curl -f http://localhost:8000/health || exit 1
```

### Log Management

**Centralized Logging with ELK:**
```yaml
# docker-compose.logging.yml
services:
  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    
  logstash:
    image: logstash:8.8.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    
  kibana:
    image: kibana:8.8.0
    ports:
      - "5601:5601"
```

## Backup and Recovery

### Data Backup Strategy

```bash
# Backup ChromaDB data
docker-compose exec chroma tar -czf /backup/chroma-$(date +%Y%m%d).tar.gz /chroma/chroma

# Backup Ollama models
docker-compose exec ollama tar -czf /backup/ollama-$(date +%Y%m%d).tar.gz /root/.ollama

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

docker-compose exec -T chroma tar -czf - /chroma/chroma > $BACKUP_DIR/chroma.tar.gz
docker-compose exec -T ollama tar -czf - /root/.ollama > $BACKUP_DIR/ollama.tar.gz

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR s3://your-backup-bucket/$(date +%Y%m%d)/ --recursive
```

### Disaster Recovery

```bash
# Restore from backup
docker-compose down
docker volume rm colombia-rag-chatbot_chroma_data colombia-rag-chatbot_ollama_models

# Restore data
docker run --rm -v colombia-rag-chatbot_chroma_data:/chroma \
  -v /backups/20240101:/backup alpine \
  tar -xzf /backup/chroma.tar.gz -C /

docker run --rm -v colombia-rag-chatbot_ollama_models:/root/.ollama \
  -v /backups/20240101:/backup alpine \
  tar -xzf /backup/ollama.tar.gz -C /

docker-compose up -d
```

## Troubleshooting

### Common Issues

**1. Ollama Model Download Fails:**
```bash
# Check Ollama logs
docker-compose logs ollama

# Manually pull model
docker-compose exec ollama ollama pull llama3.2:1b

# Verify model availability
docker-compose exec ollama ollama list
```

**2. ChromaDB Connection Issues:**
```bash
# Check ChromaDB health
curl http://localhost:8001/api/v1/version

# Reset ChromaDB data
docker-compose down
docker volume rm colombia-rag-chatbot_chroma_data
docker-compose up -d chroma
```

**3. Memory Issues:**
```bash
# Monitor memory usage
docker stats

# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G
```

**4. API Timeout Issues:**
```bash
# Increase timeout settings
environment:
  - OLLAMA_TIMEOUT=120
  - REQUEST_TIMEOUT=180
```

### Performance Optimization

**Model Optimization:**
```bash
# Use smaller model for resource-constrained environments
OLLAMA_MODEL_NAME=llama3.2:1b  # Current
OLLAMA_MODEL_NAME=phi3:mini    # Alternative smaller model
```

**Database Optimization:**
```bash
# ChromaDB performance tuning
environment:
  - CHROMA_BATCH_SIZE=100
  - CHROMA_MAX_BATCH_SIZE=1000
```

### Security Hardening

**Production Security Checklist:**

1. **Network Security:**
   ```bash
   # Use internal Docker networks
   networks:
     internal:
       driver: bridge
       internal: true
   ```

2. **Container Security:**
   ```dockerfile
   # Run as non-root user
   USER nobody
   
   # Read-only filesystem
   read_only: true
   ```

3. **Secrets Management:**
   ```bash
   # Use Docker secrets
   docker secret create api_key api_key.txt
   
   # Reference in compose
   secrets:
     - api_key
   ```

## Scaling Considerations

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  api:
    deploy:
      replicas: 3
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
```

**Nginx Load Balancer Configuration:**
```nginx
upstream api_backend {
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Vertical Scaling

```yaml
# Increase resources for single instance
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 32G
          cpus: '8'
        reservations:
          memory: 16G
          cpus: '4'
```

## Maintenance Procedures

### Regular Maintenance Tasks

**Weekly:**
- Monitor disk usage and clean logs
- Verify backup integrity
- Update security patches

**Monthly:**
- Update Docker images
- Review performance metrics
- Optimize vector database

**Quarterly:**
- Update application dependencies
- Review and update documentation
- Disaster recovery testing

### Update Procedures

```bash
# Application updates
git pull origin main
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verify update
curl http://localhost:8000/health
docker-compose logs api
```