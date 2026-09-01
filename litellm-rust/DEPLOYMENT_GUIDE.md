# LiteLLM Rust Gateway - Deployment Guide

## Overview

This guide covers deploying the LiteLLM Rust Gateway in production environments, including standalone deployments, Docker, Kubernetes, and cloud platforms.

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 2 cores
- Memory: 512 MB
- Disk: 100 MB
- Network: Low latency to LLM providers

**Recommended:**
- CPU: 4+ cores
- Memory: 2+ GB
- Disk: 1 GB
- Network: <100ms latency to LLM providers

### Software Requirements

- Rust 1.70 or later (for building from source)
- Redis 6.0+ (for rate limiting and caching)
- PostgreSQL 13+ (for persistent spend logs)

### LLM Provider API Keys

You'll need API keys for at least one LLM provider:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/
- AWS Bedrock: AWS Console
- etc.

## Building from Source

### 1. Clone the Repository

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm/litellm-rust
```

### 2. Build the Binary

```bash
cargo build --release --features server
```

The binary will be at `target/release/litellm-ai-gateway`.

### 3. Verify the Build

```bash
./target/release/litellm-ai-gateway --help
```

## Configuration

### Basic Configuration

Create `config.yaml`:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: ${OPENAI_API_KEY}

general_settings:
  master_key: ${LITELLM_MASTER_KEY}
```

### Production Configuration

```yaml
model_list:
  # Primary model
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: ${OPENAI_API_KEY}
  
  # Fallback model
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4-turbo
      api_key: ${OPENAI_API_KEY}
  
  # Anthropic model
  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: ${ANTHROPIC_API_KEY}

general_settings:
  master_key: ${LITELLM_MASTER_KEY}
  
  # Redis for rate limiting and caching
  redis_url: ${REDIS_URL}
  
  # PostgreSQL for spend logs
  database_url: ${DATABASE_URL}
  
  # Timeouts
  request_timeout: 600
  cache_ttl: 300
  
  # Logging
  log_level: info
```

### Environment Variables

Create `.env` file:

```bash
# Master key for gateway authentication
LITELLM_MASTER_KEY=sk-your-secure-master-key

# LLM Provider API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Redis (optional but recommended)
REDIS_URL=redis://localhost:6379

# PostgreSQL (optional but recommended)
DATABASE_URL=postgresql://user:password@localhost:5432/litellm

# Gateway settings
HOST=0.0.0.0
PORT=4001
RUST_LOG=info
```

## Standalone Deployment

### 1. Prepare the Server

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y redis postgresql

# Create service user
sudo useradd -r -s /bin/false litellm
```

### 2. Install the Gateway

```bash
# Create directories
sudo mkdir -p /opt/litellm/{bin,config,logs}

# Copy binary
sudo cp target/release/litellm-ai-gateway /opt/litellm/bin/

# Copy config
sudo cp config.yaml /opt/litellm/config/

# Set permissions
sudo chown -R litellm:litellm /opt/litellm
sudo chmod +x /opt/litellm/bin/litellm-ai-gateway
```

### 3. Configure Redis

```bash
# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Verify Redis is running
redis-cli ping
```

### 4. Configure PostgreSQL

```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql <<EOF
CREATE USER litellm WITH PASSWORD 'your-secure-password';
CREATE DATABASE litellm OWNER litellm;
EOF
```

### 5. Create Systemd Service

Create `/etc/systemd/system/litellm-gateway.service`:

```ini
[Unit]
Description=LiteLLM Rust Gateway
After=network.target redis.service postgresql.service

[Service]
Type=simple
User=litellm
Group=litellm
EnvironmentFile=/opt/litellm/config/.env
ExecStart=/opt/litellm/bin/litellm-ai-gateway
Restart=always
RestartSec=10
StandardOutput=append:/opt/litellm/logs/gateway.log
StandardError=append:/opt/litellm/logs/gateway.error.log

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

### 6. Start the Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Start the service
sudo systemctl start litellm-gateway

# Enable on boot
sudo systemctl enable litellm-gateway

# Check status
sudo systemctl status litellm-gateway
```

### 7. Verify Deployment

```bash
# Check health
curl http://localhost:4001/health/liveness

# Test a request
curl http://localhost:4001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM rust:1.70-bookworm as builder

WORKDIR /app

# Copy source
COPY . .

# Build
RUN cargo build --release --features server

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/target/release/litellm-ai-gateway /usr/local/bin/

# Create non-root user
RUN useradd -r -s /bin/false litellm
USER litellm

# Expose port
EXPOSE 4001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:4001/health/liveness || exit 1

# Start gateway
CMD ["litellm-ai-gateway"]
```

### 2. Build Docker Image

```bash
docker build -t litellm-gateway:latest .
```

### 3. Run with Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # No `ports:` on redis/postgres: the gateway reaches them over the Compose
  # network, and publishing them would expose them on every host interface.
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: litellm
      POSTGRES_PASSWORD: your-secure-password
      POSTGRES_DB: litellm
    volumes:
      - postgres-data:/var/lib/postgresql/data

  gateway:
    image: litellm-gateway:latest
    ports:
      - "4001:4001"
    environment:
      - LITELLM_YAML_CONFIG=/config/config.yaml
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://litellm:your-secure-password@postgres:5432/litellm
      - RUST_LOG=info
    volumes:
      - ./config.yaml:/config/config.yaml:ro
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
```

### 4. Start with Docker Compose

```bash
# Create .env file
cat > .env <<EOF
LITELLM_MASTER_KEY=sk-your-master-key
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
EOF

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f gateway
```

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace litellm
```

### 2. Create Secrets

```bash
kubectl create secret generic litellm-secrets \
  --namespace=litellm \
  --from-literal=master-key=sk-your-master-key \
  --from-literal=openai-api-key=sk-your-openai-key \
  --from-literal=anthropic-api-key=sk-ant-your-anthropic-key
```

### 3. Create ConfigMap

```bash
kubectl create configmap litellm-config \
  --namespace=litellm \
  --from-file=config.yaml
```

### 4. Create Deployment

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-gateway
  namespace: litellm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: litellm-gateway
  template:
    metadata:
      labels:
        app: litellm-gateway
    spec:
      containers:
      - name: gateway
        image: litellm-gateway:latest
        ports:
        - containerPort: 4001
        env:
        - name: LITELLM_YAML_CONFIG
          value: /config/config.yaml
        - name: LITELLM_MASTER_KEY
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: master-key
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: anthropic-api-key
        - name: REDIS_URL
          value: redis://redis:6379
        - name: DATABASE_URL
          value: postgresql://litellm:password@postgres:5432/litellm
        - name: RUST_LOG
          value: info
        volumeMounts:
        - name: config
          mountPath: /config
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 4001
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 4001
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: litellm-config
```

### 5. Create Service

Create `service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: litellm-gateway
  namespace: litellm
spec:
  selector:
    app: litellm-gateway
  ports:
  - port: 4001
    targetPort: 4001
  type: ClusterIP
```

### 6. Create Ingress (Optional)

Create `ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: litellm-gateway
  namespace: litellm
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: litellm-gateway
            port:
              number: 4001
```

### 7. Deploy to Kubernetes

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment
kubectl get pods -n litellm
kubectl logs -n litellm deployment/litellm-gateway
```

## Cloud Platform Deployment

### AWS

#### Using ECS

1. Push Docker image to ECR
2. Create ECS cluster
3. Create task definition
4. Create service with load balancer

#### Using EKS

1. Push Docker image to ECR
2. Deploy using Kubernetes manifests above
3. Use ALB Ingress Controller

### GCP

#### Using Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/litellm-gateway

# Deploy to Cloud Run
gcloud run deploy litellm-gateway \
  --image gcr.io/YOUR_PROJECT/litellm-gateway \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars LITELLM_MASTER_KEY=sk-your-master-key
```

#### Using GKE

1. Push Docker image to Container Registry
2. Deploy using Kubernetes manifests above
3. Use GCE Ingress Controller

### Azure

#### Using Container Instances

```bash
# Push to Azure Container Registry
az acr build --registry YOUR_REGISTRY --image litellm-gateway .

# Deploy to Container Instances
az container create \
  --resource-group YOUR_RESOURCE_GROUP \
  --name litellm-gateway \
  --image YOUR_REGISTRY.azurecr.io/litellm-gateway \
  --dns-name-label litellm-gateway \
  --ports 4001 \
  --environment-variables LITELLM_MASTER_KEY=sk-your-master-key
```

#### Using AKS

1. Push Docker image to Azure Container Registry
2. Deploy using Kubernetes manifests above
3. Use Application Gateway Ingress Controller

## Monitoring

### Prometheus Metrics

The gateway exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:4001/metrics
```

Key metrics:
- `litellm_requests_total` - Total request count
- `litellm_request_duration_seconds` - Request latency
- `litellm_tokens_total` - Token usage
- `litellm_spend_usd_total` - Spend tracking

### Logging

Logs are output to stdout in JSON format. Configure log level with `RUST_LOG`:

```bash
RUST_LOG=info ./litellm-ai-gateway
```

### Health Checks

```bash
# Liveness
curl http://localhost:4001/health/liveness

# Readiness
curl http://localhost:4001/health/readiness

# Deep health check
curl http://localhost:4001/health/deep
```

## Security

### API Key Security

1. Use strong, randomly generated API keys
2. Rotate keys regularly
3. Use environment variables or secret managers
4. Never commit keys to version control

### Network Security

1. Use HTTPS/TLS for all connections
2. Use a reverse proxy (nginx, HAProxy) for TLS termination
3. Restrict access using firewalls/security groups
4. Use VPC/private networks for internal communication

### Infrastructure Security

1. Run as non-root user
2. Use container security contexts
3. Enable audit logging
4. Regular security updates
5. Monitor for unusual activity

## Scaling

### Horizontal Scaling

The gateway is stateless and can be scaled horizontally:

1. Deploy multiple instances behind a load balancer
2. Use Redis for shared rate limiting
3. Use PostgreSQL for shared spend tracking

### Vertical Scaling

Increase resources for single instances:
- CPU: For higher throughput
- Memory: For larger caches
- Network: For lower latency

### Auto-scaling

Configure auto-scaling based on:
- CPU utilization
- Request rate
- Latency

## Troubleshooting

### Gateway won't start

1. Check logs: `journalctl -u litellm-gateway` or `docker logs`
2. Verify config file is valid YAML
3. Check environment variables are set
4. Verify Redis/PostgreSQL are accessible

### High latency

1. Check provider latency
2. Check network latency
3. Check circuit breaker state
4. Check Redis/PostgreSQL performance
5. Consider scaling horizontally

### Rate limit errors

1. Check rate limits for API key
2. Check Redis for current counters
3. Consider increasing limits
4. Use multiple API keys

### Circuit breaker open

1. Check provider status
2. Check logs for error details
3. Wait for recovery timeout
4. Check provider API status page

## Backup and Recovery

### Configuration Backup

```bash
# Backup config
cp config.yaml config.yaml.backup

# Backup secrets
kubectl get secret litellm-secrets -o yaml > secrets-backup.yaml
```

### Database Backup

```bash
# PostgreSQL backup
pg_dump -U litellm litellm > backup.sql

# Restore
psql -U litellm litellm < backup.sql
```

### Redis Backup

Redis is used for caching and rate limiting. Data can be regenerated, but you may want to backup:

```bash
# Redis backup
redis-cli BGSAVE

# Copy dump.rdb
```

## Maintenance

### Updates

```bash
# Pull latest code
git pull

# Rebuild
cargo build --release

# Restart service
sudo systemctl restart litellm-gateway
```

### Log Rotation

Configure log rotation for `/opt/litellm/logs/`:

```bash
# /etc/logrotate.d/litellm
/opt/litellm/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 litellm litellm
}
```

## Support

For issues and questions:

1. Check logs for error messages
2. Review documentation
3. Check GitHub issues
4. Open a new issue if needed

## License

See the main LiteLLM repository for license information.
