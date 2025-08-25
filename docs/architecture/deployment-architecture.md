# Deployment Architecture

## Deployment Strategy

**Frontend Deployment:**
- **Platform:** Static hosting (Vercel/Netlify) or containerized
- **Build Command:** `cd ui && npm run build`
- **Output Directory:** `ui/dist`
- **CDN/Edge:** CloudFlare or cloud provider CDN

**Backend Deployment:**
- **Platform:** Kubernetes, Docker Swarm, or cloud PaaS
- **Build Command:** `docker build -f deploy/docker/Dockerfile .`
- **Deployment Method:** Rolling update with health checks

## CI/CD Pipeline
```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install --with dev
      - name: Run tests
        run: poetry run pytest tests/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t litellm:${{ github.sha }} .
      - name: Push to registry
        run: |
          docker tag litellm:${{ github.sha }} ${{ secrets.REGISTRY }}/litellm:latest
          docker push ${{ secrets.REGISTRY }}/litellm:latest
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/litellm-proxy \
            litellm=${{ secrets.REGISTRY }}/litellm:latest \
            --namespace=production
```

## Environments

| Environment | Frontend URL | Backend URL | Purpose |
|------------|--------------|-------------|---------|
| Development | http://localhost:3000 | http://localhost:4000 | Local development |
| Staging | https://staging.litellm.ai | https://api-staging.litellm.ai | Pre-production testing |
| Production | https://app.litellm.ai | https://api.litellm.ai | Live environment |
