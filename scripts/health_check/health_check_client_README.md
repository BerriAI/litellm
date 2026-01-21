# LiteLLM Health Check Client

A health check tool for testing all configured models on a LiteLLM proxy. Tests each model with completion/embedding requests and reports health status, errors, and response times.

## Features

- **YAML Config Support**: Reads models from YAML config file OR fetches from proxy API
- **Smart Mode Detection**: Detects embedding vs chat models from config or model name
- **Concurrent Testing**: Tests all models concurrently using asyncio
- **Containerized**: Docker image for easy deployment
- **Parallel Execution**: Supports parallel execution for stress testing
- **Configurable**: Customizable timeouts (default 120s) and test prompts

## Quick Start

### As a Python Script

**Option 1: Fetch models from proxy API**
```bash
export LITELLM_BASE_URL="https://litellm.example.com"
export LITELLM_API_KEY="your-api-key"
python scripts/health_check/health_check_client.py
```

**Option 2: Use YAML config file**
```bash
export LITELLM_BASE_URL="https://litellm.example.com"
export LITELLM_API_KEY="your-api-key"
export LITELLM_MODELS_YAML="/path/to/config.yaml"
python scripts/health_check/health_check_client.py
```

**Option 3: Use custom authentication header**
```bash
export LITELLM_BASE_URL="https://litellm.example.com"
export LITELLM_API_KEY="your-api-key"
export LITELLM_CUSTOM_AUTH_HEADER="x-custom-auth-header"
python scripts/health_check/health_check_client.py
```

### As a Docker Container

1. Build the Docker image:

```bash
docker build -f docker/Dockerfile.health_check -t litellm/litellm-health-check:latest .
```

2. Run a single health check:

```bash
docker run --rm \
  -e LITELLM_BASE_URL="https://litellm.example.com" \
  -e LITELLM_API_KEY="your-api-key" \
  litellm/litellm-health-check:latest
```

3. Run with custom authentication header:

```bash
docker run --rm \
  -e LITELLM_BASE_URL="https://litellm.example.com" \
  -e LITELLM_API_KEY="your-api-key" \
  -e LITELLM_CUSTOM_AUTH_HEADER="x-custom-auth-header" \
  litellm/litellm-health-check:latest
```

### Parallel Execution (Stress Testing)

Run multiple health check containers in parallel:

**PowerShell:**
```powershell
$env:LITELLM_BASE_URL="https://litellm.example.com"
$env:LITELLM_API_KEY="your-api-key"
.\scripts\health_check\run_parallel_health_checks.ps1 16
```

**Bash/Shell:**
```bash
export LITELLM_BASE_URL="https://litellm.example.com"
export LITELLM_API_KEY="your-api-key"
./scripts/health_check/run_parallel_health_checks.sh 16
```

**With Custom Auth Header:**
```powershell
$env:LITELLM_BASE_URL="https://litellm.example.com"
$env:LITELLM_API_KEY="your-api-key"
$env:LITELLM_CUSTOM_AUTH_HEADER="x-custom-auth-header"
.\scripts\health_check\run_parallel_health_checks.ps1 16
```

**With Custom Docker Image:**
```powershell
$env:LITELLM_BASE_URL="https://litellm.example.com"
$env:LITELLM_API_KEY="your-api-key"
$env:LITELLM_CUSTOM_AUTH_HEADER="x-custom-auth-header"
.\scripts\health_check\run_parallel_health_checks.ps1 -NumParallelJobs 16 -ImageName "your-registry/your-image:tag"
```

**Bash with Custom Image:**
```bash
export LITELLM_BASE_URL="https://litellm.example.com"
export LITELLM_API_KEY="your-api-key"
export LITELLM_CUSTOM_AUTH_HEADER="x-custom-auth-header"
./scripts/health_check/run_parallel_health_checks.sh 16 "your-registry/your-image:tag"
```


## Configuration

### Environment Variables

- `LITELLM_BASE_URL` (required): Base URL of the LiteLLM proxy
  - Example: `https://litellm.example.com`
- `LITELLM_API_KEY` (required): API key for authentication
- `LITELLM_CUSTOM_AUTH_HEADER` (optional): Custom header name for authentication
  - Use this when your LiteLLM proxy uses a custom authentication header instead of the standard `Authorization` header
  - Example: `x-custom-auth-header` (the API key will be sent as `Bearer <api_key>` in this header)
- `LITELLM_MODELS_YAML` (optional): Path to YAML config file with model_list
  - If provided, reads models from YAML instead of fetching from API
  - Example: `/path/to/config.yaml`
- `LITELLM_TIMEOUT` (optional): Request timeout in seconds (default: 120)
- `LITELLM_COMPLETION_PROMPT` (optional): Test prompt for chat/completion models (default: ~100k characters)
- `LITELLM_EMBEDDING_TEXT` (optional): Test text for embedding models (default: ~100k characters)
- `LITELLM_JSON_OUTPUT` (optional): Output results as JSON (default: false)

### Parallel Script Parameters

**PowerShell (`run_parallel_health_checks.ps1`):**
- `-NumParallelJobs` (optional): Number of parallel containers to run (default: 16)
- `-ImageName` (optional): Docker image to use (default: `litellm/litellm-health-check:latest`)
- `-ContainerRuntime` (optional): Container runtime to use (default: `docker`)

**Bash (`run_parallel_health_checks.sh`):**
- `[num_parallel_jobs]` (optional): Number of parallel containers to run (default: 16)
- `[image_name]` (optional): Docker image to use (default: `litellm/litellm-health-check:latest`)
- `[container_runtime]` (optional): Container runtime to use (default: `docker`)

## Output

### Standard Output (Human-Readable)

Example output format:

```
============================================================
Starting health check queries

---- gpt-4o ----
✅ Success. Response:
This is a test

---- text-embedding-3-small ----
✅ Success. Generated embedding vector with 1536 dimensions.

---- gpt-5-codex ----
❌ ERROR: HTTP 503: Service unavailable

============================================================
Health Check Summary
============================================================
Total models: 47
Healthy: 45
Unhealthy: 2
============================================================
```

Exit code: `0` if all models are healthy, `1` if any models are unhealthy.

### JSON Output

When `LITELLM_JSON_OUTPUT=true`, outputs JSON:

```json
{
  "gpt-4o": {
    "model": "gpt-4o",
    "healthy": true,
    "error": null,
    "response_time_ms": 245.67,
    "mode": "chat",
    "response_text": "This is a test"
  },
  "text-embedding-3-small": {
    "model": "text-embedding-3-small",
    "healthy": true,
    "error": null,
    "response_time_ms": 123.45,
    "mode": "embedding",
    "dimensions": 1536
  }
}
```

## How It Works

1. **Model Discovery**: 
   - If `LITELLM_MODELS_YAML` is set: Reads models from YAML config file
   - Otherwise: Queries `/v1/models` (OpenAI-compatible) or `/model/info` to get all configured models
2. **Mode Detection**: 
   - Checks `mode` field from YAML config, or falls back to model name patterns (embedding, embed, text-embedding)
3. **Concurrent Testing**: 
   - Chat models: `POST /v1/chat/completions` with configurable prompt (default: "Say this is a test")
   - Embedding models: `POST /v1/embeddings` with configurable text (default: "This is a test for vectorization.")
4. **Reporting**: Health status, errors, response times, and response details are reported

## Use Cases

### 1. Regular Health Monitoring

Run as a cron job or scheduled task:

```bash
# Cron job: Run every 5 minutes
*/5 * * * * /path/to/health_check.sh
```

### 2. Load/Stress Testing

Run multiple health checks in parallel:

**PowerShell:**
```powershell
# Using default image
.\scripts\health_check\run_parallel_health_checks.ps1 16

# Using custom image
.\scripts\health_check\run_parallel_health_checks.ps1 -NumParallelJobs 16 -ImageName "your-registry/your-image:tag"
```

**Bash:**
```bash
# Using default image
./scripts/health_check/run_parallel_health_checks.sh 16

# Using custom image
./scripts/health_check/run_parallel_health_checks.sh 16 "your-registry/your-image:tag"
```

### 3. CI/CD Integration

Add to your deployment pipeline:

```yaml
# GitHub Actions example
- name: Health Check
  run: |
    docker run --rm \
      -e LITELLM_BASE_URL="${{ secrets.LITELLM_BASE_URL }}" \
      -e LITELLM_API_KEY="${{ secrets.LITELLM_API_KEY }}" \
      litellm/litellm-health-check:latest
```

### 4. Kubernetes Deployment

Deploy as a CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: litellm-health-check
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: health-check
            image: litellm/litellm-health-check:latest
            env:
            - name: LITELLM_BASE_URL
              value: "https://litellm.example.com"
            - name: LITELLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: litellm-secrets
                  key: api-key
          restartPolicy: OnFailure
```

## Troubleshooting

### No Models Found

- Verify `LITELLM_BASE_URL` is correct
- Check that the API key has permissions to list models
- Ensure the proxy is running and accessible
- If using YAML, verify `LITELLM_MODELS_YAML` path is correct

### Timeout Errors

- Increase `LITELLM_TIMEOUT` for slower models (default is 120s)
- Check network connectivity to the proxy
- Verify proxy isn't overloaded

### Authentication Errors

- Verify `LITELLM_API_KEY` is correct
- Check API key has not expired
- Ensure the key has necessary permissions

## Dependencies

- Python 3.11+
- httpx (for async HTTP requests)
- pyyaml (for YAML config file support)
- Docker or Podman (for containerized execution)
- PowerShell (for parallel execution script on Windows)

## License

Same as LiteLLM project.
