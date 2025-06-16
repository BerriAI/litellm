import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Elasticsearch Logging with LiteLLM

Send your LLM requests, responses, costs, and performance data to Elasticsearch for analytics and monitoring.

## Quick Start

### 1. Start Elasticsearch

```bash
# Using Docker (simplest)
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

### 2. Configure LiteLLM

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

Create a `config.yaml` file:

```yaml
model_list:
  - model_name: gpt-4.1
    litellm_params:
      model: openai/gpt-4.1
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  success_callback: ["generic"]
  failure_callback: ["generic"]

general_settings:
  generic_logger_endpoint: "http://localhost:9200/litellm-logs/_doc"
  generic_logger_headers: 
    "Content-Type": "application/json"
```

Start the proxy:
```bash
litellm --config config.yaml
```

</TabItem>
<TabItem value="python-sdk" label="Python SDK">

Configure the generic logger in your Python code:

```python
import litellm
import os

# Set up Elasticsearch endpoint
os.environ["GENERIC_LOGGER_ENDPOINT"] = "http://localhost:9200/litellm-logs/_doc"
os.environ["GENERIC_LOGGER_HEADERS"] = "Content-Type=application/json"

# Enable logging
litellm.success_callback = ["generic"]
litellm.failure_callback = ["generic"]

# Make your LLM calls
response = litellm.completion(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

</TabItem>
</Tabs>

### 3. Test the Integration

Make a test request to verify logging is working:

<Tabs>
<TabItem value="curl-proxy" label="Test Proxy">

```bash
curl -X POST "http://localhost:4000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "Hello from LiteLLM!"}]
  }'
```

</TabItem>
<TabItem value="python-test" label="Test Python SDK">

```python
import litellm

response = litellm.completion(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello from LiteLLM!"}],
    user="test-user"
)
print("Response:", response.choices[0].message.content)
```

</TabItem>
</Tabs>

### 4. Verify It's Working

```bash
# Check if logs are being created
curl "localhost:9200/litellm-logs/_search?pretty&size=1"
```

You should see your LLM requests with fields like `model`, `response_cost`, `total_tokens`, `messages`, etc.

## Analytics Examples

**Total costs by model:**
```bash
curl -X GET "localhost:9200/litellm-logs/_search" -H "Content-Type: application/json" -d '{
  "size": 0,
  "aggs": {
    "models": {
      "terms": {"field": "model"},
      "aggs": {"total_cost": {"sum": {"field": "response_cost"}}}
    }
  }
}'
```

**Average response time:**
```bash
curl -X GET "localhost:9200/litellm-logs/_search" -H "Content-Type: application/json" -d '{
  "size": 0,
  "aggs": {"avg_response_time": {"avg": {"field": "response_time"}}}
}'
```

**Recent errors:**
```bash
curl -X GET "localhost:9200/litellm-logs/_search" -H "Content-Type: application/json" -d '{
  "query": {"term": {"status": "failure"}},
  "size": 10,
  "sort": [{"endTime": {"order": "desc"}}]
}'
```

## Production Setup

**With Elasticsearch Cloud:**
```yaml
general_settings:
  generic_logger_endpoint: "https://your-deployment.es.region.cloud.es.io/litellm-logs/_doc"
  generic_logger_headers:
    "Content-Type": "application/json"
    "Authorization": "Bearer your-api-key"
```

**Docker Compose (Full Stack):**
```yaml
# docker-compose.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
      
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GENERIC_LOGGER_ENDPOINT=http://elasticsearch:9200/litellm-logs/_doc
    command: ["--config", "/app/config.yaml"]
    volumes:
      - ./config.yaml:/app/config.yaml
```

**config.yaml:**
```yaml
model_list:
  - model_name: gpt-4.1
    litellm_params:
      model: openai/gpt-4.1
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  success_callback: ["generic"]
  failure_callback: ["generic"]

general_settings:
  master_key: sk-1234
```

## What's Logged

LiteLLM sends a payload for every request including:

- `model` - Model used (e.g., gpt-4.1)
- `response_cost` - Cost in USD
- `total_tokens`, `prompt_tokens`, `completion_tokens` - Token usage
- `response_time` - How long the request took
- `status` - "success" or "failure"
- `messages` - Input messages
- `response` - LLM response
- `metadata` - User info, API keys, etc.

See the full [StandardLoggingPayload specification](../proxy/logging_spec) for all available fields.