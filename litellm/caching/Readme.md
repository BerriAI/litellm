# Caching on LiteLLM

LiteLLM supports multiple caching mechanisms. This allows users to choose the most suitable caching solution for their use case.

The following caching mechanisms are supported:

1. **RedisCache**
2. **RedisSemanticCache**
3. **QdrantSemanticCache**
4. **InMemoryCache**
5. **DiskCache**
6. **S3Cache**
7. **AzureBlobCache**
8. **DualCache** (updates both Redis and an in-memory cache simultaneously)

## Folder Structure

```
litellm/caching/
├── base_cache.py
├── caching.py
├── caching_handler.py
├── disk_cache.py
├── dual_cache.py
├── in_memory_cache.py
├── qdrant_semantic_cache.py
├── redis_cache.py
├── redis_semantic_cache.py
├── s3_cache.py
```

## Documentation
- [Caching on LiteLLM Gateway](https://docs.litellm.ai/docs/proxy/caching)
- [Caching on LiteLLM Python](https://docs.litellm.ai/docs/caching/all_caches)

## Google Cloud IAM Authentication for Redis

LiteLLM supports Google Cloud IAM authentication for Redis connections, which is useful when connecting to Google Cloud Memorystore for Redis clusters with IAM authentication enabled.

### Prerequisites

Install the required Google Cloud dependency:

```bash
pip install google-cloud-iam
```

### Configuration

#### Environment Variables

Set the following environment variable:

```bash
export REDIS_GCP_IAM_SERVICE_ACCOUNT="projects/-/serviceAccounts/your-service-account@project.iam.gserviceaccount.com"
```

Additional Redis configuration:

```bash
export REDIS_HOST="your-redis-cluster-discovery-endpoint"
export REDIS_PORT="6379"  # Default Redis port
export REDIS_SSL="true"   # Enable SSL
export REDIS_SSL_CA_CERTS="/path/to/server-ca.crt"  # Path to trusted server CA file
```

#### Python Code Configuration

```python
import litellm
from litellm.caching import RedisCache

# Configure Redis cache with Google Cloud IAM authentication
litellm.cache = RedisCache(
    host="your-redis-cluster-discovery-endpoint",
    port=6379,
    gcp_iam_service_account="projects/-/serviceAccounts/your-service-account@project.iam.gserviceaccount.com",
    ssl=True,
    ssl_ca_certs="/path/to/server-ca.crt"
)

# Use LiteLLM with caching
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    caching=True
)
```

#### For Redis Cluster

When using Redis Cluster with IAM authentication:

```python
from litellm.caching import RedisCache

# Redis cluster nodes configuration
startup_nodes = [
    {"host": "node1.redis.cluster", "port": 6379},
    {"host": "node2.redis.cluster", "port": 6379},
    {"host": "node3.redis.cluster", "port": 6379}
]

litellm.cache = RedisCache(
    startup_nodes=startup_nodes,
    gcp_iam_service_account="projects/-/serviceAccounts/your-service-account@project.iam.gserviceaccount.com",
    ssl=True,
    ssl_ca_certs="/path/to/server-ca.crt"
)
```

### How It Works

The Google Cloud IAM authentication:

1. Uses the IAM Credentials API to generate short-lived access tokens
2. Authenticates to Redis using the generated access token
3. Handles token refresh automatically on reconnection
4. Supports both regular Redis and Redis Cluster configurations
5. Works with both sync and async Redis clients

### Service Account Requirements

Ensure your service account has the following permissions:

- `redis.instances.connect` - To connect to the Redis instance
- `iam.serviceAccounts.generateAccessToken` - To generate access tokens (if impersonating)

The service account should be granted the **Redis Client** role or equivalent permissions on the Redis instance.







