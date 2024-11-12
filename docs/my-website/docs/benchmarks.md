# Benchmarks

Benchmarks for LiteLLM Gateway (Proxy Server)

Locust Settings:
- 2500 Users
- 100 user Ramp Up


## Basic Benchmarks

Overhead when using a Deployed Proxy vs Direct to LLM
- Latency overhead added by LiteLLM Proxy: 107ms

| Metric | Direct to Fake Endpoint | Basic Litellm Proxy |
|--------|------------------------|---------------------|
| RPS | 1196 | 1133.2 |
| Median Latency (ms) | 33 | 140 |


## Logging Callbacks

### [GCS Bucket Logging](https://docs.litellm.ai/docs/proxy/bucket)

Using GCS Bucket has **no impact on latency, RPS compared to Basic Litellm Proxy**

| Metric | Basic Litellm Proxy | LiteLLM Proxy with GCS Bucket Logging |
|--------|------------------------|---------------------|
| RPS | 1133.2 | 1137.3 |
| Median Latency (ms) | 140 | 138 |


### [LangSmith logging](https://docs.litellm.ai/docs/proxy/logging)

Using LangSmith has **no impact on latency, RPS compared to Basic Litellm Proxy**

| Metric | Basic Litellm Proxy | LiteLLM Proxy with LangSmith |
|--------|------------------------|---------------------|
| RPS | 1133.2 | 1135 |
| Median Latency (ms) | 140 | 132 |

