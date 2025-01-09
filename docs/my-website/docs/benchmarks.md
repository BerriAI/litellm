
import Image from '@theme/IdealImage';

# Benchmarks

Benchmarks for LiteLLM Gateway (Proxy Server) tested against a fake OpenAI endpoint.

Use this config for testing:

**Note:**  we're currently migrating to aiohttp which has 10x higher throughput. We recommend using the `aiohttp_openai/` provider for load testing.

```yaml
model_list:
  - model_name: "fake-openai-endpoint"
    litellm_params:
      model: aiohttp_openai/any
      api_base: https://your-fake-openai-endpoint.com/chat/completions
      api_key: "test"
```

## 1 Instance LiteLLM Proxy


| Metric | Litellm Proxy (1 Instance) |
|--------|------------------------|
| Median Latency (ms) | 110 |
| RPS | 250 |

<!-- <Image img={require('../img/1_instance_proxy.png')} /> -->

<!-- ## **Horizontal Scaling - 10K RPS**

<Image img={require('../img/instances_vs_rps.png')} /> -->

#### Key Findings
- Single instance: 250 RPS @ 100ms latency
- 4 LiteLLM instances: 1000 RPS @ 100ms latency

### 2 Instances

**Adding 1 instance, will double the RPS and maintain the `100ms-110ms` median latency.**

| Metric | Litellm Proxy (2 Instances) |
|--------|------------------------|
| Median Latency (ms) | 100 |
| RPS | 500 |




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



## Locust Settings

- 2500 Users
- 100 user Ramp Up
