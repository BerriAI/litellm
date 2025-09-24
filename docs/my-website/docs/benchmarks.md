
import Image from '@theme/IdealImage';

# Benchmarks

Benchmarks for LiteLLM Gateway (Proxy Server) tested against a fake OpenAI endpoint.

Use this config for testing:

```yaml
model_list:
  - model_name: "fake-openai-endpoint"
    litellm_params:
      model: openai/any
      api_base: https://your-fake-openai-endpoint.com/chat/completions
      api_key: "test"
```

### 1 Instance LiteLLM Proxy

In these tests the baseline latency characteristics are measured against a fake-openai-endpoint.

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Requests per Second (RPS)** | 475 |
| **End-to-End Latency P50 (ms)** | 100 |
| **LiteLLM Overhead P50 (ms)** | 3 |
| **LiteLLM Overhead P90 (ms)** | 17 |
| **LiteLLM Overhead P99 (ms)** | 31 |

<!-- <Image img={require('../img/1_instance_proxy.png')} /> -->

<!-- ## **Horizontal Scaling - 10K RPS**

<Image img={require('../img/instances_vs_rps.png')} /> -->

#### Key Findings
- Single instance: 475 RPS @ 100ms median latency
- LiteLLM adds 3ms P50 overhead, 17ms P90 overhead, 31ms P99 overhead
- 2 LiteLLM instances: 950 RPS @ 100ms latency
- 4 LiteLLM instances: 1900 RPS @ 100ms latency

### 2 Instances

**Adding 1 instance, will double the RPS and maintain the `100ms-110ms` median latency.**

| Metric | Litellm Proxy (2 Instances) |
|--------|------------------------|
| Median Latency (ms) | 100 |
| RPS | 950 |


## Machine Spec used for testing

Each machine deploying LiteLLM had the following specs:

- 2 CPU
- 4GB RAM

## How to measure LiteLLM Overhead

All responses from litellm will include the `x-litellm-overhead-duration-ms` header, this is the latency overhead in milliseconds added by LiteLLM Proxy.


If you want to measure this on locust you can use the following code:

```python showLineNumbers title="Locust Code for measuring LiteLLM Overhead"
import os
import uuid
from locust import HttpUser, task, between, events

# Custom metric to track LiteLLM overhead duration
overhead_durations = []

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    if response and hasattr(response, 'headers'):
        overhead_duration = response.headers.get('x-litellm-overhead-duration-ms')
        if overhead_duration:
            try:
                duration_ms = float(overhead_duration)
                overhead_durations.append(duration_ms)
                # Report as custom metric
                events.request.fire(
                    request_type="Custom",
                    name="LiteLLM Overhead Duration (ms)",
                    response_time=duration_ms,
                    response_length=0,
                )
            except (ValueError, TypeError):
                pass

class MyUser(HttpUser):
    wait_time = between(0.5, 1)  # Random wait time between requests

    def on_start(self):
        self.api_key = os.getenv('API_KEY', 'sk-1234567890')
        self.client.headers.update({'Authorization': f'Bearer {self.api_key}'})

    @task
    def litellm_completion(self):
        # no cache hits with this
        payload = {
            "model": "db-openai-endpoint",
            "messages": [{"role": "user", "content": f"{uuid.uuid4()} This is a test there will be no cache hits and we'll fill up the context" * 150}],
            "user": "my-new-end-user-1"
        }
        response = self.client.post("chat/completions", json=payload)
        
        if response.status_code != 200:
            # log the errors in error.txt
            with open("error.txt", "a") as error_log:
                error_log.write(response.text + "\n")
```



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
