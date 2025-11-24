
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

### 2 Instance LiteLLM Proxy

In these tests the baseline latency characteristics are measured against a fake-openai-endpoint.

#### Performance Metrics

| **Type** | **Name** | **Median (ms)** | **95%ile (ms)** | **99%ile (ms)** | **Average (ms)** | **Current RPS** |
| --- | --- | --- | --- | --- | --- | --- |
| POST | /chat/completions | 200 | 630 | 1200 | 262.46 | 1035.7 |
| Custom | LiteLLM Overhead Duration (ms) | 12 | 29 | 43 | 14.74 | 1035.7 |
|  | Aggregated | 100 | 430 | 930 | 138.6 | 2071.4 |

<!-- <Image img={require('../img/1_instance_proxy.png')} /> -->

<!-- ## **Horizontal Scaling - 10K RPS**

<Image img={require('../img/instances_vs_rps.png')} /> -->


### 4 Instances

| **Type** | **Name** | **Median (ms)** | **95%ile (ms)** | **99%ile (ms)** | **Average (ms)** | **Current RPS** |
| --- | --- | --- | --- | --- | --- | --- |
| POST | /chat/completions | 100 | 150 | 240 | 111.73 | 1170 |
| Custom | LiteLLM Overhead Duration (ms) | 2 | 8 | 13 | 3.32 | 1170 |
|  | Aggregated | 77 | 130 | 180 | 57.53 | 2340 |

#### Key Findings
- Doubling from 2 to 4 LiteLLM instances halves median latency: 200 ms → 100 ms.
- High-percentile latencies drop significantly: P95 630 ms → 150 ms, P99 1,200 ms → 240 ms.
- Setting workers equal to CPU count gives optimal performance.

## Machine Spec used for testing

Each machine deploying LiteLLM had the following specs:

- 4 CPU
- 8GB RAM

## Configuration

- Database: PostgreSQL
- Redis: Not used

## Locust Settings

- 1000 Users
- 500 user Ramp Up

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


## LiteLLM vs Portkey Performance Comparison

**Test Configuration**: 4 CPUs, 8 GB RAM per instance | Load: 1k concurrent users, 500 ramp-up
**Versions:** Portkey **v1.14.0** | LiteLLM **v1.79.1-stable**  
**Test Duration:** 5 minutes  

### Multi-Instance (4×) Performance

| Metric              | Portkey (no DB) | LiteLLM (with DB) | Comment        |
| ------------------- | --------------- | ----------------- | -------------- |
| **Total Requests**  | 293,796         | 312,405           | LiteLLM higher |
| **Failed Requests** | 0               | 0                 | Same           |
| **Median Latency**  | 100 ms          | 100 ms            | Same           |
| **p95 Latency**     | 230 ms          | 150 ms            | LiteLLM lower  |
| **p99 Latency**     | 500 ms          | 240 ms            | LiteLLM lower  |
| **Average Latency** | 123 ms          | 111 ms            | LiteLLM lower  |
| **Current RPS**     | 1,170.9         | 1,170             | Same           |


*Lower is better for latency metrics; higher is better for requests and RPS.*

### Technical Insights

**Portkey**

**Pros**

* Low memory footprint
* Stable latency with minimal spikes

**Cons**

* CPU utilization capped around ~40%, indicating underutilization of available compute resources
* Experienced three I/O timeout outages

**LiteLLM**

**Pros**

* Fully utilizes available CPU capacity
* Strong connection handling and low latency after initial warm-up spikes

**Cons**

* High memory usage during initialization and per request



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
