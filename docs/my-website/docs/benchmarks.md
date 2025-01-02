
import Image from '@theme/IdealImage';

# Benchmarks

Benchmarks for LiteLLM Gateway (Proxy Server) tested against a fake OpenAI endpoint.

## 1 Instance LiteLLM Proxy


| Metric | Litellm Proxy (1 Instance) |
|--------|------------------------|
| Median Latency (ms) | 110 |
| RPS | 68.2 |

<Image img={require('../img/1_instance_proxy.png')} />

## **Horizontal Scaling**

- Single instance: 68.2 RPS @ 100ms latency
- 10 instances: 4.3% efficiency loss (653 RPS vs expected 682 RPS), latency stable at `100ms`
- For 10,000 RPS: Need ~153 instances @ 95.7% efficiency, `100ms latency`


### 2 Instances

**Adding 1 instance, will double the RPS and maintain the `100ms-110ms` median latency.**

| Metric | Litellm Proxy (2 Instances) |
|--------|------------------------|
| Median Latency (ms) | 100 |
| RPS | 142 |


<Image img={require('../img/2_instance_proxy.png')} />


### 10 Instances

| Metric | Litellm Proxy (10 Instances) |
|--------|------------------------|
| Median Latency (ms) | 110 |
| RPS | 653 |

<Image img={require('../img/10_instance_proxy.png')} />


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
