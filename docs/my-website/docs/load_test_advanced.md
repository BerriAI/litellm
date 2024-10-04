# LiteLLM Proxy - 1K RPS Load test on locust 

Tutorial on how to get to 1K+ RPS with LiteLLM Proxy on locust

## Expected Performance

| Metric | Value |
|--------|-------|
| Requests per Second | 1174+ |
| Median Response Time | `96ms` |
| Average Response Time | `142.18ms` |



## Pre-Testing Checklist
- [ ] Ensure you're using the **latest `-stable` version** of litellm
    - [Github releases](https://github.com/BerriAI/litellm/releases)
    - [litellm docker containers](https://github.com/BerriAI/litellm/pkgs/container/litellm)
    - [litellm database docker container](https://github.com/BerriAI/litellm/pkgs/container/litellm-database)
- [ ] Ensure you're following **ALL** [best practices for production](./proxy/production_setup.md)
- [ ] Locust - Ensure you're Locust instance can create 1K+ requests per second



## Running the Load Test - Fake OpenAI Endpoint

1. Add `fake-openai-endpoint` to your proxy config.yaml and start your litellm proxy
litellm provides a hosted `fake-openai-endpoint` you can load test against

```yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
```

2. `pip install locust`

3. Create a file called `locustfile.py` on your local machine. Copy the contents from the litellm load test located [here](https://github.com/BerriAI/litellm/blob/main/.github/workflows/locustfile.py)

4. Start locust
  Run `locust` in the same directory as your `locustfile.py` from step 2

  ```shell
  locust
  ```

  Output on terminal 
  ```
  [2024-03-15 07:19:58,893] Starting web interface at http://0.0.0.0:8089
  [2024-03-15 07:19:58,898] Starting Locust 2.24.0
  ```

5. Run Load test on locust

  Head to the locust UI on http://0.0.0.0:8089

  Set **Users=1000, Ramp Up Users=500**, Host=Base URL of your LiteLLM Proxy

## Running the Load test - Endpoints with Rate Limits 



## Machine Specifications for Running Locust

| Metric | Value |
|--------|-------|
| `locust --processes 4`  | 4|
| `vCPUs` on Load Testing Machine | 2.0 vCPUs |
| `Memory` on Load Testing Machine | 450 MB |
| `Replicas` of Load Testing Machine | 1 |

## Machine Specifications for Running LiteLLM Proxy

**Number of Replicas of LiteLLM Proxy=20**

| Service | Spec | CPUs | Memory | Architecture | Version|
| --- | --- | --- | --- | --- | --- | 
| Server | `t2.large`. | `2vCPUs` | `8GB` | `x86` |

    


