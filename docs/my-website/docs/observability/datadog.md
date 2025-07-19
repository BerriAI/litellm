import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# DataDog

LiteLLM Supports logging to the following Datdog Integrations:
- `datadog` [Datadog Logs](https://docs.datadoghq.com/logs/)
- `datadog_llm_observability` [Datadog LLM Observability](https://www.datadoghq.com/product/llm-observability/)
- `ddtrace-run` [Datadog Tracing](#datadog-tracing)

<Tabs>
<TabItem value="datadog" label="Datadog Logs">

We will use the `--config` to set `litellm.callbacks = ["datadog"]` this will log all successful LLM calls to DataDog

**Step 1**: Create a `config.yaml` file and set `litellm_settings`: `success_callback`

```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  callbacks: ["datadog"] # logs llm success + failure logs on datadog
  service_callback: ["datadog"] # logs redis, postgres failures on datadog
```

</TabItem>
<TabItem value="datadog_llm_observability" label="Datadog LLM Observability">

```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  callbacks: ["datadog_llm_observability"] # logs llm success logs on datadog
```

</TabItem>
</Tabs>

**Step 2**: Set Required env variables for datadog

```shell
DD_API_KEY="5f2d0f310***********" # your datadog API Key
DD_SITE="us5.datadoghq.com"       # your datadog base url
DD_SOURCE="litellm_dev"       # [OPTIONAL] your datadog source. use to differentiate dev vs. prod deployments
```

**Step 3**: Start the proxy, make a test request

Start proxy

```shell
litellm --config config.yaml --debug
```

Test Request

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "metadata": {
        "your-custom-metadata": "custom-field",
    }
}'
```

Expected output on Datadog

<Image img={require('../../img/dd_small1.png')} />

#### Datadog Tracing

Use `ddtrace-run` to enable [Datadog Tracing](https://ddtrace.readthedocs.io/en/stable/installation_quickstart.html) on litellm proxy

**DD Tracer**
Pass `USE_DDTRACE=true` to the docker run command. When `USE_DDTRACE=true`, the proxy will run `ddtrace-run litellm` as the `ENTRYPOINT` instead of just `litellm`

**DD Profiler**

Pass `USE_DDPROFILER=true` to the docker run command. When `USE_DDPROFILER=true`, the proxy will activate the [Datadog Profiler](https://docs.datadoghq.com/profiler/enabling/python/). This is useful for debugging CPU% and memory usage.

We don't recommend using `USE_DDPROFILER` in production. It is only recommended for debugging CPU% and memory usage.


```bash
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -e USE_DDTRACE=true \
    -e USE_DDPROFILER=true \
    -p 4000:4000 \
    ghcr.io/berriai/litellm:main-latest \
    --config /app/config.yaml --detailed_debug
```

### Set DD variables (`DD_SERVICE` etc)

LiteLLM supports customizing the following Datadog environment variables

| Environment Variable | Description | Default Value | Required |
|---------------------|-------------|---------------|----------|
| `DD_API_KEY` | Your Datadog API key for authentication | None | ✅ Yes |
| `DD_SITE` | Your Datadog site (e.g., "us5.datadoghq.com") | None | ✅ Yes |
| `DD_ENV` | Environment tag for your logs (e.g., "production", "staging") | "unknown" | ❌ No |
| `DD_SERVICE` | Service name for your logs | "litellm-server" | ❌ No |
| `DD_SOURCE` | Source name for your logs | "litellm" | ❌ No |
| `DD_VERSION` | Version tag for your logs | "unknown" | ❌ No |
| `HOSTNAME` | Hostname tag for your logs | "" | ❌ No |
| `POD_NAME` | Pod name tag (useful for Kubernetes deployments) | "unknown" | ❌ No |

