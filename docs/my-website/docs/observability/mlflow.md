# MLflow

## What is MLflow?

**MLflow** is an end-to-end open source MLOps platform for [experiment tracking](https://www.mlflow.org/docs/latest/tracking.html), [model management](https://www.mlflow.org/docs/latest/models.html), [evaluation](https://www.mlflow.org/docs/latest/llms/llm-evaluate/index.html), [observability (tracing)](https://www.mlflow.org/docs/latest/llms/tracing/index.html), and [deployment](https://www.mlflow.org/docs/latest/deployment/index.html). MLflow empowers teams to collaboratively develop and refine LLM applications efficiently.

MLflow’s integration with LiteLLM supports advanced observability compatible with OpenTelemetry.


<Image img={require('../../img/mlflow_tracing.png')} />


## Getting Started

Install MLflow:

```shell
pip install mlflow
```

To enable LiteLLM tracing:

```python
import mlflow

mlflow.litellm.autolog()

# Alternative, you can set the callback manually in LiteLLM
# litellm.callbacks = ["mlflow"]
```

Since MLflow is open-source, no sign-up or API key is needed to log traces!

```
import litellm
import os

# Set your LLM provider's API key
os.environ["OPENAI_API_KEY"] = ""

# Call LiteLLM as usual
response = litellm.completion(
    model="gpt-4o-mini",
    messages=[
      {"role": "user", "content": "Hi 👋 - i'm openai"}
    ]
)
```

Open the MLflow UI and go to the `Traces` tab to view logged traces:

```bash
mlflow ui
```

## Exporting Traces to OpenTelemetry collectors

MLflow traces are compatible with OpenTelemetry. You can export traces to any OpenTelemetry collector (e.g., Jaeger, Zipkin, Datadog, New Relic) by setting the endpoint URL in the environment variables.

```
# Set the endpoint of the OpenTelemetry Collector
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4317/v1/traces"
# Optionally, set the service name to group traces
os.environ["OTEL_SERVICE_NAME"] = "<your-service-name>"
```

See [MLflow documentation](https://mlflow.org/docs/latest/llms/tracing/index.html#using-opentelemetry-collector-for-exporting-traces) for more details.

## Combine LiteLLM Trace with Your Application Trace

LiteLLM is often part of larger LLM applications, such as agentic models. MLflow Tracing allows you to instrument custom Python code, which can then be combined with LiteLLM traces.

```python
import litellm
import mlflow
from mlflow.entities import SpanType

# Enable LiteLLM tracing
mlflow.litellm.autolog()


class CustomAgent:
    # Use @mlflow.trace to instrument Python functions.
    @mlflow.trace(span_type=SpanType.AGENT)
    def run(self, query: str):
        # do something

        while i < self.max_turns:
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=messages,
            )

            action = self.get_action(response)
            ...

    @mlflow.trace
    def get_action(llm_response):
        ...
```

This approach generates a unified trace, combining your custom Python code with LiteLLM calls.


## Support

* For advanced usage and integrations of tracing, visit the [MLflow Tracing documentation](https://mlflow.org/docs/latest/llms/tracing/index.html).
* For any question or issue with this integration, please [submit an issue](https://github.com/mlflow/mlflow/issues/new/choose) on our [Github](https://github.com/mlflow/mlflow) repository!