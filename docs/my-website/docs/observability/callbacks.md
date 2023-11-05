# Callbacks

## Use Callbacks to send Output Data to Posthog, Sentry etc

liteLLM provides `input_callbacks`, `success_callbacks` and `failure_callbacks`, making it easy for you to send data to a particular provider depending on the status of your responses.

liteLLM supports:

- [Custom Callback Functions](https://docs.litellm.ai/docs/observability/custom_callback)
- [LLMonitor](https://llmonitor.com/docs)
- [Helicone](https://docs.helicone.ai/introduction)
- [Traceloop](https://traceloop.com/docs)
- [Sentry](https://docs.sentry.io/platforms/python/)
- [PostHog](https://posthog.com/docs/libraries/python)
- [Slack](https://slack.dev/bolt-python/concepts)

### Quick Start

```python
from litellm import completion

# set callbacks
litellm.input_callback=["sentry"] # for sentry breadcrumbing - logs the input being sent to the api
litellm.success_callback=["posthog", "helicone", "llmonitor"]
litellm.failure_callback=["sentry", "llmonitor"]

## set env variables
os.environ['SENTRY_DSN'], os.environ['SENTRY_API_TRACE_RATE']= ""
os.environ['POSTHOG_API_KEY'], os.environ['POSTHOG_API_URL'] = "api-key", "api-url"
os.environ["HELICONE_API_KEY"] = ""
os.environ["TRACELOOP_API_KEY"] = ""
os.environ["LLMONITOR_APP_ID"] = ""

response = completion(model="gpt-3.5-turbo", messages=messages)
```
