# Callbacks

## Use Callbacks to send Output Data to Posthog, Sentry etc

liteLLM provides `success_callbacks` and `failure_callbacks`, making it easy for you to send data to a particular provider depending on the status of your responses.

liteLLM supports:

- [LLMonitor](https://llmonitor.com/docs)
- [Helicone](https://docs.helicone.ai/introduction)
- [Sentry](https://docs.sentry.io/platforms/python/)
- [PostHog](https://posthog.com/docs/libraries/python)
- [Slack](https://slack.dev/bolt-python/concepts)

### Quick Start

```python
from litellm import completion

# set callbacks
litellm.success_callback=["posthog", "helicone", "llmonitor"]
litellm.failure_callback=["sentry", "llmonitor"]

## set env variables
os.environ['SENTRY_DSN'], os.environ['SENTRY_API_TRACE_RATE']= ""
os.environ['POSTHOG_API_KEY'], os.environ['POSTHOG_API_URL'] = "api-key", "api-url"
os.environ["HELICONE_API_KEY"] = ""

response = completion(model="gpt-3.5-turbo", messages=messages)
```
