# LLMonitor Tutorial

[LLMonitor](https://llmonitor.com/) is an open-source observability platform that provides cost tracking, user tracking and powerful agent tracing.

<video controls width='900' >
  <source src='https://llmonitor.com/videos/demo-annotated.mp4'/>
</video>

## Use LLMonitor to log requests across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)

liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

:::info
We want to learn how we can make the callbacks better! Meet the [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
::: 

### Using Callbacks

First, sign up to get an app ID on the [LLMonitor dashboard](https://llmonitor.com).

Use just 2 lines of code, to instantly log your responses **across all providers** with llmonitor:

```python
litellm.success_callback = ["llmonitor"]
litellm.failure_callback = ["llmonitor"]
```

Complete code

```python
from litellm import completion

## set env variables
os.environ["LLMONITOR_APP_ID"] = "your-llmonitor-app-id"
# Optional: os.environ["LLMONITOR_API_URL"] = "self-hosting-url"

os.environ["OPENAI_API_KEY"], os.environ["COHERE_API_KEY"] = "", ""

# set callbacks
litellm.success_callback = ["llmonitor"]
litellm.failure_callback = ["llmonitor"]

#openai call
response = completion(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
  user="ishaan_litellm"
)

#cohere call
response = completion(
  model="command-nightly", 
  messages=[{"role": "user", "content": "Hi 👋 - i'm cohere"}],
  user="ishaan_litellm"
)
```

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
- Meet the LLMonitor team on [Discord](http://discord.com/invite/8PafSG58kK) or via [email](mailto:vince@llmonitor.com).