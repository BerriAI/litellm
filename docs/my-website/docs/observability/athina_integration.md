import Image from '@theme/IdealImage';

# Athina

[Athina](https://athina.ai/) is an evaluation framework and production monitoring platform for your LLM-powered app. Athina is designed to enhance the performance and reliability of AI applications through real-time monitoring, granular analytics, and plug-and-play evaluations.

<Image img={require('../../img/athina_dashboard.png')} />

## Getting Started

Use Athina to log requests across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)

liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

## Using Callbacks

First, sign up to get an API_KEY on the [Athina dashboard](https://app.athina.ai).

Use just 1 line of code, to instantly log your responses **across all providers** with Athina:

```python
litellm.success_callback = ["athina"]
```

### Complete code

```python
from litellm import completion

## set env variables
os.environ["ATHINA_API_KEY"] = "your-athina-api-key"
os.environ["OPENAI_API_KEY"]= ""

# set callback
litellm.success_callback = ["athina"]

#openai call
response = completion(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}]
) 
```

## Support & Talk with Athina Team

- [Schedule Demo 👋](https://cal.com/shiv-athina/30min)
- [Website 💻](https://athina.ai/?utm_source=litellm&utm_medium=website)
- [Docs 📖](https://docs.athina.ai/?utm_source=litellm&utm_medium=website)
- [Demo Video 📺](https://www.loom.com/share/d9ef2c62e91b46769a39c42bb6669834?sid=711df413-0adb-4267-9708-5f29cef929e3)
- Our emails ✉️ shiv@athina.ai, akshat@athina.ai, vivek@athina.ai
