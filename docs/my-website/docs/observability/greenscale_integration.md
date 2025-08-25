# Greenscale - Track LLM Spend and Responsible Usage


:::tip

This is community maintained, Please make an issue if you run into a bug
https://github.com/BerriAI/litellm

:::


[Greenscale](https://greenscale.ai/) is a production monitoring platform for your LLM-powered app that provides you granular key insights into your GenAI spending and responsible usage. Greenscale only captures metadata to minimize the exposure risk of personally identifiable information (PII).

## Getting Started

Use Greenscale to log requests across all LLM Providers

liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

## Using Callbacks

First, email `hello@greenscale.ai` to get an API_KEY.

Use just 1 line of code, to instantly log your responses **across all providers** with Greenscale:

```python
litellm.success_callback = ["greenscale"]
```

### Complete code

```python
from litellm import completion

## set env variables
os.environ['GREENSCALE_API_KEY'] = 'your-greenscale-api-key'
os.environ['GREENSCALE_ENDPOINT'] = 'greenscale-endpoint'
os.environ["OPENAI_API_KEY"]= ""

# set callback
litellm.success_callback = ["greenscale"]

#openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}]
  metadata={
    "greenscale_project": "acme-project",
    "greenscale_application": "acme-application"
  }
)
```

## Additional information in metadata

You can send any additional information to Greenscale by using the `metadata` field in completion and `greenscale_` prefix. This can be useful for sending metadata about the request, such as the project and application name, customer_id, environment, or any other information you want to track usage. `greenscale_project` and `greenscale_application` are required fields.

```python
#openai call with additional metadata
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  metadata={
    "greenscale_project": "acme-project",
    "greenscale_application": "acme-application",
    "greenscale_customer_id": "customer-123"
  }
)
```

## Support & Talk with Greenscale Team

- [Schedule Demo 👋](https://calendly.com/nandesh/greenscale)
- [Website 💻](https://greenscale.ai)
- Our email ✉️ `hello@greenscale.ai`
