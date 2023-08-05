# BerriSpend Tutorial 
BerriSpend is a free dashboard to monitor your cost and logs across llm providers. 

## Use BerriSpend to see total spend across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)
liteLLM provides `success_callbacks` and `failure_callbacks`, making it easy for you to send data to a particular provider depending on the status of your responses. 

In this case, we want to log requests to BerriSpend when a request succeeds. 

### Use Callbacks 
Use just 2 lines of code, to instantly see costs and log your responses **across all providers** with BerriSpend: 

```
litellm.success_callback=["berrispend"]
litellm.failure_callback=["berrispend"]
```

Complete code
```python
from litellm import completion

## set env variables
os.environ["BERRISPEND_ACCOUNT_ID"] = "your-email-id" 
os.environ["OPENAI_API_KEY"] = ""

# set callbacks
litellm.success_callback=["berrispend"]
litellm.failure_callback=["berrispend"]

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}]) 

#bad call
response = completion(model="chatgpt-test", messages=[{"role": "user", "content": "Hi 👋 - i'm a bad call to test error logging"}]) 
```

Then go to https://litellm-ui.vercel.app/<your_email_id> to view your logs and cost 😊