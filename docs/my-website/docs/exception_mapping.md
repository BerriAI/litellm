# Exception Mapping

LiteLLM maps the 3 most common exceptions across all providers. 
- Rate Limit Errors
- Context Window Errors
- InvalidAuth errors (key rotation stuff)

Base case - we return the original exception.

For all 3 cases, the exception returned inherits from the original OpenAI Exception but contains 3 additional attributes: 
* status_code - the http status code of the exception
* message - the error message
* llm_provider - the provider raising the exception

## usage

```python 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "bad-key"
try: 
    # some code 
    completion(model="claude-instant-1", messages=[{"role": "user", "content": "Hey, how's it going?"}])
except Exception as e:
    print(e.llm_provider)
```

## details 

To see how it's implemented - [check out the code](https://github.com/BerriAI/litellm/blob/a42c197e5a6de56ea576c73715e6c7c6b19fa249/litellm/utils.py#L1217)

[Create an issue](https://github.com/BerriAI/litellm/issues/new) **or** [make a PR](https://github.com/BerriAI/litellm/pulls) if you want to improve the exception mapping. 

**Note** For OpenAI and Azure we return the original exception (since they're of the OpenAI Error type). But we add the 'llm_provider' attribute to them. [See code](https://github.com/BerriAI/litellm/blob/a42c197e5a6de56ea576c73715e6c7c6b19fa249/litellm/utils.py#L1221)

| Original LLM Provider | Initial Status Code / Initial Error Message | Returned Exception | Returned Status Code
|----------------------|------------------------|-----------------|
| Anthropic | 401 | AuthenticationError | 401 |
| Anthropic | Could not resolve authentication method. Expected either api_key or auth_token to be set. | AuthenticationError | 401 |
| Anthropic | 400 | InvalidRequestError | 400 | 
| Anthropic | 429 | RateLimitError | 429 | 
| Replicate | Incorrect authentication token | AuthenticationError | 401 | 
| Replicate | ModelError | InvalidRequestError | 400 |
| Replicate | Request was throttled | RateLimitError | 429 |
| Replicate | ReplicateError | ServiceUnavailableError | 500 |
| Cohere | invalid api token | AuthenticationError | 401 |
| Cohere | too many tokens | InvalidRequestError | 400 |
| Cohere | CohereConnectionError | RateLimitError | 429 |
| Huggingface | 401 | AuthenticationError | 401 |
| Huggingface | 400 | InvalidRequestError | 400 | 
| Huggingface | 429 | RateLimitError | 429 | 

