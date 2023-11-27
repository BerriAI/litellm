# Exception Mapping

LiteLLM maps exceptions across all providers to their OpenAI counterparts.
- Rate Limit Errors
- Invalid Request Errors
- Authentication Errors
- Timeout Errors `openai.APITimeoutError`
- ServiceUnavailableError 
- APIError 
- APIConnectionError

Base case we return APIConnectionError

All our exceptions inherit from OpenAI's exception types, so any error-handling you have for that, should work out of the box with LiteLLM. 

For all cases, the exception returned inherits from the original OpenAI Exception but contains 3 additional attributes: 
* status_code - the http status code of the exception
* message - the error message
* llm_provider - the provider raising the exception

## Usage

```python 
import litellm
import openai

try:
    response = litellm.completion(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": "hello, write a 20 pageg essay"
                    }
                ],
                timeout=0.01, # this will raise a timeout exception
            )
except openai.APITimeoutError as e:
    print("Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e)
    print(type(e))
    pass
```

## Usage - Catching Streaming Exceptions
```python
import litellm
try:
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello, write a 20 pg essay"
            }
        ],
        timeout=0.0001, # this will raise an exception
        stream=True,
    )
    for chunk in response:
        print(chunk)
except openai.APITimeoutError as e:
    print("Passed: Raised correct exception. Got openai.APITimeoutError\nGood Job", e)
    print(type(e))
    pass
except Exception as e:
    print(f"Did not raise error `openai.APITimeoutError`. Instead raised error type: {type(e)}, Error: {e}")

```

## Details 

To see how it's implemented - [check out the code](https://github.com/BerriAI/litellm/blob/a42c197e5a6de56ea576c73715e6c7c6b19fa249/litellm/utils.py#L1217)

[Create an issue](https://github.com/BerriAI/litellm/issues/new) **or** [make a PR](https://github.com/BerriAI/litellm/pulls) if you want to improve the exception mapping. 

**Note** For OpenAI and Azure we return the original exception (since they're of the OpenAI Error type). But we add the 'llm_provider' attribute to them. [See code](https://github.com/BerriAI/litellm/blob/a42c197e5a6de56ea576c73715e6c7c6b19fa249/litellm/utils.py#L1221)

## Custom mapping list

Base case - we return the original exception.

|               | ContextWindowExceededError | AuthenticationError | InvalidRequestError | RateLimitError | ServiceUnavailableError |
|---------------|----------------------------|---------------------|---------------------|---------------|-------------------------|
| Anthropic     | ✅                          | ✅                   | ✅                   | ✅             |                         |
| OpenAI        | ✅                          | ✅                     |✅                     |✅               |✅|
| Replicate     | ✅                          | ✅                   | ✅                   | ✅             | ✅                       |
| Cohere        | ✅                          | ✅                   | ✅                    | ✅             | ✅                        |
| Huggingface   | ✅                          | ✅                   | ✅                   | ✅             |                         |
| Openrouter    | ✅                          | ✅                   | ✅                    | ✅             |                         |
| AI21          | ✅                          | ✅                   | ✅                   | ✅             |                         |
| VertexAI          |                           |                   |✅                   |             |                         |
| Bedrock          |                           |                   |✅                   |             |                         |
| Sagemaker          |                           |                   |✅                   |             |                         |
| TogetherAI    | ✅                          | ✅                   | ✅                   | ✅             |                         |
| AlephAlpha    | ✅                          | ✅                   | ✅                   | ✅             | ✅                        |


> For a deeper understanding of these exceptions, you can check out [this](https://github.com/BerriAI/litellm/blob/d7e58d13bf9ba9edbab2ab2f096f3de7547f35fa/litellm/utils.py#L1544) implementation for additional insights.

The `ContextWindowExceededError` is a sub-class of `InvalidRequestError`. It was introduced to provide more granularity for exception-handling scenarios. Please refer to [this issue to learn more](https://github.com/BerriAI/litellm/issues/228).

Contributions to improve exception mapping are [welcome](https://github.com/BerriAI/litellm#contributing)
