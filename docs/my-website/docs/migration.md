# Migration Guide - LiteLLM v1.0.0+ 

When we have breaking changes (i.e. going from 1.x.x to 2.x.x), we will document those changes here.


## `1.0.0` 

**Last Release before breaking change**: 0.14.0

**What changed?**

- Requires `openai>=1.0.0`
- `openai.InvalidRequestError` → `openai.BadRequestError`
- `openai.ServiceUnavailableError` → `openai.APIStatusError`
- *NEW* litellm client, allow users to pass api_key
    - `litellm.Litellm(api_key="sk-123")`
- response objects now inherit from `BaseModel` (prev. `OpenAIObject`)
- *NEW* default exception - `APIConnectionError` (prev. `APIError`)
- litellm.get_max_tokens() now returns an int not a dict
    ```python
    max_tokens = litellm.get_max_tokens("gpt-3.5-turbo") # returns an int not a dict 
    assert max_tokens==4097
    ```
- Streaming - OpenAI Chunks now return `None` for empty stream chunks. This is how to process stream chunks with content
    ```python
    response = litellm.completion(model="gpt-3.5-turbo", messages=messages, stream=True)
    for part in response:
        print(part.choices[0].delta.content or "")
    ```

**How can we communicate changes better?**
Tell us
- [Discord](https://discord.com/invite/wuPM9dRgDw)
- Email (krrish@berri.ai/ishaan@berri.ai)
- Text us (+17708783106)
