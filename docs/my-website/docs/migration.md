# Migration

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

**How can we communicate changes better?**
Tell us
- [Discord](https://discord.com/invite/wuPM9dRgDw)
- Email (krrish@berri.ai/ishaan@berri.ai)
- Text us (+17708783106)
