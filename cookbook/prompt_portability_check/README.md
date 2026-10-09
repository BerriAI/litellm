# Lint a request for cross-provider portability before a LiteLLM provider swap

[LiteLLM](https://github.com/BerriAI/litellm) standardizes the *call* -- one
`completion()` signature across OpenAI, Anthropic, Gemini, Bedrock, and more.
It does not check whether the *content* of your messages/params (a hardcoded
"as an OpenAI assistant" system prompt, a provider-specific stop-sequence
limit, a temperature outside the target provider's accepted range, a leaked
chat-template token) will still behave correctly once you point `model=` at
a different provider.

[`prompt-portability`](https://github.com/nac7/prompt-portability) is an
open-source CLI/library that lints exactly that gap. This example lints a
request before handing it to `litellm.completion()`.

## Run it

```bash
pip install prompt-portability litellm
python check_before_provider_swap.py
```

## What it does

1. Builds a request payload written and tested against OpenAI.
2. Runs `prompt-portability`'s linter against it, surfacing portability
   issues that would only otherwise show up once the request actually hits
   a different provider.
3. Calls `litellm.completion()` with the same payload (mocked, so no API key
   is needed to run this example) to show LiteLLM's own call succeeds
   regardless -- confirming these are issues LiteLLM's translation layer
   does not catch on its own.
