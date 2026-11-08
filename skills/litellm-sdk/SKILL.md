---
name: litellm-sdk
description: Set up the LiteLLM Python SDK in an existing app, from project and credential detection through install, migration, optional Router fallbacks, and a verified completion with cost and latency. Use when the user asks to use LiteLLM inside Python or replace direct OpenAI or Anthropic client calls. For a self-hosted gateway or a non-Python app, use litellm-setup instead
---

# Set up the LiteLLM Python SDK

Take the user's existing Python project to a verified LiteLLM call. Run the commands yourself, preserve the project's conventions, and change only the integration. No proxy, database, or master key is needed.

Never print a secret, dump the environment, read a whole `.env` into output, or enable debug logs with credentials. Use existing environment variables without copying their values. Put only empty key names in `.env.example`. Never write real keys. If credentials are missing, ask the user to set them in their environment or existing secret manager, not paste them into chat. Keep `.env` out of git. If the project already loads `.env`, reuse its loader for the checks and calls below without printing values.

## 1. Inspect and ask once

Read the project instructions, manifests, lockfiles, entry points, direct provider calls, and tests. Detect language, framework (such as FastAPI, Django, or Flask), sync or async execution, and package manager from `uv.lock`, `poetry.lock`, `pdm.lock`, or pip requirements. For a non-Python project, stop. Keep its OpenAI SDK and point to the `litellm-setup` skill and https://docs.litellm.ai/docs/proxy/docker_quick_start for a Gateway instead.

Check credentials before asking. This prints names and presence only, including common cloud credential routes, never values:

```bash
python -c 'import os; names = "OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY AZURE_API_KEY AZURE_API_BASE AZURE_API_VERSION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN AWS_REGION_NAME AWS_REGION GOOGLE_APPLICATION_CREDENTIALS VERTEXAI_PROJECT VERTEXAI_LOCATION GROQ_API_KEY MISTRAL_API_KEY COHERE_API_KEY DEEPSEEK_API_KEY TOGETHERAI_API_KEY FIREWORKS_AI_API_KEY XAI_API_KEY".split(); print("\n".join(n + "=set" for n in names if os.environ.get(n)))'
```

For Bedrock, an AWS profile, role, or workload identity can replace static keys. For Vertex, Application Default Credentials can replace a key file. Check the project's configured credential route without printing tokens. Presence is not proof of access. Azure also needs an endpoint, API version, and deployed model name. Vertex needs project and location, and Bedrock needs a region. Ask for missing configuration, not a key that is already available.

Ask only unresolved questions in one message, with defaults they can accept with "go". Confirm permission to use the named variables, since they can belong to another account. Default to the existing provider when configured, otherwise a provider with credentials, and its newest suitable chat model from step 3. Offer `litellm.Router` fallbacks, default yes when two or more providers have credentials, otherwise no. Resolve an ambiguous package manager here. Explain that verification makes a small paid request. If the user already authorized these choices, continue without asking again.

## 2. Install with their tool

Use `uv add litellm`, `poetry add litellm`, or `pdm add litellm` in that project. For pip, use the project's virtual environment, run `python -m pip install litellm`, and add the dependency to its existing requirements file using its pinning convention. Preserve and update the appropriate lockfile. Do not install proxy extras or use a global tool install. For testing a local checkout, use `uv add --editable /absolute/path/to/litellm` or the tool's equivalent instead.

Run later Python commands with the same environment, for example replace `python` with `uv run python`, `poetry run python`, or `pdm run python` in the commands below.

## 3. Choose current models

Use the installed catalog, not model names from memory. This concrete one-liner prints provider-prefixed chat candidates for Anthropic. Repeat with each chosen provider (`openai`, `gemini`, `azure`, `bedrock`, `vertex_ai`, `groq`, `mistral`, and so on):

```bash
python -c 'import litellm, sys; p = sys.argv[1]; rows = ((n, r) for n, r in litellm.model_cost.items() if isinstance(r, dict)); print("\n".join(sorted(n if n.startswith(p + "/") else p + "/" + n for n, r in rows if str(r.get("litellm_provider", "")).startswith(p) and r.get("mode") == "chat")))' anthropic
```

If the catalog is unavailable, read `model_prices_and_context_window.json` from the checkout. Select the newest suitable model per chosen provider from these candidates. The catalog does not consistently include release dates, and alphabetical order does not mean newest. Compare family versions and dated IDs, consult the provider's current release documentation when ambiguous, and exclude deprecated models. Do not invent an ID or silently choose an older model after an access error. For Azure use `azure/<deployment-name>` for the user's deployment of the selected model, not a catalog ID as the deployment name. For Bedrock use a region-supported model or inference profile from the catalog. Confirm required provider configuration before a call.

## 4. Migrate the app

Replace direct `OpenAI().chat.completions.create(...)` calls with `litellm.completion(...)` and awaited async client calls with `await litellm.acompletion(...)`. Pass the provider-prefixed model selected above and keep messages, tool definitions, tool choice, generation settings, and `stream=True` where used. Let LiteLLM read credentials from the environment. Do not hard-code keys. In async framework handlers use `acompletion` rather than blocking the event loop.

For Anthropic `messages.create`, move the top-level system prompt into a `system` message. Convert tools from `input_schema` to OpenAI function tools with `parameters`, `tool_use` blocks into assistant `tool_calls` with JSON argument strings, and `tool_result` blocks into `tool` messages with matching `tool_call_id`. Convert multimodal content using the target provider's supported OpenAI message format. Preserve the tool execution loop and check unsupported provider options rather than silently dropping them.

Read text from `response.choices[0].message.content`, not Anthropic `response.content[0].text`. Read tool requests from `response.choices[0].message.tool_calls`. For streaming, iterate chunks (or `async for` with `acompletion`) and read `chunk.choices[0].delta.content`; it can be `None`. Accumulate `delta.tool_calls` by index, including split argument strings, before executing tools. Replace Anthropic event parsing and update provider-specific exception handling. Extend the project's tests for any text, async, stream, or tool paths changed, including empty content and provider failures. Do not rewrite unrelated code or remove a provider dependency still used elsewhere.

If fallbacks were accepted, configure two model groups with the selected provider model IDs and environment credentials. Substitute actual IDs for `PRIMARY_MODEL` and `BACKUP_MODEL` below:

```python
import litellm

router = litellm.Router(
    model_list=[
        {"model_name": "primary", "litellm_params": {"model": PRIMARY_MODEL}},
        {"model_name": "backup", "litellm_params": {"model": BACKUP_MODEL}},
    ],
    fallbacks=[{"primary": ["backup"]}],
)
response = router.completion(model="primary", messages=messages)
```

Use `await router.acompletion(...)` in async code. A fallback can send the same prompt to another provider, so get consent before enabling it. Add empty entries for only the required credential and configuration names to `.env.example`, never values from the current environment.

## 5. Verify and hand off

Run one real non-streaming completion in the project environment with its chosen credentials. Set `LITELLM_MODEL` to the selected provider-prefixed ID (not a secret). If the app uses Router, run its configured Router path instead of the direct call below. Use a short prompt and print only the reply, estimated USD cost, and elapsed seconds:

```python
import os
from time import perf_counter
import litellm

started = perf_counter()
response = litellm.completion(
    model=os.environ["LITELLM_MODEL"],
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_tokens=128,
)
elapsed = perf_counter() - started
cost = litellm.completion_cost(completion_response=response)
print("Reply:", response.choices[0].message.content)
print("Cost (USD):", cost)
print(f"Latency (s): {elapsed:.3f}")
```

`litellm.completion_cost(completion_response=response)` uses response usage and catalog pricing. A missing price is not zero cost. Resolve the model's pricing or report verification incomplete. For Router, prefer its response's `response._hidden_params.get("response_cost")` so the actual served model's cost is used, with `completion_cost` as a fallback when absent. Streaming chunks are not a complete response for this cost calculation, so keep this check non-streaming.

Run the project's existing test command with its own tool, plus tests for the migrated paths. Fix integration failures before handoff. On authentication, access, quota, or model errors, report the failing provider and missing permission without exposing credentials. Do not claim success or substitute a mock for the real call. Show `git diff --check`, the final diff (never secret files), reply, cost, latency, and test result. Summarize the chosen models, changed files, and required environment variable names. Leave the user's project uncommitted unless asked to commit.
