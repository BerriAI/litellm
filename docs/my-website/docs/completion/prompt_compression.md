# Prompt Compression (`compress()`)

Use `litellm.compress()` to shrink long conversation history before calling `completion()`.

The function keeps high-relevance and recent context, replaces low-relevance content with lightweight stubs, and returns a retrieval tool so the model can request full content only when needed.

## Quickstart

```python
import litellm

messages = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "# auth.py\n" + "def authenticate():\n    pass\n" * 2000},
    {"role": "user", "content": "# utils.py\n" + "def helper():\n    pass\n" * 2000},
    {"role": "user", "content": "Fix the bug in auth.py"},
]

compressed = litellm.compress(
    messages=messages,
    model="gpt-4o",
    compression_trigger=1000,
    compression_target=500,
)

response = litellm.completion(
    model="gpt-4o",
    messages=compressed["messages"],
    tools=compressed["tools"],
)
```

## What It Returns

`compress()` returns a dictionary with:

- `messages`: compressed conversation messages
- `original_tokens`: token count before compression
- `compressed_tokens`: token count after compression
- `compression_ratio`: fraction of tokens removed
- `cache`: key-value mapping of stub key -> original full content
- `tools`: retrieval tool definition (`litellm_content_retrieve`) for on-demand restoration

## Parameters

- `messages` (`List[dict]`, required): input conversation messages
- `model` (`str`, required): model name used for token counting
- `compression_trigger` (`int`, default `200000`): compress only if input token count exceeds this
- `compression_target` (`Optional[int]`, default `compression_trigger // 2`): desired post-compression token budget
- `embedding_model` (`Optional[str]`): if set, combines BM25 + embedding relevance scoring
- `embedding_model_params` (`Optional[dict]`): additional kwargs passed to `litellm.embedding()`
- `compression_cache` (`Optional[DualCache]`): optional cache used by embedding scoring

## Behavior Notes

- Messages below `compression_trigger` are passed through unchanged.
- System messages, the last user message, and the last assistant message are always preserved.
- If a relevant message does not fully fit the remaining budget, `compress()` may keep a truncated version of it.
- Compressed-out content is never lost; it is stored in `cache` and addressable by `litellm_content_retrieve`.

## Handling Retrieval Tool Calls

If the model calls `litellm_content_retrieve`, look up the requested key in `compressed["cache"]` and return that value as tool output.

```python
import json

tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
full_content = compressed["cache"][args["key"]]
```

## Evaluate Compression Quality

You can benchmark baseline vs compressed behavior with:

```bash
python scripts/eval_compression.py --model gpt-4o --problems 5
```
