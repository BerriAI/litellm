# Prompt Compression (`compress()`)

Use `litellm.compress()` to shrink long conversation history before calling `completion()`.

The function keeps high-relevance and recent context, replaces low-relevance content with lightweight stubs, and returns a retrieval tool so the model can request full content only when needed.

## Quickstart

```python
import litellm
from litellm.types.utils import CallTypes

messages = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "# auth.py\n" + "def authenticate():\n    pass\n" * 2000},
    {"role": "user", "content": "# utils.py\n" + "def helper():\n    pass\n" * 2000},
    {"role": "user", "content": "Fix the bug in auth.py"},
]

compressed = litellm.compress(
    messages=messages,
    model="gpt-4o",
    call_type=CallTypes.completion,
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
- `call_type` (`CallTypes`, default `CallTypes.completion`): the LiteLLM call type whose message schema these messages follow. Supported values: `CallTypes.completion` / `CallTypes.acompletion` (OpenAI chat-completions shape) and `CallTypes.anthropic_messages` (Anthropic Messages shape)
- `compression_trigger` (`int`, default `200000`): compress only if input token count exceeds this
- `compression_target` (`Optional[int]`, default `70% of compression_trigger`): desired post-compression token budget
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

## Server-side Callback Loop (`/v1/messages`)

You can enable callback-based compression interception to make retrieval loops
transparent for Anthropic Messages calls:

```yaml
litellm_settings:
  callbacks: ["compression_interception"]
  compression_interception_params:
    enabled: true
    compression_trigger: 10000
    compression_target: 7000
```

With this enabled, LiteLLM runs the following server-side flow:

1. Compresses inbound messages before the first provider call.
2. Injects the `litellm_content_retrieve` tool.
3. Detects retrieval `tool_use` blocks in the model response.
4. Resolves retrieval keys from the compression cache.
5. Reruns the model via agentic loop and returns the final answer.

## Performance

Benchmarked on [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_bm25_27K) (real GitHub issues with ~27k tokens of BM25-retrieved repo context per problem).

### Claude Opus — 5 problems, trigger=10k

| Metric | Baseline | Compressed | Delta |
|---|---|---|---|
| File overlap | 1.000 | 1.000 | +0.000 |
| Exact file match | 100% | 100% | +0.0% |
| Hunk overlap | 0.582 | 0.361 | -0.221 |
| Content similarity | 0.367 | 0.373 | +0.006 |
| Avg prompt tokens | 30,828 | 6,890 | -77.7% |
| Avg cost/problem | $0.488 | $0.136 | **-72.0%** |

**Key takeaways:**

- **File-level targeting is fully preserved** — the model edits the same files with or without compression.
- **Content similarity matches baseline** — the actual lines changed are comparable.
- **Hunk overlap drops modestly** (-0.221) — the model targets the right files but may edit slightly different line ranges with less surrounding context.
- **72% cost savings** with 78% token reduction.

### Metrics explained

| Metric | What it measures |
|---|---|
| **File overlap** | Fraction of gold-patch files present in the generated patch |
| **Exact file match** | Whether the generated patch touches exactly the same set of files |
| **Hunk overlap** | Fraction of gold hunk line ranges covered by generated hunks |
| **Content similarity** | Jaccard similarity of changed lines (added/removed) between gold and generated patches |

### Running the SWE-bench eval

```bash
# 5-problem quick check
python tests/eval_swe_bench.py --model claude-opus-4-20250514 --problems 5

# Custom trigger/target
python tests/eval_swe_bench.py --model gpt-4o --problems 20 \
    --compression-trigger 15000 --compression-target 10000

# With embedding scoring
python tests/eval_swe_bench.py --model gpt-4o --problems 10 \
    --embedding-model text-embedding-3-small
```

### Running the HumanEval-style eval

```bash
python scripts/eval_compression.py --model gpt-4o --problems 5
```
