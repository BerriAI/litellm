# Token0 — Vision Token Optimizer

Token0 is an open-source vision token optimizer that integrates with LiteLLM as a
`CustomLogger` pre-call hook. It automatically compresses images in your `messages`
payload before every LLM call — reducing vision token costs by 35–99% with no code
changes beyond adding the hook.

## Quick Start

**1. Install Token0**

```bash
pip install token0
```

**2. Add the hook — LiteLLM SDK**

```python
import litellm
from token0.litellm_hook import Token0Hook

litellm.callbacks = [Token0Hook()]

response = litellm.completion(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }]
)

# Check savings
print(response._hidden_params["metadata"]["token0"])
# {"tokens_saved": 1020, "optimizations": ["resize 4000x3000→1568x1176", "prompt-aware→low detail"]}
```

**2b. Add the hook — LiteLLM Proxy (`config.yaml`)**

```yaml
litellm_settings:
  callbacks: ["token0.litellm_hook.Token0Hook"]
```

Then install Token0 in the same environment as the proxy:

```bash
pip install token0
```

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable_cascade` | `bool` | `False` | Auto-route simple tasks to cheaper models (GPT-4o → GPT-4o-mini) |
| `detail_override` | `str \| None` | `None` | Force `"low"` or `"high"` detail mode for all images (OpenAI only) |

```python
# Enable model cascade
litellm.callbacks = [Token0Hook(enable_cascade=True)]

# Force low detail (fast, cheap — for classification tasks)
litellm.callbacks = [Token0Hook(detail_override="low")]
```

## What Gets Optimized

Token0 applies up to 7 optimizations per image, in order:

| Optimization | Savings | When Applied |
|---|---|---|
| Smart resize | Varies | Image exceeds provider's max resolution |
| OCR routing | 47–70% | Image is text-heavy (receipt, screenshot, invoice) |
| JPEG recompression | 10–30% | PNG without transparency |
| Prompt-aware detail | Up to 92% | Simple prompts ("classify", "yes/no") |
| Tile-optimized resize | 44% | Mid-size images on OpenAI (512px tile snapping) |
| Model cascade | 5–20x cost | `enable_cascade=True` + simple task detected |
| Semantic/fuzzy cache | 100% | Same or similar image+prompt seen before |

## Benchmarks

Benchmarked on 5 Ollama vision models across real-world images (photos, receipts, invoices, screenshots):

| Model | Direct Tokens | Token0 Tokens | Savings |
|---|---|---|---|
| granite3.2-vision | 129,836 | 60,924 | 53.1% |
| minicpm-v | 10,877 | 6,276 | 42.3% |
| moondream | 16,457 | 10,240 | 37.8% |
| llava-llama3 | 13,365 | 8,486 | 36.5% |
| llava:7b | 13,384 | 8,701 | 35.0% |

GPT-4.1 projections (using published token formulas):

| Optimization Set | Savings |
|---|---|
| Resize + OCR + PDF text extraction | 70.3% |
| All optimizations + model cascade | 98.9% |

## Supported Providers

Token0 is provider-aware and applies provider-specific optimizations:

| Provider | Models | Notes |
|---|---|---|
| OpenAI | GPT-4o, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano | Detail mode + tile optimization |
| Anthropic | Claude Sonnet/Opus/Haiku | Pixel-based token formula |
| Google | Gemini 2.5 Flash/Pro | |
| Ollama | Any vision model | Free, local inference |

## Text-Only Safety

Token0 is a no-op for text-only messages. It only activates when a `messages` array
contains at least one `image_url` content part. All text fields, tool calls, and
non-image content parts are passed through unmodified.

## Links

- [GitHub](https://github.com/Pritom14/token0)
- [PyPI](https://pypi.org/project/token0/)
- [License: Apache 2.0](https://github.com/Pritom14/token0/blob/main/LICENSE)
