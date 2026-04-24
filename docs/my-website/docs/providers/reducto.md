# Reducto

## Overview

| Property | Details |
|-------|-------|
| Description | Reducto parse support over LiteLLM's existing OCR API |
| Provider Route on LiteLLM | `reducto/` |
| Supported Operations | `/ocr` |
| Supported Models | `reducto/parse-v3`, `reducto/parse-legacy` |
| Link to Provider Doc | [Reducto ↗](https://platform.reducto.ai/) |

Reducto is exposed through LiteLLM's OCR surface, so this provider uses `litellm.ocr()` and `litellm.aocr()`.

## Quick Start

### LiteLLM SDK

```python showLineNumbers title="SDK Usage"
import litellm
import os

os.environ["REDUCTO_API_KEY"] = "your-api-key"

response = litellm.ocr(
    model="reducto/parse-v3",
    document={"type": "file", "file": "document.pdf"},
)

for page in response.pages:
    print(page.markdown)
```

You can also override credentials per call with `api_key=` and `api_base=`.

### LiteLLM Proxy

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: reducto-parse
    litellm_params:
      model: reducto/parse-v3
      api_key: os.environ/REDUCTO_API_KEY
    model_info:
      mode: ocr
```

## Parse V3

`reducto/parse-v3` maps to Reducto's current parse API and accepts:

- `formatting`
- `retrieval`
- `settings`

```python showLineNumbers title="Parse V3"
response = await litellm.aocr(
    model="reducto/parse-v3",
    document={"type": "file", "file": "document.pdf"},
    formatting={"table_output_format": "html"},
    retrieval={"chunk_mode": "section"},
    settings={"ocr_system": "standard"},
)
```

## Parse Legacy

`reducto/parse-legacy` keeps the legacy request shape and accepts `enhance`.

```python showLineNumbers title="Parse Legacy"
response = litellm.ocr(
    model="reducto/parse-legacy",
    document={"type": "file", "file": "document.pdf"},
    enhance={"agentic": [{"type": "table"}]},
)
```

## Upload Behavior

- `document={"type":"file","file":...}` is auto-converted by LiteLLM into a data URI, then uploaded to Reducto's `/upload` endpoint before `/parse`.
- `document_url="reducto://..."` is passed through directly and skips upload.
- Plain `http(s)` document URLs are rejected for Reducto. Upload the file first or pass a local file to LiteLLM.
- Image files also work through `type="file"`; LiteLLM normalizes them to `image_url` data URIs before the Reducto upload step.

## Cost Tracking

Reducto returns OCR usage in credits. LiteLLM supports credit-priced OCR models via `ocr_cost_per_credit`.

If you want spend tracking, register pricing for your deployment:

```python showLineNumbers title="Register OCR Credit Pricing"
import litellm

litellm.register_model(
    {
        "reducto/parse-v3": {
            "litellm_provider": "reducto",
            "mode": "ocr",
            "ocr_cost_per_credit": 0.003,
        }
    }
)
```
