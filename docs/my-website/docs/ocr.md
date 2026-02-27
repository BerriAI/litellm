# /ocr

| Feature | Supported | 
|---------|-----------|
| Cost Tracking | ✅ |
| Logging | ✅ (Basic Logging not supported) |
| Load Balancing | ✅ |
| Supported Providers | `mistral`, `azure_ai`, `vertex_ai` |

:::tip

LiteLLM follows the [Mistral API request/response for the OCR API](https://docs.mistral.ai/capabilities/vision/#optical-character-recognition-ocr)

:::

## **LiteLLM Python SDK Usage**
### Quick Start 

```python
from litellm import ocr
import os

os.environ["MISTRAL_API_KEY"] = "sk-.."

response = ocr(
    model="mistral/mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234"
    }
)

# Access extracted text
for page in response.pages:
    print(f"Page {page.index}:")
    print(page.markdown)
```

### Async Usage 

```python
from litellm import aocr
import os, asyncio

os.environ["MISTRAL_API_KEY"] = "sk-.."

async def test_async_ocr(): 
    response = await aocr(
        model="mistral/mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": "https://arxiv.org/pdf/2201.04234"
        }
    )
    
    # Access extracted text
    for page in response.pages:
        print(f"Page {page.index}:")
        print(page.markdown)

asyncio.run(test_async_ocr())
```

### Using Local Files

LiteLLM can read local files directly — no manual base64 encoding needed:

```python
from litellm import ocr

# OCR with a local PDF file path
response = ocr(
    model="mistral/mistral-ocr-latest",
    document={
        "type": "file",
        "file": "/path/to/document.pdf"
    }
)

# OCR with a file object
response = ocr(
    model="mistral/mistral-ocr-latest",
    document={
        "type": "file",
        "file": open("document.pdf", "rb")
    }
)

# OCR with raw bytes
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

response = ocr(
    model="mistral/mistral-ocr-latest",
    document={
        "type": "file",
        "file": pdf_bytes,
        "mime_type": "application/pdf"  # recommended for raw bytes (auto-detected from extension for file paths)
    }
)
```

The `file` field accepts:
- **File path** (`str` or `pathlib.Path`) — LiteLLM reads the file and detects the MIME type from the extension
- **File object** (binary file-like object) — e.g. `open("doc.pdf", "rb")`
- **Raw bytes** (`bytes`) — use `mime_type` to specify the content type

LiteLLM automatically converts file inputs to base64 data URIs internally, so all providers work seamlessly.

### Using Base64 Encoded Documents

```python
import base64
from litellm import ocr

# Encode PDF to base64
with open("document.pdf", "rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')

response = ocr(
    model="mistral/mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}"
    }
)
```

### Optional Parameters

```python
response = ocr(
    model="mistral/mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://example.com/doc.pdf"
    },
    # Optional Mistral parameters
    pages=[0, 1, 2],              # Only process specific pages
    include_image_base64=True,     # Include extracted images
    image_limit=10,                # Max images to return
    image_min_size=100             # Min image size to include
)
```

## **LiteLLM Proxy Usage**

LiteLLM provides a Mistral API compatible `/ocr` endpoint for OCR calls.

**Setup**

Add this to your litellm proxy config.yaml

```yaml
model_list:
  - model_name: mistral-ocr
    litellm_params:
      model: mistral/mistral-ocr-latest
      api_key: os.environ/MISTRAL_API_KEY
```

Start litellm

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**Test request — JSON body**

```bash
curl http://0.0.0.0:4000/v1/ocr \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-ocr",
    "document": {
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234"
    }
  }'
```

**Test request — multipart file upload**

Upload a file directly using multipart form data. No need to base64-encode the file yourself.

```bash
curl http://0.0.0.0:4000/v1/ocr \
  -H "Authorization: Bearer sk-1234" \
  -F "model=mistral-ocr" \
  -F "file=@/path/to/document.pdf"
```

You can also pass optional parameters as additional form fields:

```bash
curl http://0.0.0.0:4000/v1/ocr \
  -H "Authorization: Bearer sk-1234" \
  -F "model=mistral-ocr" \
  -F "file=@screenshot.png" \
  -F 'pages=[0,1,2]' \
  -F "include_image_base64=true"
```

## **Request/Response Format**

:::info

LiteLLM follows the **Mistral OCR API specification**. 

See the [official Mistral OCR documentation](https://docs.mistral.ai/capabilities/vision/#optical-character-recognition-ocr) for complete details.

:::

### Example Request

```python
{
    "model": "mistral/mistral-ocr-latest",
    "document": {
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234"
    },
    "pages": [0, 1, 2],              # Optional: specific pages to process
    "include_image_base64": True,     # Optional: include extracted images
    "image_limit": 10,                # Optional: max images to return
    "image_min_size": 100             # Optional: min image size in pixels
}
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | The OCR model to use (e.g., `"mistral/mistral-ocr-latest"`) |
| `document` | object | Yes | Document to process. Must contain `type` and the corresponding field |
| `document.type` | string | Yes | `"document_url"` for PDFs/docs, `"image_url"` for images, or `"file"` for local files |
| `document.document_url` | string | Conditional | URL or data URI to the document (required if `type` is `"document_url"`) |
| `document.image_url` | string | Conditional | URL or data URI to the image (required if `type` is `"image_url"`) |
| `document.file` | string/bytes/file | Conditional | File path, bytes, or file-like object (required if `type` is `"file"`) |
| `document.mime_type` | string | No | Explicit MIME type for file inputs (auto-detected from extension if not provided) |
| `pages` | array | No | List of specific page indices to process (0-indexed) |
| `include_image_base64` | boolean | No | Whether to include extracted images as base64 strings |
| `image_limit` | integer | No | Maximum number of images to return |
| `image_min_size` | integer | No | Minimum size (in pixels) for images to include |

#### Document Format Examples

**For PDFs and documents (URL):**
```json
{
  "type": "document_url",
  "document_url": "https://example.com/document.pdf"
}
```

**For images (URL):**
```json
{
  "type": "image_url",
  "image_url": "https://example.com/image.png"
}
```

**For base64-encoded content:**
```json
{
  "type": "document_url",
  "document_url": "data:application/pdf;base64,JVBERi0xLjQKJ..."
}
```

**For local files (SDK):**
```python
{"type": "file", "file": "/path/to/document.pdf"}
{"type": "file", "file": open("image.png", "rb")}
{"type": "file", "file": pdf_bytes, "mime_type": "application/pdf"}
```

**For file uploads (Proxy — multipart form):**
```bash
curl http://0.0.0.0:4000/v1/ocr \
  -H "Authorization: Bearer sk-1234" \
  -F "model=mistral-ocr" \
  -F "file=@document.pdf"
```

### Response Format

The response follows Mistral's OCR format with the following structure:

```json
{
  "pages": [
    {
      "index": 0,
      "markdown": "# Document Title\n\nExtracted text content...",
      "dimensions": {
        "dpi": 200,
        "height": 2200,
        "width": 1700
      },
      "images": [
        {
          "image_base64": "base64string...",
          "bbox": {
            "x": 100,
            "y": 200,
            "width": 300,
            "height": 400
          }
        }
      ]
    }
  ],
  "model": "mistral-ocr-2505-completion",
  "usage_info": {
    "pages_processed": 29,
    "doc_size_bytes": 3002783
  },
  "document_annotation": null,
  "object": "ocr"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `pages` | array | List of processed pages with extracted content |
| `pages[].index` | integer | Page number (0-indexed) |
| `pages[].markdown` | string | Extracted text in Markdown format |
| `pages[].dimensions` | object | Page dimensions (dpi, height, width in pixels) |
| `pages[].images` | array | Extracted images from the page (if `include_image_base64=true`) |
| `model` | string | The model used for OCR processing |
| `usage_info` | object | Processing statistics (pages processed, document size) |
| `document_annotation` | object | Optional document-level annotations |
| `object` | string | Always `"ocr"` for OCR responses |


## **Supported Providers**

| Provider    | Link to Usage      |
|-------------|--------------------|
| Mistral AI  |   [Usage](#quick-start)                 |
| Azure AI    |   [Usage](../docs/providers/azure_ocr)                 |
| Vertex AI   |   [Usage](../docs/providers/vertex_ocr)                 |

