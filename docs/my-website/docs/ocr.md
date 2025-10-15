# /ocr

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

Test request

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

### Response Format

The OCR endpoint returns a response in Mistral's native OCR format:

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
      "images": []
    }
  ],
  "model": "mistral-ocr-2505-completion",
  "usage_info": {
    "pages_processed": 29,
    "doc_size_bytes": 3002783
  },
  "object": "ocr"
}
```

### Accessing Response Data

```python
# Get total pages
print(f"Processed {len(response.pages)} pages")

# Extract all text
all_text = "\n\n".join(page.markdown for page in response.pages)

# Get usage information
if response.usage_info:
    print(f"Pages processed: {response.usage_info.pages_processed}")
    print(f"Document size: {response.usage_info.doc_size_bytes} bytes")

# Access specific page
first_page = response.pages[0]
print(f"First page dimensions: {first_page.dimensions.width}x{first_page.dimensions.height}")
```

## **Supported Providers**

| Provider    | Link to Usage      |
|-------------|--------------------|
| Mistral AI  |   [Usage](#quick-start)                 |

