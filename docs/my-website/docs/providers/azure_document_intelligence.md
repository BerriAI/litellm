# Azure Document Intelligence OCR

## Overview

| Property | Details |
|-------|-------|
| Description | Azure Document Intelligence (formerly Form Recognizer) provides advanced document analysis capabilities including text extraction, layout analysis, and structure recognition |
| Provider Route on LiteLLM | `azure_ai/doc-intelligence/` |
| Supported Operations | `/ocr` |
| Link to Provider Doc | [Azure Document Intelligence ↗](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)

Extract text and analyze document structure using Azure Document Intelligence's powerful prebuilt models.

## Quick Start

### **LiteLLM SDK**

```python showLineNumbers title="SDK Usage"
import litellm
import os

# Set environment variables
os.environ["AZURE_DOCUMENT_INTELLIGENCE_API_KEY"] = "your-api-key"
os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"] = "https://your-resource.cognitiveservices.azure.com"

# OCR with PDF URL
response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-layout",
    document={
        "type": "document_url",
        "document_url": "https://example.com/document.pdf"
    }
)

# Access extracted text
for page in response.pages:
    print(f"Page {page.index}:")
    print(page.markdown)
```

### **LiteLLM PROXY**

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: azure-doc-intel
    litellm_params:
      model: azure_ai/doc-intelligence/prebuilt-layout
      api_key: os.environ/AZURE_DOCUMENT_INTELLIGENCE_API_KEY
      api_base: os.environ/AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
    model_info:
      mode: ocr
```

**Start Proxy**
```bash
litellm --config proxy_config.yaml
```

**Call OCR via Proxy**
```bash showLineNumbers title="cURL Request"
curl -X POST http://localhost:4000/ocr \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "azure-doc-intel",
    "document": {
      "type": "document_url",
      "document_url": "https://arxiv.org/pdf/2201.04234"
    }
  }'
```

## Authentication

Azure Document Intelligence uses subscription key authentication.

### Environment Variables

```python showLineNumbers title="Environment Variable Auth"
import os

os.environ["AZURE_DOCUMENT_INTELLIGENCE_API_KEY"] = "your-subscription-key"
os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"] = "https://your-resource.cognitiveservices.azure.com"

response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-layout",
    document={"type": "document_url", "document_url": "https://..."}
)
```

### Direct Parameters

```python showLineNumbers title="Parameter Auth"
response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-layout",
    document={"type": "document_url", "document_url": "https://..."},
    api_key="your-subscription-key",
    api_base="https://your-resource.cognitiveservices.azure.com"
)
```

## Supported Models

Azure Document Intelligence offers several prebuilt models optimized for different use cases:

### prebuilt-layout (Recommended)

Best for general document OCR with structure preservation.

```python showLineNumbers title="Layout Model"
response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-layout",
    document={
        "type": "document_url",
        "document_url": "https://example.com/document.pdf"
    }
)
```

**Features:**
- Text extraction with markdown formatting
- Table detection and extraction
- Document structure analysis
- Paragraph and section recognition

**Pricing:** $10 per 1,000 pages

### prebuilt-read

Optimized for reading text from documents.

```python showLineNumbers title="Read Model"
response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-read",
    document={
        "type": "document_url",
        "document_url": "https://example.com/document.pdf"
    }
)
```

**Features:**
- Fast text extraction
- Optimized for reading-heavy documents
- Basic structure recognition

**Pricing:** $1.50 per 1,000 pages

### prebuilt-document

General-purpose document analysis.

```python showLineNumbers title="Document Model"
response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-document",
    document={
        "type": "document_url",
        "document_url": "https://example.com/document.pdf"
    }
)
```

**Pricing:** $10 per 1,000 pages

## Document Types

Azure Document Intelligence supports various document formats.

### PDF Documents

```python showLineNumbers title="PDF OCR"
response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-layout",
    document={
        "type": "document_url",
        "document_url": "https://example.com/document.pdf"
    }
)
```

### Image Documents

```python showLineNumbers title="Image OCR"
response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-layout",
    document={
        "type": "image_url",
        "image_url": "https://example.com/image.png"
    }
)
```

**Supported image formats:** JPEG, PNG, BMP, TIFF

### Base64 Encoded Documents

```python showLineNumbers title="Base64 PDF"
import base64

# Read and encode PDF
with open("document.pdf", "rb") as f:
    pdf_base64 = base64.b64encode(f.read()).decode()

response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-layout",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{pdf_base64}"
    }
)
```

## Response Format

```python showLineNumbers title="Response Structure"
# Response has the following structure
response.pages          # List of pages with extracted text
response.model          # Model used
response.object         # "ocr"
response.usage_info     # Token usage information

# Access page content
for page in response.pages:
    print(f"Page {page.index}:")
    print(page.markdown)
    
    # Page dimensions (in pixels)
    if page.dimensions:
        print(f"Width: {page.dimensions.width}px")
        print(f"Height: {page.dimensions.height}px")
```

## Async Support

```python showLineNumbers title="Async Usage"
import litellm
import asyncio

async def process_document():
    response = await litellm.aocr(
        model="azure_ai/doc-intelligence/prebuilt-layout",
        document={
            "type": "document_url",
            "document_url": "https://example.com/document.pdf"
        }
    )
    return response

# Run async function
response = asyncio.run(process_document())
```

## Cost Tracking

LiteLLM automatically tracks costs for Azure Document Intelligence OCR:

| Model | Cost per 1,000 Pages |
|-------|---------------------|
| prebuilt-read | $1.50 |
| prebuilt-layout | $10.00 |
| prebuilt-document | $10.00 |

```python showLineNumbers title="View Cost"
response = litellm.ocr(
    model="azure_ai/doc-intelligence/prebuilt-layout",
    document={"type": "document_url", "document_url": "https://..."}
)

# Access cost information
print(f"Cost: ${response._hidden_params.get('response_cost', 0)}")
```

## Comparison with Azure AI Mistral OCR

LiteLLM supports two Azure OCR services:

| Feature | Azure Document Intelligence | Azure AI Mistral OCR |
|---------|---------------------------|---------------------|
| Provider Route | `azure_ai/doc-intelligence/<model>` | `azure_ai/mistral-document-ai-2505` |
| Service | Azure Cognitive Services | Azure AI Foundry |
| API | Document Intelligence v4.0 | Mistral OCR API |
| Auth | Subscription Key | API Key |
| Models | Multiple prebuilt models | Single Mistral model |
| Best For | Production scenarios, advanced features | Quick integration, Mistral ecosystem |

## Important Notes

:::info API Version Changes
Azure Document Intelligence v4.0 (2024-11-30) uses the `/documentintelligence/` endpoint path instead of the older `/formrecognizer/` path used in previous versions.
:::

:::info Document Limits
Azure Document Intelligence supports:
- **PDFs/TIFFs:** Up to 2,000 pages
- **File Size:** Up to 500 MB
- **Image Dimensions:** 50 x 50 to 10,000 x 10,000 pixels
:::

:::tip Regional Deployment
Create your Azure Document Intelligence resource in the region closest to your data for optimal performance and compliance.
:::

:::info Unified Format
LiteLLM transforms Azure Document Intelligence responses to the unified Mistral OCR format, ensuring consistency across all OCR providers.
:::

## Error Handling

```python showLineNumbers title="Error Handling"
import litellm
from litellm.exceptions import APIError

try:
    response = litellm.ocr(
        model="azure_ai/doc-intelligence/prebuilt-layout",
        document={
            "type": "document_url",
            "document_url": "https://example.com/document.pdf"
        }
    )
except APIError as e:
    print(f"API Error: {e}")
except ValueError as e:
    print(f"Configuration Error: {e}")
```

## Additional Resources

- [Azure Document Intelligence Documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- [Pricing Details](https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/)
- [Supported File Formats](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-model-overview)
- [LiteLLM OCR Documentation](https://docs.litellm.ai/docs/ocr)

