# Azure AI OCR

## Overview

| Property | Details |
|-------|-------|
| Description | Azure AI OCR provides document intelligence capabilities powered by Mistral, enabling text extraction from PDFs and images |
| Provider Route on LiteLLM | `azure_ai/` |
| Supported Operations | `/ocr` |
| Link to Provider Doc | [Azure AI ↗](https://ai.azure.com/)

Extract text from documents and images using Azure AI's OCR models, powered by Mistral.

## Quick Start

### **LiteLLM SDK**

```python showLineNumbers title="SDK Usage"
import litellm
import os

# Set environment variables
os.environ["AZURE_AI_API_KEY"] = ""
os.environ["AZURE_AI_API_BASE"] = ""

# OCR with PDF URL
response = litellm.ocr(
    model="azure_ai/mistral-document-ai-2505",
    document={
        "type": "document_url",
        "document_url": "https://example.com/document.pdf"
    }
)

# Access extracted text
for page in response.pages:
    print(page.text)
```

### **LiteLLM PROXY**

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: azure-ocr
    litellm_params:
      model: azure_ai/mistral-document-ai-2505
      api_key: "os.environ/AZURE_AI_API_KEY"
      api_base: "os.environ/AZURE_AI_API_BASE"
    model_info:
      mode: ocr
```

## Document Types

Azure AI OCR supports both PDFs and images.

### PDF Documents

```python showLineNumbers title="PDF OCR"
response = litellm.ocr(
    model="azure_ai/mistral-document-ai-2505",
    document={
        "type": "document_url",
        "document_url": "https://example.com/document.pdf"
    }
)
```

### Image Documents

```python showLineNumbers title="Image OCR"
response = litellm.ocr(
    model="azure_ai/mistral-document-ai-2505",
    document={
        "type": "image_url",
        "image_url": "https://example.com/image.png"
    }
)
```

### Base64 Encoded Documents

```python showLineNumbers title="Base64 PDF"
import base64

# Read and encode PDF
with open("document.pdf", "rb") as f:
    pdf_base64 = base64.b64encode(f.read()).decode()

response = litellm.ocr(
    model="azure_ai/mistral-document-ai-2505",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{pdf_base64}"
    }
)
```

## Supported Parameters

```python showLineNumbers title="All Parameters"
response = litellm.ocr(
    model="azure_ai/mistral-document-ai-2505",
    document={                           # Required: Document to process
        "type": "document_url",
        "document_url": "https://..."
    },
    include_image_base64=True,           # Optional: Include base64 images
    pages=[0, 1, 2],                     # Optional: Specific pages to process
    image_limit=10                       # Optional: Limit number of images
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
    print(f"Page {page.page_number}:")
    print(page.text)
```

## Async Support

```python showLineNumbers title="Async Usage"
import litellm

response = await litellm.aocr(
    model="azure_ai/mistral-document-ai-2505",
    document={
        "type": "document_url",
        "document_url": "https://example.com/document.pdf"
    }
)
```

## Important Notes

:::info URL Conversion
Azure AI OCR endpoints don't have internet access. LiteLLM automatically converts public URLs to base64 data URIs before sending requests to Azure AI.
:::

## Supported Models

- `mistral-document-ai-2505` - Latest Mistral OCR model on Azure AI

Use the Azure AI provider prefix: `azure_ai/<model-name>`

