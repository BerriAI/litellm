# Voyage AI
https://docs.voyageai.com/embeddings/

## API Key
```python
# env variable
os.environ['VOYAGE_API_KEY']
```

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['VOYAGE_API_KEY'] = ""
response = embedding(
    model="voyage/voyage-3.5",
    input=["good morning from litellm"],
)
print(response)
```

## Supported Parameters

VoyageAI embeddings support the following optional parameters:

- `input_type`: Specifies the type of input for retrieval optimization
  - `"query"`: Use for search queries
  - `"document"`: Use for documents being indexed
- `dimensions`: Output embedding dimensions (256, 512, 1024, or 2048)
- `encoding_format`: Output format (`"float"`, `"int8"`, `"uint8"`, `"binary"`, `"ubinary"`)
- `truncation`: Whether to truncate inputs exceeding max tokens (default: `True`)

### Example with Parameters

```python
from litellm import embedding
import os

os.environ['VOYAGE_API_KEY'] = "your-api-key"

# Embedding with custom dimensions and input type
response = embedding(
    model="voyage/voyage-3.5",
    input=["Your text here"],
    dimensions=512,
    input_type="document"
)
print(f"Embedding dimensions: {len(response.data[0]['embedding'])}")
```

## Supported Models
All models listed here https://docs.voyageai.com/embeddings/#models-and-specifics are supported

| Model Name              | Function Call                                              |
|-------------------------|------------------------------------------------------------|
| voyage-3.5              | `embedding(model="voyage/voyage-3.5", input)`              | 
| voyage-3.5-lite         | `embedding(model="voyage/voyage-3.5-lite", input)`         | 
| voyage-3-large          | `embedding(model="voyage/voyage-3-large", input)`          | 
| voyage-3                | `embedding(model="voyage/voyage-3", input)`                | 
| voyage-3-lite           | `embedding(model="voyage/voyage-3-lite", input)`           | 
| voyage-code-3           | `embedding(model="voyage/voyage-code-3", input)`           | 
| voyage-finance-2        | `embedding(model="voyage/voyage-finance-2", input)`        | 
| voyage-law-2            | `embedding(model="voyage/voyage-law-2", input)`            | 
| voyage-code-2           | `embedding(model="voyage/voyage-code-2", input)`           | 
| voyage-multilingual-2   | `embedding(model="voyage/voyage-multilingual-2	", input)`  | 
| voyage-large-2-instruct | `embedding(model="voyage/voyage-large-2-instruct", input)` | 
| voyage-large-2          | `embedding(model="voyage/voyage-large-2", input)`          |
| voyage-2                | `embedding(model="voyage/voyage-2", input)`                | 
| voyage-lite-02-instruct | `embedding(model="voyage/voyage-lite-02-instruct", input)` | 
| voyage-01               | `embedding(model="voyage/voyage-01", input)`               | 
| voyage-lite-01          | `embedding(model="voyage/voyage-lite-01", input)`          |
| voyage-lite-01-instruct | `embedding(model="voyage/voyage-lite-01-instruct", input)` |

## Contextual Embeddings (voyage-context-3)

VoyageAI's `voyage-context-3` model provides contextualized chunk embeddings, where each chunk is embedded with awareness of its surrounding document context. This significantly improves retrieval quality compared to standard context-agnostic embeddings.

### Key Benefits
- Chunks understand their position and role within the full document
- Improved retrieval accuracy for long documents (outperforms competitors by 7-23%)
- Better handling of ambiguous references and cross-chunk dependencies
- Seamless drop-in replacement for standard embeddings in RAG pipelines

### Usage

Contextual embeddings require a **nested input format** where each inner list represents chunks from a single document:

```python
from litellm import embedding
import os

os.environ['VOYAGE_API_KEY'] = "your-api-key"

# Single document with multiple chunks
response = embedding(
    model="voyage/voyage-context-3",
    input=[
        [
            "Chapter 1: Introduction to AI",
            "This chapter covers the basics of artificial intelligence.",
            "We will explore machine learning and deep learning."
        ]
    ]
)
print(f"Number of chunk groups: {len(response.data)}")

# Multiple documents
response = embedding(
    model="voyage/voyage-context-3",
    input=[
        ["Paris is the capital of France.", "It is known for the Eiffel Tower."],
        ["Tokyo is the capital of Japan.", "It is a major economic hub."]
    ]
)
print(f"Processed {len(response.data)} documents")
```

### Specifications
- Model: `voyage-context-3`
- Context length: 32,000 tokens per document
- Output dimensions: 256, 512, 1024 (default), or 2048
- Max inputs: 1,000 per request
- Max total tokens: 120,000
- Max chunks: 16,000
- Pricing: $0.18 per million tokens

### When to Use Contextual Embeddings

**Use `voyage-context-3` when:**
- Processing long documents split into chunks
- Document structure and flow are important
- References between sections matter
- You need to preserve document hierarchy

**Use standard models (voyage-3.5, voyage-3-large) when:**
- Embedding independent pieces of text
- Processing short queries
- Document context is not relevant
- You need faster/cheaper processing

## Model Selection Guide

| Model | Best For | Context Length | Price/M Tokens |
|-------|----------|----------------|----------------|
| voyage-3.5 | General-purpose, multilingual | 32K | $0.06 |
| voyage-3.5-lite | Latency-sensitive applications | 32K | $0.02 |
| voyage-3-large | Best overall quality | 32K | $0.18 |
| voyage-code-3 | Code retrieval and search | 32K | $0.18 |
| voyage-finance-2 | Financial documents | 32K | $0.12 |
| voyage-law-2 | Legal documents | 16K | $0.12 |
| voyage-context-3 | Contextual document embeddings | 32K | $0.18 |
