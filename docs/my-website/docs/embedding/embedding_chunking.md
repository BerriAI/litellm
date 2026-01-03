import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Embedding Context Limit Handling

LiteLLM Router can automatically handle large embedding inputs that exceed a model's context limit by chunking the text and merging the resulting embeddings.

## Overview

Many embedding models have token limits (e.g., 512 tokens for some models, 8192 for others). When you have text that exceeds these limits, LiteLLM can:

1. **Automatically chunk** the input text into smaller pieces
2. **Embed each chunk** separately (with concurrent processing)
3. **Merge the embeddings** by averaging the vectors

This is useful for:
- Processing long documents without manual chunking
- Ensuring consistent embedding dimensions regardless of input length
- Avoiding "context length exceeded" errors

## Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "text-embedding-ada-002",
            "litellm_params": {"model": "text-embedding-ada-002"},
        }
    ],
    # Enable automatic chunking for large inputs
    enforce_embedding_context_limit=True,
    embedding_chunk_size=512,  # tokens per chunk
)

# This will automatically chunk if the text exceeds 512 tokens
response = router.embedding(
    model="text-embedding-ada-002",
    input="Your very long text here..."
)

# Works with async too
response = await router.aembedding(
    model="text-embedding-ada-002",
    input="Your very long text here..."
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
# config.yaml
model_list:
  - model_name: text-embedding-ada-002
    litellm_params:
      model: text-embedding-ada-002

router_settings:
  enforce_embedding_context_limit: true
  embedding_chunk_size: 512
```

```bash
$ litellm --config /path/to/config.yaml
```

```shell
curl --location 'http://0.0.0.0:4000/v1/embeddings' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "input": "Your very long text here...",
    "model": "text-embedding-ada-002"
}'
```

</TabItem>
</Tabs>

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enforce_embedding_context_limit` | `bool` | `False` | When `True`, automatically chunks inputs that exceed `embedding_chunk_size` |
| `embedding_chunk_size` | `int` | `512` | Maximum tokens per chunk. Set based on your model's context limit |

## How It Works

### Token Estimation

LiteLLM uses a fast approximation for token counting:
- **Base formula**: `tokens ≈ characters / 4`
- **Safety factor**: 1.1x multiplier to account for tokenizer variations

This conservative estimate ensures chunks won't exceed the actual token limit (for English language).

### Chunking Strategy

1. Text is split into chunks of approximately `embedding_chunk_size` tokens
2. Chunks break at word boundaries when possible (searches last 10% for spaces)
3. A 0.9x safety margin is applied to chunk sizes

### Embedding Merging

When text is chunked, the resulting embeddings are merged by **element-wise averaging**:

```
merged_embedding[i] = mean(chunk_1[i], chunk_2[i], ..., chunk_n[i])
```

This produces a single embedding vector with the same dimensions as individual chunk embeddings.

## Common Chunk Sizes

| Model | Recommended `embedding_chunk_size` |
|-------|-----------------------------------|
| OpenAI text-embedding-ada-002 | 8191 |
| OpenAI text-embedding-3-small | 8191 |
| OpenAI text-embedding-3-large | 8191 |
| Cohere embed-english-v3.0 | 512 |
| Most other models | 512 (safe default) |

## Example: Processing Long Documents

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "embeddings",
            "litellm_params": {"model": "text-embedding-ada-002"},
        }
    ],
    enforce_embedding_context_limit=True,
    embedding_chunk_size=2048,  # Use larger chunks for ada-002
)

# Process a long document
long_document = open("research_paper.txt").read()  # e.g., 50,000 characters

response = router.embedding(
    model="embeddings",
    input=long_document
)

# Single embedding vector returned, regardless of document length
embedding = response.data[0]["embedding"]
print(f"Embedding dimensions: {len(embedding)}")
```

## Example: Batch Processing

```python
# Multiple inputs - each will be chunked independently if needed
inputs = [
    "Short text",
    "A" * 10000,  # Very long text - will be chunked
    "Medium length text here",
]

response = router.embedding(
    model="embeddings",
    input=inputs
)

# Returns one embedding per input
for i, item in enumerate(response.data):
    print(f"Input {i}: embedding dimensions = {len(item['embedding'])}")
```

## Performance Considerations

- **Concurrent Processing**: Chunks are processed concurrently for better performance
- **Token Estimation**: Uses fast character-based estimation (no tokenizer overhead)
- **Memory**: Large inputs are processed in chunks, reducing peak memory usage

## When to Use This Feature

✅ **Use when:**
- Processing documents of unknown/variable length
- You want to avoid "context length exceeded" errors
- You need consistent embedding dimensions for any input size
- Document classification or clustering (understanding general topic/theme)

❌ **Don't use when:**
- **Precise Retrieval (RAG)**: If you need to retrieve specific facts or passages, averaging the chunk vectors will "smear" the meaning across the entire document. For RAG applications, you should manually chunk your documents and store each chunk as a separate vector record in your database.
- You need precise control over how text is chunked (use manual chunking)
- Your text is always within the model's context limit
