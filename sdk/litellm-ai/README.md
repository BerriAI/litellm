# LiteLLM TypeScript SDK

A fully typed TypeScript SDK for the [LiteLLM AI Gateway](https://github.com/BerriAI/litellm), auto-generated from the OpenAPI spec using [Hey API](https://heyapi.dev/).

One SDK to call **all** LiteLLM endpoints — `/chat/completions`, `/responses`, `/embeddings`, `/ocr`, `/rerank`, `/images/generations`, and every management API — with full TypeScript type safety.

## Installation

```bash
npm install litellm
```

## Quick Start

```typescript
import { client, chatCompletionV1ChatCompletionsPost } from "litellm";

// Configure the client to point at your LiteLLM proxy
client.setConfig({
  baseUrl: "http://localhost:4000",
  headers: {
    Authorization: "Bearer sk-your-api-key",
  },
});

// Chat Completions — fully typed request & response
const { data, error } = await chatCompletionV1ChatCompletionsPost({
  body: {
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hello!" }],
  },
});

if (error) {
  console.error(error);
} else {
  console.log(data);
}
```

## Usage Examples

### Embeddings

```typescript
import { embeddingsV1EmbeddingsPost } from "litellm";

const { data } = await embeddingsV1EmbeddingsPost({
  body: {
    model: "text-embedding-3-small",
    input: "The quick brown fox jumps over the lazy dog",
  },
});
```

### OCR

```typescript
import { ocrV1OcrPost } from "litellm";

const { data } = await ocrV1OcrPost({
  body: {
    model: "gpt-4o",
    url: "https://example.com/document.pdf",
  },
});
```

### Responses API

```typescript
import { responsesApiV1ResponsesPost } from "litellm";

const { data } = await responsesApiV1ResponsesPost({
  body: {
    model: "gpt-4o",
    input: "Tell me a joke",
  },
});
```

### Image Generation

```typescript
import { imageGenerationV1ImagesGenerationsPost } from "litellm";

const { data } = await imageGenerationV1ImagesGenerationsPost({
  body: {
    model: "dall-e-3",
    prompt: "A sunset over mountains",
  },
});
```

### Rerank

```typescript
import { rerankV1RerankPost } from "litellm";

const { data } = await rerankV1RerankPost({
  body: {
    model: "rerank-english-v3.0",
    query: "What is the capital of France?",
    documents: ["Paris is the capital of France", "London is a city in England"],
  },
});
```

### Key Management

```typescript
import { generateKeyFnKeyGeneratePost, listKeysKeyListGet } from "litellm";

// Generate a new API key
const { data: newKey } = await generateKeyFnKeyGeneratePost({
  body: {
    models: ["gpt-4o"],
    max_budget: 100,
  },
});

// List all keys
const { data: keys } = await listKeysKeyListGet();
```

### Custom Client Instance

```typescript
import { createClient, createConfig, chatCompletionV1ChatCompletionsPost } from "litellm";

const myClient = createClient(
  createConfig({
    baseUrl: "http://my-litellm-proxy:4000",
    headers: {
      Authorization: "Bearer sk-my-key",
    },
  })
);

const { data } = await chatCompletionV1ChatCompletionsPost({
  client: myClient,
  body: {
    model: "claude-sonnet-4-20250514",
    messages: [{ role: "user", content: "Hello!" }],
  },
});
```

### Error Handling with `throwOnError`

```typescript
import { chatCompletionV1ChatCompletionsPost } from "litellm";

try {
  const { data } = await chatCompletionV1ChatCompletionsPost({
    throwOnError: true,
    body: {
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hello!" }],
    },
  });
  console.log(data);
} catch (error) {
  console.error("Request failed:", error);
}
```

## Regenerating the SDK

The SDK is generated from the LiteLLM proxy's OpenAPI spec. To regenerate:

```bash
# 1. Extract the OpenAPI spec from the proxy FastAPI app
uv run python sdk/litellm-ai/scripts/extract-openapi.py

# 2. Regenerate the TypeScript SDK
cd sdk/litellm-ai
npm run generate
```

## Building

```bash
npm run build
```

## Architecture

This SDK is auto-generated using [@hey-api/openapi-ts](https://heyapi.dev/) from the LiteLLM proxy's OpenAPI specification. The generated code includes:

- **`src/client/types.gen.ts`** — TypeScript interfaces for all request/response types
- **`src/client/sdk.gen.ts`** — Typed SDK functions for every API endpoint
- **`src/client/client.gen.ts`** — Pre-configured HTTP client (Fetch-based)
- **`src/client/client/`** — Core HTTP client implementation
- **`src/index.ts`** — Public entry point re-exporting everything

The SDK uses the **flat strategy** by default, meaning each endpoint is a standalone tree-shakeable function — your bundler will only include the endpoints you actually use.
