# [BETA] Vertex AI Endpoints (Pass-Through)

Pass-through endpoints for Vertex AI - call provider-specific endpoint, in native format (no translation).

:::tip

Looking for the Unified API (OpenAI format) for VertexAI ? [Go here - using vertexAI with LiteLLM SDK or LiteLLM Proxy Server](../docs/providers/vertex.md)

:::

## Supported API Endpoints

- Gemini API
- Embeddings API
- Imagen API
- Code Completion API
- Batch prediction API
- Tuning API
- CountTokens API

## Quick Start Usage 

#### 1. Set `default_vertex_config` on your `config.yaml`


Add the following credentials to your litellm config.yaml to use the Vertex AI endpoints.

```yaml
default_vertex_config:
  vertex_project: "adroit-crow-413218"
  vertex_location: "us-central1"
  vertex_credentials: "/Users/ishaanjaffer/Downloads/adroit-crow-413218-a956eef1a2a8.json" # Add path to service account.json
```

#### 2. Start litellm proxy

```shell
litellm --config /path/to/config.yaml
```

#### 3. Test it 

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/textembedding-gecko@001:countTokens \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{"instances":[{"content": "gm"}]}'
```
## Usage Examples

### Gemini API (Generate Content)

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.5-flash-001:generateContent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```

### Embeddings API

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/textembedding-gecko@001:predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"instances":[{"content": "gm"}]}'
```

### Imagen API

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/imagen-3.0-generate-001:predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"instances":[{"prompt": "make an otter"}], "parameters": {"sampleCount": 1}}'
```

### Count Tokens API

```shell
curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.5-flash-001:countTokens \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
```

### Tuning API 

Create Fine Tuning Job

```shell
curl http://localhost:4000/vertex-ai/tuningJobs \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer sk-1234" \
      -d '{
  "baseModel": "gemini-1.0-pro-002",
  "supervisedTuningSpec" : {
      "training_dataset_uri": "gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl"
  }
}'
```