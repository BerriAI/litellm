# [BETA] Vertex AI Endpoints

## Supported APIs

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