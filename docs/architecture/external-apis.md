# External APIs

## OpenAI API
- **Purpose:** Primary LLM provider for GPT models
- **Documentation:** https://platform.openai.com/docs
- **Base URL(s):** https://api.openai.com/v1
- **Authentication:** Bearer token (API key)
- **Rate Limits:** Varies by tier and model

**Key Endpoints Used:**
- `POST /chat/completions` - Chat completions
- `POST /embeddings` - Text embeddings
- `POST /audio/transcriptions` - Speech to text

**Integration Notes:** Native OpenAI format, no translation needed

## Anthropic API
- **Purpose:** Claude model provider
- **Documentation:** https://docs.anthropic.com
- **Base URL(s):** https://api.anthropic.com
- **Authentication:** X-API-Key header
- **Rate Limits:** Varies by tier

**Key Endpoints Used:**
- `POST /messages` - Claude chat completions
- `POST /complete` - Legacy completions

**Integration Notes:** Requires message format translation

## Azure OpenAI API
- **Purpose:** Enterprise Azure-hosted OpenAI models
- **Documentation:** https://docs.microsoft.com/azure/cognitive-services/openai
- **Base URL(s):** https://{resource}.openai.azure.com
- **Authentication:** api-key header or Azure AD
- **Rate Limits:** Configured per deployment

**Key Endpoints Used:**
- `POST /openai/deployments/{deployment}/chat/completions` - Chat completions
- `POST /openai/deployments/{deployment}/embeddings` - Embeddings

**Integration Notes:** Requires deployment name mapping

## Lago Billing API
- **Purpose:** Usage-based billing and metering
- **Documentation:** Internal/Moneta documentation
- **Base URL(s):** Configured per environment
- **Authentication:** Bearer token
- **Rate Limits:** Standard API limits

**Key Endpoints Used:**
- `POST /events` - Usage event reporting
- `GET /subscriptions/{id}/usage` - Usage retrieval
- `GET /customers/{id}/balance` - Balance checking

**Integration Notes:** Async event batching for performance
