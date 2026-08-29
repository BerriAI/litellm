No transformation is required for hosted_vllm embedding.

VLLM is a superset of OpenAI's `embedding` endpoint.

## `encoding_format`

For OpenAI-compatible embedding calls (including `openai/...` with a custom `api_base` pointing at vLLM), LiteLLM resolves `encoding_format` when it is not set on the request:

1. Explicit value on the embedding call (`encoding_format=...`).
2. Model config (`litellm_params.encoding_format` on the proxy `model_list` entry).
3. Environment variable `LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT` (e.g. in `.env` or container env).

If none of those is set, or the winning value is the literal string `none`, the field is omitted from the upstream request entirely (LiteLLM also bypasses the OpenAI SDK's own base64 default), so OpenAI-compatible servers that reject `encoding_format` keep working.

To pass provider-specific parameters, see [provider-specific params](https://docs.litellm.ai/docs/completion/provider_specific_params).
