# /v1/messages

This folder contains transformation logic for calling bedrock models in the Anthropic /v1/messages API spec.

## Important Notes

- The `anthropic_beta` header is only forwarded to **Claude models** (models with provider "anthropic")
- Non-Claude models (Qwen, Nova, etc.) will have this header stripped automatically
- This prevents "unknown variant: anthropic_beta" errors on models that don't support Anthropic beta features
- Provider detection is performed using `BedrockLLM.get_bedrock_invoke_provider()` for robust model identification