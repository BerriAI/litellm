# rules

- Claude on Azure AI Foundry: reuses `anthropic/messages` shaping with the Azure `/anthropic` path
- Auth follows Python's `BaseAzureLLM._base_validate_azure_environment` order: the call's key, `litellm.api_key`, `litellm.azure_key`, `AZURE_OPENAI_API_KEY`, `AZURE_API_KEY`, each skipped only when empty, sent as `x-api-key`
- Without a key the config walks Python's `get_azure_ad_token` order, reading every setting from the call and then the secret lookup. A static token, the caller's provider and client secret run natively (`litellm-auth-azure`). Username and password, OIDC references, `os.environ/` references, an uncallable provider and token refresh go to the host's `get_azure_ad_token`, so Python orders and mints those. No token sends the call unauthenticated
- Deliberate differences from Python: a caller `x-api-key`, non-blank bearer or `api-key` in any casing replaces the deployment's key instead of being sent beside it, and a caller `api-key` is sent as `x-api-key`

# references

- https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/how-to/use-foundry-models-claude
