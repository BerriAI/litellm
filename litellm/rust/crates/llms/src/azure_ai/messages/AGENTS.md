# rules

- Claude on Azure AI Foundry: reuses `anthropic/messages` shaping with the Azure `/anthropic` path
- Auth follows Python's `BaseAzureLLM._base_validate_azure_environment` order: the call's key, `litellm.api_key`, `litellm.azure_key`, `AZURE_OPENAI_API_KEY`, `AZURE_API_KEY`, each skipped only when empty, sent as `x-api-key`. Without a key an Entra ID token from `litellm-auth-azure` is sent as a bearer, and without one the call goes out unauthenticated
- Deliberate differences from Python: a caller `x-api-key` or non-blank bearer replaces the deployment's key instead of being sent beside it
- Python's `azure_username` / `azure_password` token flow has no native counterpart yet

# references

- https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/how-to/use-foundry-models-claude
