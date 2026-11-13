# rules

- Claude on Azure AI Foundry: reuses `anthropic/messages` shaping with the Azure `/anthropic` path
- Auth follows Python's `BaseAzureLLM._base_validate_azure_environment` order: the call's key, `litellm.api_key`, `litellm.azure_key`, `AZURE_OPENAI_API_KEY`, `AZURE_API_KEY`, each skipped only when empty, sent as `x-api-key`
- Without a key the config walks Python's `get_azure_ad_token` order, reading every setting from the call and then where Python reads it: `AZURE_AD_TOKEN` through the secret lookup (`get_secret_str`, deferred through `deferred_secret_names`), the other Entra variables from the process environment (`os.getenv`), all only when no key or caller credential is present. A static token, the caller's provider and client secret run natively (`litellm-auth-azure`). Username and password, OIDC references, `os.environ/` references, an uncallable provider and token refresh go to the host's `get_azure_ad_token`, so Python orders and mints those. No token sends the call unauthenticated
- The base is Python's `api_base or get_secret_str("AZURE_API_BASE")`: `AZURE_API_BASE` is a deferred secret read only when the call's base is empty, and the URL is cut at `/anthropic` in the path only, untrimmed, as `azure_anthropic_messages_url` does
- Deliberate differences from Python, approved by the owner: a caller `x-api-key`, `api-key` in any casing (sent as `x-api-key`) or `Authorization` other than a blank bearer is the only credential sent. No deployment key and no Entra ID token goes beside or over it, and the Entra ID path is never run

# references

- https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/how-to/use-foundry-models-claude
