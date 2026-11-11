# Azure AI

## Native Cohere embeddings on Microsoft Foundry

Use `azure_ai/Cohere-Embed-V5-Fast` or `azure_ai/Cohere-Embed-V5-Pro` with the
Foundry resource URL. LiteLLM sends the native Cohere request to
`/providers/cohere/v2/embed` and calculates cost using the existing Azure AI
pricing entries and the provider's billed text and image tokens.

```python
import os

from litellm import embedding

response = embedding(
    model="azure_ai/Cohere-Embed-V5-Fast",
    api_base="https://my-resource.services.ai.azure.com",
    api_key=os.environ["AZURE_AI_API_KEY"],
    input=["Searchable document text"],
    input_type="search_document",
    dimensions=256,
)
```

The same configuration works with `aembedding`. Azure authentication also
supports Entra access tokens, service principals, OIDC federation and managed
identity through the existing Azure credential options. Omit the API key when
using those credentials.

The API base can be the resource root, `/models`, `/providers/cohere`,
`/providers/cohere/v2`, or the complete `/providers/cohere/v2/embed` endpoint.
For a custom deployment name, set the API base to one of the explicit
`/providers/cohere` paths. Query parameters are preserved.

Existing generic Azure embeddings and legacy Cohere v3/v4 deployments retain
their current endpoints unless an explicit native Cohere API base is supplied.

The native endpoint format is documented in
[Cohere's Azure examples](https://github.com/cohere-ai/cohere-azure-workshops).
