"""HTTP helpers for Azure OpenAI image generation (REST, not SDK)."""


def azure_deployment_image_generation_json_body(api_base: str, data: dict) -> dict:
    """
    Build the JSON body for Azure OpenAI image generation POSTs.

    For ``.../openai/deployments/{deployment}/images/generations``, routing uses the
    deployment in the URL only; sending ``model`` in the body (especially the deployment
    name) breaks some models (e.g. gpt-image-2). See LiteLLM #26316.

    Provider-style URLs (e.g. ``/providers/...`` for FLUX on Azure AI) keep all keys
    so non–OpenAI-deployment payloads still work.
    """
    if "images/generations" in api_base and "/openai/deployments/" in api_base:
        return {k: v for k, v in data.items() if k != "model"}
    return data
