"""HTTP helpers for Azure OpenAI image generation (REST, not SDK)."""

from typing import Final


def azure_deployment_image_generation_json_body(api_base: str, data: dict, deployment_name: str | None = None) -> dict:
    """
    Build the JSON body for Azure OpenAI image generation POSTs.

    For ``.../openai/deployments/{deployment}/images/generations``, routing uses the
    deployment in the URL only; sending ``model`` in the body (especially the deployment
    name) breaks some models (e.g. gpt-image-2). See LiteLLM #26316.

    For the v1 surface (``.../openai/v1/images/...``), Azure routes by the deployment
    name in the body ``model`` field, so the deployment name must replace any base
    model name there or Azure answers 404 DeploymentNotFound.

    Provider-style URLs (e.g. ``/providers/...`` for FLUX on Azure AI) keep all keys
    so non–OpenAI-deployment payloads still work.
    """
    drop_model: Final = "images/generations" in api_base and "/openai/deployments/" in api_base
    v1_route: Final = "/openai/v1/images/" in api_base and bool(deployment_name)
    if not drop_model and not v1_route:
        return data
    entries: Final = (
        tuple((k, v) for k, v in data.items() if k != "model")
        if drop_model
        else (*data.items(), ("model", deployment_name))
    )
    return {k: v for k, v in entries}
