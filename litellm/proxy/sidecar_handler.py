"""
Sidecar request handler.

Translates between LiteLLM's internal data format and the sidecar's
HTTP forwarding protocol. Returns ModelResponse objects compatible
with the rest of the proxy pipeline.
"""

import json
from typing import Optional

from litellm.proxy.sidecar_client import get_sidecar_client
from litellm.types.utils import ModelResponse


def _extract_provider_info(data: dict, deployment: Optional[dict] = None) -> dict:
    """Extract provider URL, API key, and other forwarding metadata from request data."""
    info = {
        "api_base": "",
        "api_key": "",
        "timeout": 300,
        "stream": False,
        "model": "",
    }

    if deployment:
        litellm_params = deployment.get("litellm_params", {})
        info["api_base"] = litellm_params.get("api_base", "")
        info["api_key"] = litellm_params.get("api_key", "")
        info["timeout"] = litellm_params.get("timeout", 300)
        info["model"] = litellm_params.get("model", "")

    # Override with request-level data
    if "api_base" in data:
        info["api_base"] = data["api_base"]
    if "api_key" in data:
        info["api_key"] = data["api_key"]
    if "timeout" in data:
        info["timeout"] = data["timeout"]
    if "stream" in data:
        info["stream"] = data["stream"]
    if "model" in data:
        info["model"] = data["model"]

    return info


async def sidecar_acompletion(data: dict, deployment: dict) -> ModelResponse:
    """
    Forward a chat completion request through the sidecar.

    Returns a ModelResponse compatible with litellm's response format.
    """
    client = get_sidecar_client()
    if client is None or not client.is_healthy:
        raise RuntimeError("Sidecar not available")

    provider_info = _extract_provider_info(data, deployment)

    # Build the request body (what the provider expects)
    request_body = {
        "model": provider_info["model"].split("/", 1)[-1]
        if "/" in provider_info["model"]
        else provider_info["model"],
        "messages": data.get("messages", []),
    }

    # Forward optional params
    for key in [
        "max_tokens",
        "temperature",
        "top_p",
        "n",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "response_format",
        "seed",
        "tools",
        "tool_choice",
        "stream",
    ]:
        if key in data and data[key] is not None:
            request_body[key] = data[key]

    timeout = provider_info["timeout"]
    if isinstance(timeout, (int, float)):
        timeout = int(timeout)
    else:
        timeout = 300

    resp = await client.forward_request(
        provider_url=provider_info["api_base"],
        api_key=provider_info["api_key"],
        request_body=request_body,
        path="/v1/chat/completions",
        timeout=timeout,
        stream=False,
    )

    resp_body = await resp.read()
    resp_json = json.loads(resp_body)

    if resp.status != 200:
        raise Exception(
            f"Sidecar forwarding failed with status {resp.status}: {resp_json}"
        )

    # Convert to ModelResponse
    model_response = ModelResponse(**resp_json)
    return model_response


def is_sidecar_enabled() -> bool:
    """Check if the sidecar is enabled and healthy."""
    client = get_sidecar_client()
    return client is not None and client.is_healthy
