"""
Transformation layer for Imagen API.

Uses the JSON-based bridge engine for declarative transformations between:
- Google Imagen format (instances/predictions)
- LiteLLM/OpenAI image_generation format

The bridge definition is in bridge.json - edit that file to modify mappings.
"""

import os
from typing import Any, Dict, List, Optional

from litellm.experimental.endpoint_definitions.bridge_engine import (
    BridgeEngine,
    acall_litellm_with_bridge,
    call_litellm_with_bridge,
)

# Load the bridge definition from JSON
_bridge_path = os.path.join(os.path.dirname(__file__), "bridge.json")
_bridge = BridgeEngine.from_json(_bridge_path)


def imagen_request_to_litellm(
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Transform Imagen request format to LiteLLM image_generation format.
    
    Uses the bridge.json definition for field mappings.
    """
    source_data: Dict[str, Any] = {"instances": instances}
    if parameters:
        source_data["parameters"] = parameters
    return _bridge.transform_request(source_data)


def litellm_response_to_imagen(response: Any) -> Dict[str, Any]:
    """
    Transform LiteLLM ImageResponse to Imagen response format.
    
    Uses the bridge.json definition for field mappings.
    """
    return _bridge.transform_response(response)


def call_litellm_image_generation(
    model: str,
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call litellm.image_generation with Imagen-style inputs and return Imagen-style output.
    
    Args:
        model: The model to use (e.g., "gpt-image-1", "dall-e-3")
        instances: Imagen-style instances [{"prompt": "..."}]
        parameters: Imagen-style parameters {"sampleCount": N}
        **kwargs: Additional kwargs passed to litellm.image_generation
    
    Returns:
        Imagen-style response {"predictions": [...]}
    """
    source_request: Dict[str, Any] = {"instances": instances}
    if parameters:
        source_request["parameters"] = parameters

    return call_litellm_with_bridge(
        bridge=_bridge,
        model=model,
        source_request=source_request,
        litellm_method="image_generation",
        **kwargs,
    )


async def acall_litellm_image_generation(
    model: str,
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Async version of call_litellm_image_generation."""
    source_request: Dict[str, Any] = {"instances": instances}
    if parameters:
        source_request["parameters"] = parameters

    return await acall_litellm_with_bridge(
        bridge=_bridge,
        model=model,
        source_request=source_request,
        litellm_method="aimage_generation",
        **kwargs,
    )
