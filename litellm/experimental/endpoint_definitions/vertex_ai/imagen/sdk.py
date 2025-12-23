"""
Vertex AI Imagen SDK - True Passthrough.

The SDK matches the Vertex AI API exactly. Users pass the same JSON body
they would pass to the API directly.

Usage:
    from litellm.experimental.endpoint_definitions.vertex_ai.imagen import (
        generate_image,
        agenerate_image,
    )
    
    # Pass exact same body as Vertex AI API
    response = generate_image(
        instances=[{"prompt": "A photorealistic image of a cat"}],
        parameters={"sampleCount": 2},
    )
    
    # Response is exact same as Vertex AI API
    for prediction in response["predictions"]:
        image_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
"""

import json
import os
from typing import Any, Dict, List, Optional, cast

from litellm.experimental.endpoint_definitions.generic_handler import (
    GenericEndpointHandler,
)
from litellm.experimental.endpoint_definitions.schema import EndpointDefinition
from litellm.experimental.endpoint_definitions.vertex_ai.imagen.hooks import (
    VertexAIImagenAuthHook,
)


def _load_definition() -> EndpointDefinition:
    """Load the endpoint definition from the JSON file in this directory."""
    definition_path = os.path.join(os.path.dirname(__file__), "definition.json")
    with open(definition_path) as f:
        data = json.load(f)
    return EndpointDefinition(**data)


# Load the endpoint definition and create handler with auth hook
_definition = _load_definition()
_hook = VertexAIImagenAuthHook()
_handler = GenericEndpointHandler(_definition, hooks=_hook)


def generate_image(
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    region: str = "us-central1",
    model: str = "imagen-3.0-generate-002",
    vertex_credentials: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate images using Vertex AI Imagen API.
    
    This is a true passthrough - the request/response format matches
    the Vertex AI API exactly.
    
    Args:
        instances: List of instance objects, e.g. [{"prompt": "A cat"}]
        parameters: Optional parameters dict, e.g. {"sampleCount": 2}
        project_id: Google Cloud project ID (optional, uses VERTEXAI_PROJECT env var if not set)
        region: Google Cloud region (default: us-central1)
        model: Imagen model version (default: imagen-3.0-generate-002)
        vertex_credentials: Optional path to credentials JSON or credentials dict
        
    Returns:
        Raw API response: {"predictions": [{"bytesBase64Encoded": "...", "mimeType": "..."}]}
    """
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
    # Imagen doesn't support streaming, so result is always Dict
    result = _handler.execute_sync(
        operation_name="generate",
        extra_headers=extra_headers,
        timeout=timeout,
        project_id=project_id,
        region=region,
        model=model,
        vertex_credentials=vertex_credentials,
        **body,
    )
    return cast(Dict[str, Any], result)


async def agenerate_image(
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    region: str = "us-central1",
    model: str = "imagen-3.0-generate-002",
    vertex_credentials: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Async version of generate_image()."""
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
    # Imagen doesn't support streaming, so result is always Dict
    result = await _handler.execute_async(
        operation_name="generate",
        extra_headers=extra_headers,
        timeout=timeout,
        project_id=project_id,
        region=region,
        model=model,
        vertex_credentials=vertex_credentials,
        **body,
    )
    return cast(Dict[str, Any], result)

