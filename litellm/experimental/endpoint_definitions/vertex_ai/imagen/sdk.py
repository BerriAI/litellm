"""
Vertex AI Imagen SDK - True Passthrough with Model Routing.

The SDK matches the Vertex AI API exactly. Users pass the same JSON body
they would pass to the API directly.

When a non-Imagen model is specified (e.g., "gpt-image-1", "dall-e-3"),
the request is automatically transformed and routed through litellm.image_generation,
then the response is transformed back to Imagen format.

Usage:
    from litellm.experimental.endpoint_definitions.vertex_ai.imagen import (
        generate_image,
        agenerate_image,
    )
    
    # Use Vertex AI Imagen directly
    response = generate_image(
        instances=[{"prompt": "A photorealistic image of a cat"}],
        parameters={"sampleCount": 2},
    )
    
    # Or route to OpenAI GPT-Image
    response = generate_image(
        model="gpt-image-1",
        instances=[{"prompt": "A photorealistic image of a cat"}],
        parameters={"sampleCount": 2},
    )
    
    # Response is always in Imagen format
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
from litellm.experimental.endpoint_definitions.vertex_ai.imagen.transformation import (
    acall_litellm_image_generation,
    call_litellm_image_generation,
)

# Models that should be routed to Vertex AI Imagen
IMAGEN_MODELS = {
    "imagen-3.0-generate-002",
    "imagen-3.0-generate-001",
    "imagen-3.0-fast-generate-001",
    "imagegeneration@006",
    "imagegeneration@005",
    "imagegeneration@002",
}


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


def _is_imagen_model(model: str) -> bool:
    """Check if the model should be routed to Vertex AI Imagen."""
    return model in IMAGEN_MODELS or model.startswith("imagen-")


def generate_image(
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    region: str = "us-central1",
    model: str = "imagen-3.0-generate-002",
    vertex_credentials: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate images using Imagen-style API.
    
    When model is an Imagen model (default), calls Vertex AI directly.
    When model is another provider (e.g., "gpt-image-1", "dall-e-3"),
    routes through litellm.image_generation and transforms the response.
    
    Args:
        instances: List of instance objects, e.g. [{"prompt": "A cat"}]
        parameters: Optional parameters dict, e.g. {"sampleCount": 2}
        project_id: Google Cloud project ID (for Imagen models)
        region: Google Cloud region (for Imagen models, default: us-central1)
        model: Model to use. Imagen models route to Vertex AI, others to litellm.
               Examples: "imagen-3.0-generate-002", "gpt-image-1", "dall-e-3"
        vertex_credentials: Optional path to credentials JSON (for Imagen models)
        **kwargs: Additional kwargs passed to litellm.image_generation (for non-Imagen models)
        
    Returns:
        Imagen-style response: {"predictions": [{"bytesBase64Encoded": "...", "mimeType": "..."}]}
    """
    # Route to litellm.image_generation for non-Imagen models
    if not _is_imagen_model(model):
        return call_litellm_image_generation(
            model=model,
            instances=instances,
            parameters=parameters,
            timeout=timeout,
            **kwargs,
        )
    
    # Use Vertex AI Imagen directly
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
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
    **kwargs,
) -> Dict[str, Any]:
    """Async version of generate_image()."""
    # Route to litellm.aimage_generation for non-Imagen models
    if not _is_imagen_model(model):
        return await acall_litellm_image_generation(
            model=model,
            instances=instances,
            parameters=parameters,
            timeout=timeout,
            **kwargs,
        )
    
    # Use Vertex AI Imagen directly
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
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


def edit_image(
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    region: str = "us-central1",
    model: str = "imagen-3.0-capability-001",
    vertex_credentials: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Edit images using Vertex AI Imagen API.
    
    This is a true passthrough - the request/response format matches
    the Vertex AI API exactly.
    
    Args:
        instances: List of instance objects containing:
            - prompt: Text prompt describing the desired edit
            - image: {"bytesBase64Encoded": "..."} - The source image
            - mask: {"bytesBase64Encoded": "..."} - Optional mask (white=edit, black=preserve)
        parameters: Optional parameters dict, e.g. {"sampleCount": 2, "editConfig": {"editMode": "inpaint-insert"}}
        project_id: Google Cloud project ID (optional, uses VERTEXAI_PROJECT env var if not set)
        region: Google Cloud region (default: us-central1)
        model: Imagen model version (default: imagen-3.0-capability-001)
        vertex_credentials: Optional path to credentials JSON or credentials dict
        
    Returns:
        Raw API response: {"predictions": [{"bytesBase64Encoded": "...", "mimeType": "..."}]}
    """
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
    result = _handler.execute_sync(
        operation_name="edit",
        extra_headers=extra_headers,
        timeout=timeout,
        project_id=project_id,
        region=region,
        model=model,
        vertex_credentials=vertex_credentials,
        **body,
    )
    return cast(Dict[str, Any], result)


async def aedit_image(
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    region: str = "us-central1",
    model: str = "imagen-3.0-capability-001",
    vertex_credentials: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Async version of edit_image()."""
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
    result = await _handler.execute_async(
        operation_name="edit",
        extra_headers=extra_headers,
        timeout=timeout,
        project_id=project_id,
        region=region,
        model=model,
        vertex_credentials=vertex_credentials,
        **body,
    )
    return cast(Dict[str, Any], result)

