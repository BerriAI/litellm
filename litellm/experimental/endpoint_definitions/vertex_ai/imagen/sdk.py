"""
Vertex AI Imagen SDK - Passthrough with Model Routing.

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
    )
"""

import json
import os
from typing import Any, Dict, List, Optional

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
    "imagen-3.0-capability-001",
}


def _load_definition() -> EndpointDefinition:
    """Load the endpoint definition from the JSON file."""
    definition_path = os.path.join(os.path.dirname(__file__), "definition.json")
    with open(definition_path) as f:
        return EndpointDefinition(**json.load(f))


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
    When model is another provider (e.g., "gpt-image-1"), routes through litellm.
    
    Args:
        instances: List of instance objects, e.g. [{"prompt": "A cat"}]
        parameters: Optional parameters dict, e.g. {"sampleCount": 2}
        model: Model to use. Imagen models route to Vertex AI, others to litellm.
        
    Returns:
        Imagen-style response: {"predictions": [{"bytesBase64Encoded": "...", "mimeType": "..."}]}
    """
    if not _is_imagen_model(model):
        return call_litellm_image_generation(
            model=model, instances=instances, parameters=parameters, timeout=timeout, **kwargs
        )
    
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
    return _handler.execute_sync(
        operation_name="generate",
        extra_headers=extra_headers,
        timeout=timeout,
        project_id=project_id,
        region=region,
        model=model,
        vertex_credentials=vertex_credentials,
        **body,
    )


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
    if not _is_imagen_model(model):
        return await acall_litellm_image_generation(
            model=model, instances=instances, parameters=parameters, timeout=timeout, **kwargs
        )
    
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
    return await _handler.execute_async(
        operation_name="generate",
        extra_headers=extra_headers,
        timeout=timeout,
        project_id=project_id,
        region=region,
        model=model,
        vertex_credentials=vertex_credentials,
        **body,
    )


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
    
    Args:
        instances: List containing prompt, image, and optional mask
        parameters: Optional parameters dict
        
    Returns:
        Imagen-style response: {"predictions": [{"bytesBase64Encoded": "...", "mimeType": "..."}]}
    """
    body: Dict[str, Any] = {"instances": instances}
    if parameters:
        body["parameters"] = parameters
    
    return _handler.execute_sync(
        operation_name="edit",
        extra_headers=extra_headers,
        timeout=timeout,
        project_id=project_id,
        region=region,
        model=model,
        vertex_credentials=vertex_credentials,
        **body,
    )


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
    
    return await _handler.execute_async(
        operation_name="edit",
        extra_headers=extra_headers,
        timeout=timeout,
        project_id=project_id,
        region=region,
        model=model,
        vertex_credentials=vertex_credentials,
        **body,
    )
