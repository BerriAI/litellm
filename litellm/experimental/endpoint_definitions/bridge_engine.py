"""
Bridge Engine - Transforms requests/responses between API formats.

Reads JSON bridge definitions to transform between different API formats
(e.g., Vertex AI Imagen <-> LiteLLM image_generation).
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import litellm


class BridgeDefinition:
    """Represents a parsed bridge definition from JSON."""

    def __init__(self, data: Dict[str, Any]):
        self.bridge_name = data["bridge_name"]
        self.request_transform = data.get("request_transform", {})
        self.response_transform = data.get("response_transform", {})
        self.model_config = data.get("model_config", {})


def _get_nested_value(obj: Any, path: str) -> Any:
    """
    Get a nested value from an object using dot notation.
    
    Supports:
    - Simple paths: "prompt", "parameters.sampleCount"
    - Array access: "instances[0].prompt", "data[*].b64_json"
    """
    if obj is None:
        return None

    parts = re.split(r"\.(?![^\[]*\])", path)
    current = obj

    for i, part in enumerate(parts):
        if current is None:
            return None

        match = re.match(r"(\w+)\[(\d+|\*)\]", part)
        if match:
            key, index = match.groups()
            current = current.get(key) if isinstance(current, dict) else getattr(current, key, None)
            if current is None:
                return None

            if index == "*":
                if isinstance(current, list):
                    remaining_parts = parts[i + 1:]
                    if remaining_parts:
                        remaining_path = ".".join(remaining_parts)
                        return [_get_nested_value(item, remaining_path) for item in current]
                    return current
                return None
            else:
                idx = int(index)
                current = current[idx] if isinstance(current, list) and len(current) > idx else None
        else:
            current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)

    return current


class BridgeEngine:
    """Engine that applies bridge transformations."""

    def __init__(self, definition: BridgeDefinition):
        self.definition = definition
        self._transforms: Dict[str, Callable] = {}
        self._register_builtin_transforms()

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "BridgeEngine":
        """Load a bridge engine from a JSON file."""
        with open(json_path) as f:
            data = json.load(f)
        return cls(BridgeDefinition(data))

    def _register_builtin_transforms(self) -> None:
        """Register built-in transform functions."""
        transforms = self.definition.request_transform.get("transforms", {})
        for name, config in transforms.items():
            if config.get("type") == "map":
                mapping = config.get("mapping", {})
                default = config.get("default")
                self._transforms[name] = lambda v, m=mapping, d=default: m.get(v, d)

    def transform_request(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a request from source format to target format."""
        result: Dict[str, Any] = {}
        mappings = self.definition.request_transform.get("field_mappings", [])

        for mapping in mappings:
            source_path = mapping["source"]
            target_path = mapping["target"]
            default = mapping.get("default")
            transform_name = mapping.get("transform")

            value = _get_nested_value(source_data, source_path)

            if value is not None and transform_name:
                transform_fn = self._transforms.get(transform_name)
                if transform_fn:
                    value = transform_fn(value)

            if value is None and default is not None:
                value = default

            if value is not None:
                result[target_path] = value

        return result

    def transform_response(self, source_response: Any) -> Dict[str, Any]:
        """Transform a response from source format to target format."""
        result: Dict[str, Any] = {}
        mappings = self.definition.response_transform.get("field_mappings", [])
        static_fields = self.definition.response_transform.get("static_fields", {})

        # Track array items for building predictions
        predictions: List[Dict[str, Any]] = []

        for mapping in mappings:
            source_path = mapping["source"]
            target_path = mapping["target"]

            if "[*]" in source_path:
                source_items = _get_nested_value(source_response, source_path)
                if source_items and isinstance(source_items, list):
                    # Get target field name after [*].
                    target_field = target_path.split("[*].")[-1] if "[*]." in target_path else None
                    
                    # Ensure predictions array is big enough
                    while len(predictions) < len(source_items):
                        predictions.append({})
                    
                    # Set values
                    for i, item in enumerate(source_items):
                        if item is not None and target_field:
                            predictions[i][target_field] = item

        # Apply static fields to all prediction items
        for target_path, value in static_fields.items():
            if "[*]" in target_path:
                target_field = target_path.split("[*].")[-1] if "[*]." in target_path else None
                if target_field:
                    for item in predictions:
                        item[target_field] = value

        if predictions:
            result["predictions"] = predictions

        return result

    def supports_response_format(self, model: str) -> bool:
        """Check if a model supports response_format parameter."""
        supported = self.definition.model_config.get("supports_response_format", [])
        return model in supported


def call_litellm_with_bridge(
    bridge: BridgeEngine,
    model: str,
    source_request: Dict[str, Any],
    litellm_method: str = "image_generation",
    **kwargs,
) -> Dict[str, Any]:
    """Call a LiteLLM method using a bridge transformation."""
    litellm_params = bridge.transform_request(source_request)

    call_kwargs = {"model": model, **litellm_params, **kwargs}

    if bridge.supports_response_format(model):
        call_kwargs["response_format"] = "b64_json"

    method = getattr(litellm, litellm_method)
    response = method(**call_kwargs)

    return bridge.transform_response(response)


async def acall_litellm_with_bridge(
    bridge: BridgeEngine,
    model: str,
    source_request: Dict[str, Any],
    litellm_method: str = "aimage_generation",
    **kwargs,
) -> Dict[str, Any]:
    """Async version of call_litellm_with_bridge."""
    litellm_params = bridge.transform_request(source_request)

    call_kwargs = {"model": model, **litellm_params, **kwargs}

    if bridge.supports_response_format(model):
        call_kwargs["response_format"] = "b64_json"

    method = getattr(litellm, litellm_method)
    response = await method(**call_kwargs)

    return bridge.transform_response(response)
