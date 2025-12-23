"""
Bridge Engine - Generic transformation layer for SDK bridges.

Reads JSON bridge definitions to transform requests/responses between
different API formats (e.g., Vertex AI Imagen <-> LiteLLM image_generation).

This makes it easy to add new SDK methods by defining transformations in JSON
rather than writing custom Python code.
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
        self.description = data.get("description", "")
        self.request_transform = data.get("request_transform", {})
        self.response_transform = data.get("response_transform", {})
        self.model_config = data.get("model_config", {})


def _get_nested_value(obj: Any, path: str) -> Any:
    """
    Get a nested value from an object using dot notation.
    
    Supports:
    - Simple paths: "prompt", "parameters.sampleCount"
    - Array access: "instances[0].prompt", "data[*].b64_json"
    
    Args:
        obj: The object to extract from
        path: Dot-notation path like "instances[0].prompt"
    
    Returns:
        The extracted value, or None if not found
    """
    if obj is None:
        return None

    parts = re.split(r"\.(?![^\[]*\])", path)  # Split on dots not inside brackets
    current = obj

    for i, part in enumerate(parts):
        if current is None:
            return None

        # Check for array index
        match = re.match(r"(\w+)\[(\d+|\*)\]", part)
        if match:
            key, index = match.groups()
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return None

            if current is None:
                return None

            if index == "*":
                # For [*], extract the remaining path from each item
                if isinstance(current, list):
                    remaining_parts = parts[i + 1:]
                    if remaining_parts:
                        # There are more parts after [*], extract from each item
                        remaining_path = ".".join(remaining_parts)
                        return [_get_nested_value(item, remaining_path) for item in current]
                    else:
                        # No more parts, return the array itself
                        return current
                return None
            else:
                # Return specific index
                idx = int(index)
                if isinstance(current, list) and len(current) > idx:
                    current = current[idx]
                else:
                    return None
        else:
            # Simple key access
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

    return current


def _set_nested_value(obj: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a nested value in an object using dot notation.
    
    Args:
        obj: The object to modify
        path: Dot-notation path like "predictions[0].bytesBase64Encoded"
        value: The value to set
    """
    parts = re.split(r"\.(?![^\[]*\])", path)
    current = obj

    for i, part in enumerate(parts[:-1]):
        match = re.match(r"(\w+)\[(\d+|\*)\]", part)
        if match:
            key, index = match.groups()
            if key not in current:
                current[key] = [] if index in ("*", "0") else {}
            current = current[key]
            if index != "*" and isinstance(current, list):
                idx = int(index)
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
        else:
            if part not in current:
                # Look ahead to see if next part is an array
                next_part = parts[i + 1] if i + 1 < len(parts) else ""
                if "[" in next_part:
                    current[part] = []
                else:
                    current[part] = {}
            current = current[part]

    # Set the final value
    last_part = parts[-1]
    match = re.match(r"(\w+)\[(\d+|\*)\]", last_part)
    if match:
        key, index = match.groups()
        if key not in current:
            current[key] = []
        if index == "*":
            # Setting all items - value should be a list
            current[key] = value if isinstance(value, list) else [value]
        else:
            idx = int(index)
            while len(current[key]) <= idx:
                current[key].append(None)
            current[key][idx] = value
    else:
        current[last_part] = value


class BridgeEngine:
    """
    Engine that applies bridge transformations.
    
    Usage:
        engine = BridgeEngine.from_json("path/to/bridge.json")
        litellm_params = engine.transform_request(imagen_request)
        imagen_response = engine.transform_response(litellm_response)
    """

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeEngine":
        """Create a bridge engine from a dictionary."""
        return cls(BridgeDefinition(data))

    def _register_builtin_transforms(self) -> None:
        """Register built-in transform functions."""
        # Register transforms defined in the JSON
        transforms = self.definition.request_transform.get("transforms", {})
        for name, config in transforms.items():
            if config.get("type") == "map":
                mapping = config.get("mapping", {})
                default = config.get("default")
                self._transforms[name] = lambda v, m=mapping, d=default: m.get(v, d)

    def transform_request(
        self,
        source_data: Dict[str, Any],
        **extra_kwargs,
    ) -> Dict[str, Any]:
        """
        Transform a request from source format to target format.
        
        Args:
            source_data: The source request data (e.g., Imagen format)
            **extra_kwargs: Additional kwargs to include in the result
        
        Returns:
            Transformed request data (e.g., LiteLLM format)
        """
        result: Dict[str, Any] = {}
        mappings = self.definition.request_transform.get("field_mappings", [])

        for mapping in mappings:
            source_path = mapping["source"]
            target_path = mapping["target"]
            required = mapping.get("required", False)
            default = mapping.get("default")
            transform_name = mapping.get("transform")

            # Extract value from source
            value = _get_nested_value(source_data, source_path)

            # Apply transform if specified
            if value is not None and transform_name:
                transform_fn = self._transforms.get(transform_name)
                if transform_fn:
                    value = transform_fn(value)

            # Use default if value is None
            if value is None and default is not None:
                value = default

            # Check required
            if value is None and required:
                raise ValueError(f"Required field '{source_path}' is missing")

            # Set in result if we have a value
            if value is not None:
                _set_nested_value(result, target_path, value)

        # Add extra kwargs
        result.update(extra_kwargs)

        return result

    def transform_response(
        self,
        source_response: Any,
    ) -> Dict[str, Any]:
        """
        Transform a response from source format to target format.
        
        Args:
            source_response: The source response (e.g., LiteLLM ImageResponse)
        
        Returns:
            Transformed response data (e.g., Imagen format)
        """
        result: Dict[str, Any] = {}
        mappings = self.definition.response_transform.get("field_mappings", [])
        static_fields = self.definition.response_transform.get("static_fields", {})

        # Process field mappings
        for mapping in mappings:
            source_path = mapping["source"]
            target_path = mapping["target"]

            # Check if this is an array mapping (contains [*])
            if "[*]" in source_path:
                # Extract the array
                source_items = _get_nested_value(source_response, source_path)
                if source_items:
                    # Get the target array key
                    target_array_match = re.match(r"(\w+)\[\*\]", target_path)
                    if target_array_match:
                        target_array_key = target_array_match.group(1)
                        target_field = target_path.split("[*].")[-1] if "[*]." in target_path else None

                        if target_array_key not in result:
                            result[target_array_key] = []

                        # Ensure we have enough items in the target array
                        while len(result[target_array_key]) < len(source_items):
                            result[target_array_key].append({})

                        # Set values
                        for i, item in enumerate(source_items):
                            if item is not None and target_field:
                                result[target_array_key][i][target_field] = item
            else:
                # Simple mapping
                value = _get_nested_value(source_response, source_path)
                if value is not None:
                    _set_nested_value(result, target_path, value)

        # Apply static fields
        for target_path, value in static_fields.items():
            if "[*]" in target_path:
                # Apply to all items in array
                target_array_match = re.match(r"(\w+)\[\*\]", target_path)
                if target_array_match:
                    target_array_key = target_array_match.group(1)
                    target_field = target_path.split("[*].")[-1] if "[*]." in target_path else None

                    if target_array_key in result and target_field:
                        for item in result[target_array_key]:
                            item[target_field] = value
            else:
                _set_nested_value(result, target_path, value)

        return result

    def supports_response_format(self, model: str) -> bool:
        """Check if a model supports response_format parameter."""
        supported = self.definition.model_config.get("supports_response_format", [])
        return model in supported

    def get_default_response_format(self) -> Optional[str]:
        """Get the default response format for models that support it."""
        return self.definition.model_config.get("default_response_format")


def call_litellm_with_bridge(
    bridge: BridgeEngine,
    model: str,
    source_request: Dict[str, Any],
    litellm_method: str = "image_generation",
    **kwargs,
) -> Dict[str, Any]:
    """
    Call a LiteLLM method using a bridge transformation.
    
    Args:
        bridge: The bridge engine to use
        model: The model to call
        source_request: Request in source format (e.g., Imagen)
        litellm_method: The LiteLLM method to call
        **kwargs: Additional kwargs for the LiteLLM call
    
    Returns:
        Response in target format (e.g., Imagen)
    """
    # Transform request
    litellm_params = bridge.transform_request(source_request)

    # Build call kwargs
    call_kwargs = {
        "model": model,
        **litellm_params,
        **kwargs,
    }

    # Add response_format if supported
    if bridge.supports_response_format(model):
        response_format = bridge.get_default_response_format()
        if response_format:
            call_kwargs["response_format"] = response_format

    # Get the LiteLLM method
    method = getattr(litellm, litellm_method)

    # Call LiteLLM
    response = method(**call_kwargs)

    # Transform response
    return bridge.transform_response(response)


async def acall_litellm_with_bridge(
    bridge: BridgeEngine,
    model: str,
    source_request: Dict[str, Any],
    litellm_method: str = "aimage_generation",
    **kwargs,
) -> Dict[str, Any]:
    """Async version of call_litellm_with_bridge."""
    # Transform request
    litellm_params = bridge.transform_request(source_request)

    # Build call kwargs
    call_kwargs = {
        "model": model,
        **litellm_params,
        **kwargs,
    }

    # Add response_format if supported
    if bridge.supports_response_format(model):
        response_format = bridge.get_default_response_format()
        if response_format:
            call_kwargs["response_format"] = response_format

    # Get the LiteLLM method
    method = getattr(litellm, litellm_method)

    # Call LiteLLM
    response = await method(**call_kwargs)

    # Transform response
    return bridge.transform_response(response)

