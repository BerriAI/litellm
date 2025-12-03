"""
OpenAPI 3.1.0 to 3.0.3 schema downgrade utility.

This module provides functions to transform OpenAPI 3.1.0 schemas to be compatible
with OpenAPI 3.0.3 specification. This is needed for integration with tools like
Apigee that require OpenAPI 3.0.3 compatibility.

Key transformations:
- Type arrays to nullable + single type
- examples (array) to example (single value)
- Remove 3.1.0-specific keywords
- Handle JSON Schema Draft 2020-12 features
"""

import copy
from typing import Any, Dict, List, Union

from litellm._logging import verbose_proxy_logger


def downgrade_openapi_schema_to_3_0_3(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform OpenAPI 3.1.0 schema to 3.0.3 compatible schema.
    
    This function performs a deep transformation of the entire OpenAPI schema,
    converting all 3.1.0-specific features to their 3.0.3 equivalents or removing
    them if no equivalent exists.
    
    Args:
        schema: OpenAPI 3.1.0 schema dictionary
        
    Returns:
        OpenAPI 3.0.3 compatible schema dictionary
    """
    verbose_proxy_logger.debug("Starting OpenAPI 3.1.0 to 3.0.3 downgrade")
    
    # Deep copy to avoid modifying original
    result = copy.deepcopy(schema)
    
    # Update version string
    result["openapi"] = "3.0.3"
    
    # Remove webhooks if present (3.1.0 feature not in 3.0.3)
    if "webhooks" in result:
        verbose_proxy_logger.debug("Removing webhooks (3.1.0 feature)")
        del result["webhooks"]
    
    # Process info section
    if "info" in result:
        result["info"] = _process_info_object(result["info"])
    
    # Process paths
    if "paths" in result:
        result["paths"] = _process_paths_object(result["paths"])
    
    # Process components
    if "components" in result:
        result["components"] = _process_components_object(result["components"])
    
    verbose_proxy_logger.debug("Completed OpenAPI 3.1.0 to 3.0.3 downgrade")
    return result


def _process_info_object(info: Dict[str, Any]) -> Dict[str, Any]:
    """Process the info object to ensure 3.0.3 compatibility."""
    result = copy.deepcopy(info)
    
    # In 3.1.0, info.license.identifier was added, not supported in 3.0.3
    if "license" in result and "identifier" in result["license"]:
        verbose_proxy_logger.debug("Removing license.identifier (3.1.0 feature)")
        del result["license"]["identifier"]
    
    return result


def _process_paths_object(paths: Dict[str, Any]) -> Dict[str, Any]:
    """Process the paths object recursively."""
    result = {}
    for path, path_item in paths.items():
        result[path] = _process_path_item(path_item)
    return result


def _process_path_item(path_item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single path item (endpoint)."""
    result = {}
    
    # HTTP methods and other path item properties
    for key, value in path_item.items():
        if key in ["get", "put", "post", "delete", "options", "head", "patch", "trace"]:
            result[key] = _process_operation(value)
        elif isinstance(value, dict):
            result[key] = _process_schema_object(value)
        elif isinstance(value, list):
            result[key] = [_process_schema_object(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    
    return result


def _process_operation(operation: Dict[str, Any]) -> Dict[str, Any]:
    """Process an operation (HTTP method) object."""
    result = {}
    
    for key, value in operation.items():
        if key == "requestBody":
            result[key] = _process_request_body(value)
        elif key == "responses":
            result[key] = _process_responses(value)
        elif key == "parameters":
            result[key] = _process_parameters(value)
        elif isinstance(value, dict):
            result[key] = _process_schema_object(value)
        elif isinstance(value, list):
            result[key] = [_process_schema_object(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    
    return result


def _process_request_body(request_body: Dict[str, Any]) -> Dict[str, Any]:
    """Process a request body object."""
    result = copy.deepcopy(request_body)
    
    if "content" in result:
        result["content"] = _process_content(result["content"])
    
    return result


def _process_responses(responses: Dict[str, Any]) -> Dict[str, Any]:
    """Process responses object."""
    result = {}
    
    for status_code, response in responses.items():
        result[status_code] = _process_response(response)
    
    return result


def _process_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single response object."""
    result = copy.deepcopy(response)
    
    if "content" in result:
        result["content"] = _process_content(result["content"])
    
    return result


def _process_content(content: Dict[str, Any]) -> Dict[str, Any]:
    """Process content object (media types)."""
    result = {}
    
    for media_type, media_type_object in content.items():
        result[media_type] = copy.deepcopy(media_type_object)
        
        if "schema" in result[media_type]:
            result[media_type]["schema"] = _process_schema_object(result[media_type]["schema"])
        
        # Handle examples -> example conversion at media type level
        if "examples" in result[media_type]:
            examples = result[media_type]["examples"]
            if isinstance(examples, dict) and examples:
                # Take the first example
                first_example_key = list(examples.keys())[0]
                result[media_type]["example"] = examples[first_example_key].get("value", examples[first_example_key])
            del result[media_type]["examples"]
    
    return result


def _process_parameters(parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process parameters array."""
    result = []
    
    for param in parameters:
        processed_param = copy.deepcopy(param)
        
        # In 3.1.0, parameters can have 'content', in 3.0.3 they must use 'schema'
        if "content" in processed_param:
            verbose_proxy_logger.debug(f"Converting parameter content to schema: {processed_param.get('name', 'unknown')}")
            # Try to extract schema from content
            content = processed_param["content"]
            if content and isinstance(content, dict):
                first_media_type = list(content.keys())[0]
                if "schema" in content[first_media_type]:
                    processed_param["schema"] = content[first_media_type]["schema"]
            del processed_param["content"]
        
        if "schema" in processed_param:
            processed_param["schema"] = _process_schema_object(processed_param["schema"])
        
        # Handle examples -> example at parameter level
        if "examples" in processed_param:
            examples = processed_param["examples"]
            if isinstance(examples, dict) and examples:
                first_example_key = list(examples.keys())[0]
                processed_param["example"] = examples[first_example_key].get("value", examples[first_example_key])
            elif isinstance(examples, list) and examples:
                processed_param["example"] = examples[0]
            del processed_param["examples"]
        
        result.append(processed_param)
    
    return result


def _process_components_object(components: Dict[str, Any]) -> Dict[str, Any]:
    """Process components object."""
    result = {}
    
    for key, value in components.items():
        if key == "schemas":
            result[key] = _process_schemas(value)
        elif isinstance(value, dict):
            result[key] = {k: _process_schema_object(v) if isinstance(v, dict) else v for k, v in value.items()}
        else:
            result[key] = value
    
    return result


def _process_schemas(schemas: Dict[str, Any]) -> Dict[str, Any]:
    """Process schemas in components."""
    result = {}
    
    for schema_name, schema_def in schemas.items():
        result[schema_name] = _process_schema_object(schema_def)
    
    return result


def _process_schema_object(obj: Any) -> Any:
    """
    Recursively process a schema object to convert 3.1.0 features to 3.0.3.
    
    Main transformations:
    - Type arrays like ["string", "null"] -> {"type": "string", "nullable": true}
    - examples array -> example (single value)
    - Remove unsupported keywords: const, $dynamicRef, $dynamicAnchor, prefixItems
    - Handle nested objects and arrays recursively
    """
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [_process_schema_object(item) for item in obj]
        return obj
    
    result = {}
    
    # Handle type arrays (3.1.0 feature)
    if "type" in obj:
        type_value = obj["type"]
        
        if isinstance(type_value, list):
            # Type array - need to convert to nullable or oneOf
            null_present = "null" in type_value
            non_null_types = [t for t in type_value if t != "null"]
            
            if len(non_null_types) == 1:
                # Single type + null -> use nullable
                result["type"] = non_null_types[0]
                if null_present:
                    result["nullable"] = True
            elif len(non_null_types) > 1:
                # Multiple types -> use oneOf
                result["oneOf"] = [{"type": t} for t in non_null_types]
                if null_present:
                    result["nullable"] = True
            elif len(non_null_types) == 0 and null_present:
                # Only null type
                result["type"] = "null"
        else:
            # Single type, keep as is
            result["type"] = type_value
    
    # Process all other keys
    for key, value in obj.items():
        if key == "type":
            # Already handled above
            continue
        
        # Remove 3.1.0 specific keywords
        if key in ["const", "$dynamicRef", "$dynamicAnchor", "prefixItems", "$id", "$schema"]:
            verbose_proxy_logger.debug(f"Removing unsupported keyword: {key}")
            continue
        
        # Convert examples (array) to example (single)
        if key == "examples" and isinstance(value, list):
            if value:
                result["example"] = value[0]
            continue
        
        # Handle exclusiveMinimum/exclusiveMaximum
        # In 3.1.0 they are numbers, in 3.0.3 they are booleans with separate minimum/maximum
        if key in ["exclusiveMinimum", "exclusiveMaximum"]:
            if isinstance(value, (int, float)):
                # 3.1.0 format: exclusiveMinimum is the value itself
                # 3.0.3 format: exclusiveMinimum is boolean, minimum is the value
                base_key = "minimum" if key == "exclusiveMinimum" else "maximum"
                result[base_key] = value
                result[key] = True
                continue
        
        # Recursively process nested structures
        if key == "properties" and isinstance(value, dict):
            result[key] = {k: _process_schema_object(v) for k, v in value.items()}
        elif key == "items":
            result[key] = _process_schema_object(value)
        elif key in ["allOf", "anyOf", "oneOf"]:
            result[key] = [_process_schema_object(item) for item in value]
        elif key == "not":
            result[key] = _process_schema_object(value)
        elif key == "additionalProperties":
            if isinstance(value, dict):
                result[key] = _process_schema_object(value)
            else:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = _process_schema_object(value)
        elif isinstance(value, list):
            result[key] = [_process_schema_object(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    
    return result


def convert_pydantic_v2_to_openapi_3_0_3(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Pydantic v2 generated schema (JSON Schema 2020-12) to OpenAPI 3.0.3.
    
    Pydantic v2 generates schemas aligned with JSON Schema Draft 2020-12,
    which is what OpenAPI 3.1.0 uses. This function converts to 3.0.3.
    
    Args:
        schema: Pydantic v2 model_json_schema() output
        
    Returns:
        OpenAPI 3.0.3 compatible schema
    """
    # Process the schema object
    result = _process_schema_object(schema)
    
    # Remove $defs if present (will be in components/schemas in full OpenAPI doc)
    if "$defs" in result:
        verbose_proxy_logger.debug("Removing $defs from Pydantic schema (will be in components/schemas)")
        del result["$defs"]
    
    return result


def get_openapi_3_0_3_compatible_version(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get a 3.0.3 compatible version of an OpenAPI schema.
    
    This is the main entry point for converting schemas.
    
    Args:
        openapi_schema: OpenAPI schema (any version)
        
    Returns:
        OpenAPI 3.0.3 compatible schema
    """
    current_version = openapi_schema.get("openapi", "3.0.0")
    
    if current_version.startswith("3.1"):
        verbose_proxy_logger.info(f"Converting OpenAPI {current_version} to 3.0.3")
        return downgrade_openapi_schema_to_3_0_3(openapi_schema)
    elif current_version.startswith("3.0"):
        # Already 3.0.x, ensure it's exactly 3.0.3
        result = copy.deepcopy(openapi_schema)
        result["openapi"] = "3.0.3"
        verbose_proxy_logger.debug(f"Schema already OpenAPI 3.0.x, setting version to 3.0.3")
        return result
    else:
        verbose_proxy_logger.warning(f"Unknown OpenAPI version: {current_version}, attempting conversion")
        return downgrade_openapi_schema_to_3_0_3(openapi_schema)
