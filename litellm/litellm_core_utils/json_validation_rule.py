import json
from typing import Any, Dict, List, Union

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH


def normalize_json_schema_types(schema: Union[Dict[str, Any], List[Any], Any], depth: int = 0, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH) -> Union[Dict[str, Any], List[Any], Any]:
    """
    Normalize JSON schema types from uppercase to lowercase format.
    
    Some providers (like certain Google services) use uppercase types like 'BOOLEAN', 'STRING', 'ARRAY', 'OBJECT'
    but standard JSON Schema requires lowercase: 'boolean', 'string', 'array', 'object'
    
    This function recursively normalizes all type fields in a schema to lowercase.
    
    Args:
        schema: The schema to normalize (dict, list, or other)
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        The normalized schema with lowercase types
    """
    # Prevent infinite recursion
    if depth >= max_depth:
        return schema
        
    if not isinstance(schema, (dict, list)):
        return schema
        
    # Type mapping from uppercase to lowercase
    type_mapping = {
        'BOOLEAN': 'boolean',
        'STRING': 'string', 
        'ARRAY': 'array',
        'OBJECT': 'object',
        'NUMBER': 'number',
        'INTEGER': 'integer',
        'NULL': 'null'
    }
    
    if isinstance(schema, list):
        return [normalize_json_schema_types(item, depth + 1, max_depth) for item in schema]
    
    if isinstance(schema, dict):
        normalized_schema: Dict[str, Any] = {}
        
        for key, value in schema.items():
            if key == 'type' and isinstance(value, str) and value in type_mapping:
                normalized_schema[key] = type_mapping[value]
            elif key == 'properties' and isinstance(value, dict):
                # Recursively normalize properties
                normalized_schema[key] = {
                    prop_key: normalize_json_schema_types(prop_value, depth + 1, max_depth)
                    for prop_key, prop_value in value.items()
                }
            elif key == 'items' and isinstance(value, (dict, list)):
                # Recursively normalize array items
                normalized_schema[key] = normalize_json_schema_types(value, depth + 1, max_depth)
            elif isinstance(value, (dict, list)):
                # Recursively normalize any nested dict or list
                normalized_schema[key] = normalize_json_schema_types(value, depth + 1, max_depth)
            else:
                normalized_schema[key] = value
                
        return normalized_schema
    
    return schema


def normalize_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a tool's parameter schema to use standard JSON Schema lowercase types.
    
    Args:
        tool: The tool definition containing function parameters
        
    Returns:
        The tool with normalized schema types
    """
    if not isinstance(tool, dict):
        return tool
        
    normalized_tool = tool.copy()
    
    # Normalize function parameters if present
    if 'function' in tool and isinstance(tool['function'], dict):
        normalized_tool['function'] = tool['function'].copy()
        if 'parameters' in tool['function']:
            normalized_tool['function']['parameters'] = normalize_json_schema_types(
                tool['function']['parameters']
            )
    
    return normalized_tool


def validate_schema(schema: dict, response: str):
    """
    Validate if the returned json response follows the schema.

    Params:
    - schema - dict: JSON schema
    - response - str: Received json response as string.
    """
    from jsonschema import ValidationError, validate

    from litellm import JSONSchemaValidationError

    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError:
        raise JSONSchemaValidationError(
            model="", llm_provider="", raw_response=response, schema=json.dumps(schema)
        )

    try:
        validate(response_dict, schema=schema)
    except ValidationError:
        raise JSONSchemaValidationError(
            model="", llm_provider="", raw_response=response, schema=json.dumps(schema)
        )
