#!/usr/bin/env python3
"""
Simple demonstration of OpenAPI 3.1.0 to 3.0.3 transformation.
This script can run without any dependencies to show the transformation logic.
"""

import json
import copy


def _process_schema_object(obj):
    """
    Simplified version of the transformation for demonstration.
    The real implementation in openapi_downgrade.py is more comprehensive.
    """
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [_process_schema_object(item) for item in obj]
        return obj
    
    result = {}
    
    # Handle type arrays (main 3.1.0 feature)
    if "type" in obj:
        type_value = obj["type"]
        
        if isinstance(type_value, list):
            # Type array - convert to nullable
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
        else:
            # Single type
            result["type"] = type_value
    
    # Process other keys
    for key, value in obj.items():
        if key == "type":
            continue
        
        # Convert examples to example
        if key == "examples" and isinstance(value, list):
            if value:
                result["example"] = value[0]
            continue
        
        # Recursively process nested structures
        if key == "properties" and isinstance(value, dict):
            result[key] = {k: _process_schema_object(v) for k, v in value.items()}
        elif isinstance(value, dict):
            result[key] = _process_schema_object(value)
        elif isinstance(value, list):
            result[key] = [_process_schema_object(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    
    return result


def demo_transformation():
    """Demonstrate the transformation with examples."""
    
    print("=" * 70)
    print("OpenAPI 3.1.0 → 3.0.3 Transformation Demo")
    print("=" * 70)
    print()
    
    # Example 1: Type arrays
    print("Example 1: Type Array Conversion")
    print("-" * 70)
    input_schema = {
        "type": ["string", "null"],
        "description": "Optional string field"
    }
    print("Input (3.1.0):")
    print(json.dumps(input_schema, indent=2))
    
    output_schema = _process_schema_object(input_schema)
    print("\nOutput (3.0.3):")
    print(json.dumps(output_schema, indent=2))
    print()
    
    # Example 2: Examples conversion
    print("Example 2: Examples → Example Conversion")
    print("-" * 70)
    input_schema = {
        "type": "string",
        "examples": ["user", "assistant", "system"]
    }
    print("Input (3.1.0):")
    print(json.dumps(input_schema, indent=2))
    
    output_schema = _process_schema_object(input_schema)
    print("\nOutput (3.0.3):")
    print(json.dumps(output_schema, indent=2))
    print()
    
    # Example 3: Complex nested schema
    print("Example 3: Complex Nested Schema")
    print("-" * 70)
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "examples": ["John Doe"]
            },
            "email": {
                "type": ["string", "null"],
                "description": "Optional email"
            },
            "age": {
                "type": ["integer", "null"],
                "examples": [25, 30, 35]
            }
        },
        "required": ["name"]
    }
    print("Input (3.1.0):")
    print(json.dumps(input_schema, indent=2))
    
    output_schema = _process_schema_object(input_schema)
    print("\nOutput (3.0.3):")
    print(json.dumps(output_schema, indent=2))
    print()
    
    # Example 4: Multiple types (non-null)
    print("Example 4: Multiple Non-Null Types")
    print("-" * 70)
    input_schema = {
        "type": ["string", "number"],
        "description": "Can be string or number"
    }
    print("Input (3.1.0):")
    print(json.dumps(input_schema, indent=2))
    
    output_schema = _process_schema_object(input_schema)
    print("\nOutput (3.0.3):")
    print(json.dumps(output_schema, indent=2))
    print()
    
    # Example 5: LLM Chat Message Schema
    print("Example 5: Realistic LLM Chat Message Schema")
    print("-" * 70)
    input_schema = {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "examples": ["user", "assistant", "system"]
            },
            "content": {
                "type": ["string", "null"],
                "description": "Message content"
            },
            "name": {
                "type": ["string", "null"]
            },
            "function_call": {
                "type": ["object", "null"],
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "string"}
                }
            }
        },
        "required": ["role"]
    }
    print("Input (3.1.0):")
    print(json.dumps(input_schema, indent=2))
    
    output_schema = _process_schema_object(input_schema)
    print("\nOutput (3.0.3):")
    print(json.dumps(output_schema, indent=2))
    print()
    
    print("=" * 70)
    print("✓ Transformation Demo Complete")
    print("=" * 70)
    print()
    print("Key Transformations Applied:")
    print("  • Type arrays → nullable property")
    print("  • examples (array) → example (single value)")
    print("  • Multiple non-null types → oneOf")
    print("  • Nested schemas processed recursively")
    print()
    print("Full implementation: litellm/proxy/common_utils/openapi_downgrade.py")
    print("Tests: tests/test_litellm/proxy/common_utils/test_openapi_downgrade.py")
    print()


if __name__ == "__main__":
    demo_transformation()
