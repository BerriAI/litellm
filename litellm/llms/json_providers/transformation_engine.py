"""
Request/Response transformation engine for JSON-configured providers.

Supports multiple transformation types:
- Jinja2 templates for flexible request building
- JSONPath for response field extraction
- Python functions for complex transformations
"""

import importlib
import json
from typing import Any, Dict, Optional

from litellm._logging import verbose_logger

# Optional dependencies
try:
    from jinja2 import Template as Jinja2Template
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    verbose_logger.warning("jinja2 not installed, Jinja transformations will not work. Install with: pip install jinja2")

try:
    from jsonpath_ng import parse as jsonpath_parse
    HAS_JSONPATH = True
except ImportError:
    HAS_JSONPATH = False
    verbose_logger.warning("jsonpath-ng not installed, JSONPath transformations will not work. Install with: pip install jsonpath-ng")


class TransformationEngine:
    """
    Engine for transforming requests and responses for JSON-configured providers.
    
    This enables converting between LiteLLM's standard format and provider-specific formats.
    """

    @staticmethod
    def transform_request(
        litellm_params: Dict[str, Any], transformation_config: Any
    ) -> Dict[str, Any]:
        """
        Transform LiteLLM request parameters to provider format.
        
        Args:
            litellm_params: Standard LiteLLM parameters (prompt, n, size, etc.)
            transformation_config: TransformationConfig object
        
        Returns:
            Provider-specific request body
        """
        try:
            if transformation_config.type == "jinja":
                return TransformationEngine._transform_with_jinja(
                    litellm_params, transformation_config.template, transformation_config.filters
                )
            elif transformation_config.type == "jsonpath":
                # JSONPath can be used for simple field mappings in requests
                return TransformationEngine._map_fields_with_jsonpath(
                    litellm_params, transformation_config.mappings or {}
                )
            elif transformation_config.type == "function":
                return TransformationEngine._transform_with_function(
                    litellm_params, transformation_config.module, transformation_config.function
                )
            else:
                raise ValueError(f"Unknown transformation type: {transformation_config.type}")
        except Exception as e:
            verbose_logger.error(f"Request transformation failed: {e}")
            raise

    @staticmethod
    def transform_response(
        provider_response: Dict[str, Any], transformation_config: Any
    ) -> Dict[str, Any]:
        """
        Transform provider response to LiteLLM format.
        
        Args:
            provider_response: Provider's raw response
            transformation_config: TransformationConfig object
        
        Returns:
            LiteLLM-formatted response
        """
        try:
            if transformation_config.type == "jsonpath":
                return TransformationEngine._extract_with_jsonpath(
                    provider_response, transformation_config.mappings or {}
                )
            elif transformation_config.type == "function":
                return TransformationEngine._transform_with_function(
                    provider_response, transformation_config.module, transformation_config.function
                )
            else:
                raise ValueError(
                    f"Unknown transformation type for response: {transformation_config.type}"
                )
        except Exception as e:
            verbose_logger.error(f"Response transformation failed: {e}")
            raise

    @staticmethod
    def _transform_with_jinja(
        data: Dict[str, Any],
        template: Optional[Dict[str, Any]],
        filters: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Transform data using Jinja2 templates.
        
        Args:
            data: Input data (e.g., LiteLLM parameters)
            template: Template structure with Jinja2 expressions
            filters: Custom filter functions
        
        Returns:
            Transformed data
        """
        if not HAS_JINJA2:
            raise ImportError("jinja2 is required for Jinja transformations. Install with: pip install jinja2")
        
        if not template:
            return data
        
        # Convert template dict to JSON string, then render with Jinja2
        template_str = json.dumps(template)
        jinja_template = Jinja2Template(template_str)
        
        # Add custom filters
        if filters:
            for filter_name, filter_mappings in filters.items():
                def create_filter(mappings):
                    def custom_filter(value):
                        return mappings.get(str(value), value)
                    return custom_filter
                jinja_template.globals[filter_name] = create_filter(filter_mappings)
        
        # Render template with data
        rendered = jinja_template.render(**data)
        
        # Parse rendered JSON back to dict
        result = json.loads(rendered)
        
        # Clean up None values if needed
        result = TransformationEngine._remove_none_values(result)
        
        return result

    @staticmethod
    def _extract_with_jsonpath(
        data: Dict[str, Any], mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract fields from data using JSONPath expressions.
        
        Args:
            data: Source data (e.g., provider response)
            mappings: Dict mapping output field names to JSONPath expressions
        
        Returns:
            Extracted data
        """
        if not HAS_JSONPATH:
            raise ImportError("jsonpath-ng is required for JSONPath transformations. Install with: pip install jsonpath-ng")
        
        result = {}
        
        for key, jsonpath_expr in mappings.items():
            # Static values (not JSONPath expressions)
            if not jsonpath_expr.startswith("$"):
                result[key] = jsonpath_expr
                continue
            
            # Parse and execute JSONPath
            parser = jsonpath_parse(jsonpath_expr)
            matches = parser.find(data)
            
            if matches:
                # If multiple matches, return list; otherwise single value
                values = [match.value for match in matches]
                result[key] = values if len(values) > 1 else values[0]
        
        return result

    @staticmethod
    def _map_fields_with_jsonpath(
        data: Dict[str, Any], mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Map fields using simple path-based mappings (for request transformation).
        
        Args:
            data: Input data
            mappings: Dict mapping output field to input field path
        
        Returns:
            Mapped data
        """
        result = {}
        
        for target_key, source_path in mappings.items():
            # Simple dot-notation path traversal
            if source_path.startswith("$."):
                source_path = source_path[2:]  # Remove "$."
            
            keys = source_path.split(".")
            value = data
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break
            
            if value is not None:
                result[target_key] = value
        
        return result

    @staticmethod
    def _transform_with_function(
        data: Dict[str, Any], module_name: Optional[str], function_name: Optional[str]
    ) -> Dict[str, Any]:
        """
        Transform data using a Python function.
        
        Args:
            data: Input data
            module_name: Python module containing the function
            function_name: Function name
        
        Returns:
            Transformed data
        """
        if not module_name or not function_name:
            raise ValueError("module and function are required for function transformations")
        
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)
            return func(data)
        except Exception as e:
            verbose_logger.error(
                f"Failed to execute transformation function {module_name}.{function_name}: {e}"
            )
            raise

    @staticmethod
    def _remove_none_values(data: Any) -> Any:
        """
        Recursively remove None values from nested structures.
        
        Args:
            data: Data structure to clean
        
        Returns:
            Cleaned data structure
        """
        if isinstance(data, dict):
            return {
                k: TransformationEngine._remove_none_values(v)
                for k, v in data.items()
                if v is not None and v != "null" and v != "none"
            }
        elif isinstance(data, list):
            return [TransformationEngine._remove_none_values(item) for item in data]
        else:
            return data
