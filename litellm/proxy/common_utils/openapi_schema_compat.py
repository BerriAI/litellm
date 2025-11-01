"""
Utility module for handling OpenAPI schema generation compatibility with FastAPI 0.120+.

FastAPI 0.120+ has stricter schema generation that fails on certain types like openai.Timeout.
This module provides a compatibility layer to handle these cases gracefully.
"""

from typing import Any, Dict

from litellm._logging import verbose_proxy_logger


def get_openapi_schema_with_compat(
    get_openapi_func,
    title: str,
    version: str,
    description: str,
    routes: list,
) -> Dict[str, Any]:
    """
    Generate OpenAPI schema with compatibility handling for FastAPI 0.120+.
    
    This function patches Pydantic's schema generation to handle non-serializable types
    like openai.Timeout that cause PydanticSchemaGenerationError in FastAPI 0.120+.
    
    Args:
        get_openapi_func: The FastAPI get_openapi function
        title: API title
        version: API version
        description: API description
        routes: List of routes
        
    Returns:
        OpenAPI schema dictionary
    """
    # FastAPI 0.120+ may fail schema generation for certain types (e.g., openai.Timeout)
    # Patch Pydantic's schema generation to handle unknown types gracefully
    try:
        from pydantic._internal._generate_schema import GenerateSchema
        from pydantic_core import core_schema

        # Store original method
        original_unknown_type_schema = GenerateSchema._unknown_type_schema
        
        def patched_unknown_type_schema(self, obj):
            """Patch to handle openai.Timeout and other non-serializable types"""
            # Check if it's openai.Timeout or similar types
            obj_str = str(obj)
            obj_module = getattr(obj, '__module__', '')
            
            if (obj_module == 'openai' and 'Timeout' in obj_str) or \
               (hasattr(obj, '__name__') and obj.__name__ == 'Timeout' and obj_module == 'openai'):
                # Return a simple string schema for Timeout types
                return core_schema.str_schema()
            
            # For other unknown types, try to return a default schema
            # This prevents the error from propagating
            try:
                return core_schema.any_schema()
            except Exception:
                # Last resort: return string schema
                return core_schema.str_schema()
        
        # Apply patch
        setattr(GenerateSchema, '_unknown_type_schema', patched_unknown_type_schema)
        
        try:
            openapi_schema = get_openapi_func(
                title=title,
                version=version,
                description=description,
                routes=routes,
            )
        finally:
            # Restore original method
            setattr(GenerateSchema, '_unknown_type_schema', original_unknown_type_schema)
            
        return openapi_schema
            
    except (ImportError, AttributeError) as e:
        # If patching fails, try normal generation with error handling
        verbose_proxy_logger.debug(f"Could not patch Pydantic schema generation: {e}. Trying normal generation.")
        try:
            return get_openapi_func(
                title=title,
                version=version,
                description=description,
                routes=routes,
            )
        except Exception as pydantic_error:
            # Check if it's a PydanticSchemaGenerationError by checking the error type name
            # This avoids import issues if PydanticSchemaGenerationError is not available
            error_type_name = type(pydantic_error).__name__
            if error_type_name == "PydanticSchemaGenerationError" or "PydanticSchemaGenerationError" in str(type(pydantic_error)):
                # If we still get the error, log it and return minimal schema
                verbose_proxy_logger.warning(f"PydanticSchemaGenerationError during schema generation: {pydantic_error}")
                return {
                    "openapi": "3.0.0",
                    "info": {"title": title, "version": version, "description": description or ""},
                    "paths": {},
                    "components": {"schemas": {}},
                }
            else:
                # Re-raise if it's a different error
                raise

