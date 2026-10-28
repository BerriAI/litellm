"""
Utility module for handling OpenAPI schema generation compatibility with FastAPI 0.120+.

FastAPI 0.120+ has stricter schema generation that fails on certain types like openai.Timeout.
This module provides a compatibility layer to handle these cases gracefully.
"""

import contextvars
import functools
from typing import Final

from litellm._logging import verbose_proxy_logger

_BUILDING_OPENAPI_SCHEMA: Final = contextvars.ContextVar("building_openapi_schema", default=False)


@functools.cache
def _accept_unknown_types_while_building() -> None:
    from pydantic._internal._generate_schema import GenerateSchema
    from pydantic_core import core_schema

    reject_unknown_type: Final = (
        GenerateSchema._unknown_type_schema  # pyright: ignore[reportPrivateUsage]  # Pydantic schema internals
    )

    def unknown_type_schema(self: GenerateSchema, obj: object) -> core_schema.CoreSchema:
        if not _BUILDING_OPENAPI_SCHEMA.get():
            return reject_unknown_type(self, obj)
        if getattr(obj, "__module__", "") == "openai" and "Timeout" in str(obj):
            return core_schema.str_schema()
        return core_schema.any_schema()

    setattr(GenerateSchema, "_unknown_type_schema", unknown_type_schema)


def get_openapi_schema_with_compat(
    get_openapi_func,
    title: str,
    version: str,
    description: str,
    routes: list,
) -> dict[str, object]:
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
    try:
        _accept_unknown_types_while_building()
        building: Final = _BUILDING_OPENAPI_SCHEMA.set(True)
        try:
            openapi_schema: Final = get_openapi_func(
                title=title,
                version=version,
                description=description,
                routes=routes,
            )
        finally:
            _BUILDING_OPENAPI_SCHEMA.reset(building)

        return openapi_schema

    except (ImportError, AttributeError) as e:
        # If patching fails, try normal generation with error handling
        verbose_proxy_logger.debug("Could not patch Pydantic schema generation: %s. Trying normal generation.", e)
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
            error_type_name: Final = type(pydantic_error).__name__
            if error_type_name == "PydanticSchemaGenerationError" or "PydanticSchemaGenerationError" in str(
                type(pydantic_error)
            ):
                # If we still get the error, log it and return minimal schema
                verbose_proxy_logger.warning(
                    "PydanticSchemaGenerationError during schema generation: %s", pydantic_error
                )
                return {
                    "openapi": "3.0.0",
                    "info": {
                        "title": title,
                        "version": version,
                        "description": description or "",
                    },
                    "paths": {},
                    "components": {"schemas": {}},
                }
            else:
                # Re-raise if it's a different error
                raise
