"""Typed, provenance-aware resolution of proxy settings from DB then env."""

from litellm.proxy.config_resolvers._descriptors import (
    FieldDescriptor,
    FieldSource,
    resolve_fields,
)

__all__ = ["FieldDescriptor", "FieldSource", "resolve_fields"]
