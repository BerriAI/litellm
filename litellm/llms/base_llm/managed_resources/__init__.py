"""
Managed Resources Module

This module provides base classes and utilities for managing resources
(files, vector stores, etc.) with target_model_names support.

The BaseManagedResource class provides common functionality for:
- Storing unified resource IDs with model mappings
- Retrieving resources by unified ID
- Deleting resources across multiple models
- Creating resources for multiple models
- Filtering deployments based on model mappings
"""

from .base_managed_resource import BaseManagedResource
from .utils import (
    decode_unified_id,
    encode_unified_id,
    extract_model_id_from_unified_id,
    extract_provider_resource_id_from_unified_id,
    extract_resource_type_from_unified_id,
    extract_target_model_names_from_unified_id,
    extract_unified_uuid_from_unified_id,
    generate_unified_id_string,
    is_base64_encoded_unified_id,
    parse_unified_id,
)

__all__ = [
    "BaseManagedResource",
    "is_base64_encoded_unified_id",
    "extract_target_model_names_from_unified_id",
    "extract_resource_type_from_unified_id",
    "extract_unified_uuid_from_unified_id",
    "extract_model_id_from_unified_id",
    "extract_provider_resource_id_from_unified_id",
    "generate_unified_id_string",
    "encode_unified_id",
    "decode_unified_id",
    "parse_unified_id",
]
