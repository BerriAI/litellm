"""
Utility functions for managed resources.

This module provides common utility functions that can be used across
different managed resource types (files, vector stores, etc.).
"""

import base64
import re
from typing import List, Optional, Union, Literal


def is_base64_encoded_unified_id(
    resource_id: str,
    prefix: str = "litellm_proxy:",
) -> Union[str, Literal[False]]:
    """
    Check if a resource ID is a base64 encoded unified ID.
    
    Args:
        resource_id: The resource ID to check
        prefix: The expected prefix for unified IDs
        
    Returns:
        Decoded string if valid unified ID, False otherwise
    """
    # Ensure resource_id is a string
    if not isinstance(resource_id, str):
        return False
    
    # Add padding back if needed
    padded = resource_id + "=" * (-len(resource_id) % 4)
    
    # Decode from base64
    try:
        decoded = base64.urlsafe_b64decode(padded).decode()
        if decoded.startswith(prefix):
            return decoded
        else:
            return False
    except Exception:
        return False


def extract_target_model_names_from_unified_id(
    unified_id: str,
) -> List[str]:
    """
    Extract target model names from a unified resource ID.
    
    Args:
        unified_id: The unified resource ID (decoded or encoded)
        
    Returns:
        List of target model names
        
    Example:
        unified_id = "litellm_proxy:vector_store;unified_id,uuid;target_model_names,gpt-4,gemini-2.0"
        returns: ["gpt-4", "gemini-2.0"]
    """
    try:
        # Ensure unified_id is a string
        if not isinstance(unified_id, str):
            return []
        
        # Decode if it's base64 encoded
        decoded_id = is_base64_encoded_unified_id(unified_id)
        if decoded_id:
            unified_id = decoded_id
        
        # Extract model names using regex
        match = re.search(r"target_model_names,([^;]+)", unified_id)
        if match:
            # Split on comma and strip whitespace from each model name
            return [model.strip() for model in match.group(1).split(",")]
        
        return []
    except Exception:
        return []


def extract_resource_type_from_unified_id(
    unified_id: str,
) -> Optional[str]:
    """
    Extract resource type from a unified resource ID.
    
    Args:
        unified_id: The unified resource ID (decoded or encoded)
        
    Returns:
        Resource type string or None
        
    Example:
        unified_id = "litellm_proxy:vector_store;unified_id,uuid;..."
        returns: "vector_store"
    """
    try:
        # Ensure unified_id is a string
        if not isinstance(unified_id, str):
            return None
        
        # Decode if it's base64 encoded
        decoded_id = is_base64_encoded_unified_id(unified_id)
        if decoded_id:
            unified_id = decoded_id
        
        # Extract resource type (comes after prefix and before first semicolon)
        match = re.search(r"litellm_proxy:([^;]+)", unified_id)
        if match:
            return match.group(1).strip()
        
        return None
    except Exception:
        return None


def extract_unified_uuid_from_unified_id(
    unified_id: str,
) -> Optional[str]:
    """
    Extract the UUID from a unified resource ID.
    
    Args:
        unified_id: The unified resource ID (decoded or encoded)
        
    Returns:
        UUID string or None
        
    Example:
        unified_id = "litellm_proxy:vector_store;unified_id,abc-123;..."
        returns: "abc-123"
    """
    try:
        # Ensure unified_id is a string
        if not isinstance(unified_id, str):
            return None
        
        # Decode if it's base64 encoded
        decoded_id = is_base64_encoded_unified_id(unified_id)
        if decoded_id:
            unified_id = decoded_id
        
        # Extract UUID
        match = re.search(r"unified_id,([^;]+)", unified_id)
        if match:
            return match.group(1).strip()
        
        return None
    except Exception:
        return None


def extract_model_id_from_unified_id(
    unified_id: str,
) -> Optional[str]:
    """
    Extract model ID from a unified resource ID.
    
    Args:
        unified_id: The unified resource ID (decoded or encoded)
        
    Returns:
        Model ID string or None
        
    Example:
        unified_id = "litellm_proxy:vector_store;...;model_id,gpt-4-model-id;..."
        returns: "gpt-4-model-id"
    """
    try:
        # Ensure unified_id is a string
        if not isinstance(unified_id, str):
            return None
        
        # Decode if it's base64 encoded
        decoded_id = is_base64_encoded_unified_id(unified_id)
        if decoded_id:
            unified_id = decoded_id
        
        # Extract model ID
        match = re.search(r"model_id,([^;]+)", unified_id)
        if match:
            return match.group(1).strip()
        
        return None
    except Exception:
        return None


def extract_provider_resource_id_from_unified_id(
    unified_id: str,
) -> Optional[str]:
    """
    Extract provider resource ID from a unified resource ID.
    
    Args:
        unified_id: The unified resource ID (decoded or encoded)
        
    Returns:
        Provider resource ID string or None
        
    Example:
        unified_id = "litellm_proxy:vector_store;...;resource_id,vs_abc123;..."
        returns: "vs_abc123"
    """
    try:
        # Ensure unified_id is a string
        if not isinstance(unified_id, str):
            return None
        
        # Decode if it's base64 encoded
        decoded_id = is_base64_encoded_unified_id(unified_id)
        if decoded_id:
            unified_id = decoded_id
        
        # Extract resource ID (try multiple patterns for different resource types)
        patterns = [
            r"resource_id,([^;]+)",
            r"vector_store_id,([^;]+)",
            r"file_id,([^;]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, unified_id)
            if match:
                return match.group(1).strip()
        
        return None
    except Exception:
        return None


def generate_unified_id_string(
    resource_type: str,
    unified_uuid: str,
    target_model_names: List[str],
    provider_resource_id: str,
    model_id: str,
    additional_fields: Optional[dict] = None,
) -> str:
    """
    Generate a unified ID string (before base64 encoding).
    
    Args:
        resource_type: Type of resource (e.g., "vector_store", "file")
        unified_uuid: UUID for this unified resource
        target_model_names: List of target model names
        provider_resource_id: Resource ID from the provider
        model_id: Model ID from the router
        additional_fields: Additional fields to include in the ID
        
    Returns:
        Unified ID string (not yet base64 encoded)
        
    Example:
        generate_unified_id_string(
            resource_type="vector_store",
            unified_uuid="abc-123",
            target_model_names=["gpt-4", "gemini"],
            provider_resource_id="vs_xyz",
            model_id="model-id-123",
        )
        returns: "litellm_proxy:vector_store;unified_id,abc-123;target_model_names,gpt-4,gemini;resource_id,vs_xyz;model_id,model-id-123"
    """
    # Build the unified ID string
    parts = [
        f"litellm_proxy:{resource_type}",
        f"unified_id,{unified_uuid}",
        f"target_model_names,{','.join(target_model_names)}",
        f"resource_id,{provider_resource_id}",
        f"model_id,{model_id}",
    ]
    
    # Add additional fields if provided
    if additional_fields:
        for key, value in additional_fields.items():
            parts.append(f"{key},{value}")
    
    return ";".join(parts)


def encode_unified_id(unified_id_string: str) -> str:
    """
    Encode a unified ID string to base64.
    
    Args:
        unified_id_string: The unified ID string to encode
        
    Returns:
        Base64 encoded unified ID (URL-safe, padding stripped)
    """
    return (
        base64.urlsafe_b64encode(unified_id_string.encode())
        .decode()
        .rstrip("=")
    )


def decode_unified_id(encoded_unified_id: str) -> Optional[str]:
    """
    Decode a base64 encoded unified ID.
    
    Args:
        encoded_unified_id: The base64 encoded unified ID
        
    Returns:
        Decoded unified ID string or None if invalid
    """
    try:
        # Add padding back if needed
        padded = encoded_unified_id + "=" * (-len(encoded_unified_id) % 4)
        
        # Decode from base64
        decoded = base64.urlsafe_b64decode(padded).decode()
        
        # Verify it starts with the expected prefix
        if decoded.startswith("litellm_proxy:"):
            return decoded
        
        return None
    except Exception:
        return None


def parse_unified_id(
    unified_id: str,
) -> Optional[dict]:
    """
    Parse a unified ID into its components.
    
    Args:
        unified_id: The unified ID (encoded or decoded)
        
    Returns:
        Dictionary with parsed components or None if invalid
        
    Example:
        {
            "resource_type": "vector_store",
            "unified_uuid": "abc-123",
            "target_model_names": ["gpt-4", "gemini"],
            "provider_resource_id": "vs_xyz",
            "model_id": "model-id-123"
        }
    """
    try:
        # Decode if needed
        decoded_id = decode_unified_id(unified_id)
        if not decoded_id:
            # Maybe it's already decoded
            if unified_id.startswith("litellm_proxy:"):
                decoded_id = unified_id
            else:
                return None
        
        return {
            "resource_type": extract_resource_type_from_unified_id(decoded_id),
            "unified_uuid": extract_unified_uuid_from_unified_id(decoded_id),
            "target_model_names": extract_target_model_names_from_unified_id(decoded_id),
            "provider_resource_id": extract_provider_resource_id_from_unified_id(decoded_id),
            "model_id": extract_model_id_from_unified_id(decoded_id),
        }
    except Exception:
        return None
