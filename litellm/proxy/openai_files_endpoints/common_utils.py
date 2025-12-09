import base64
import re
from typing import List, Literal, Optional, Union

from litellm.types.utils import SpecialEnums


def _is_base64_encoded_unified_file_id(b64_uid: str) -> Union[str, Literal[False]]:
    # Ensure b64_uid is a string and not a mock object
    if not isinstance(b64_uid, str):
        return False
    # Add padding back if needed
    padded = b64_uid + "=" * (-len(b64_uid) % 4)
    # Decode from base64
    try:
        decoded = base64.urlsafe_b64decode(padded).decode()
        if decoded.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return decoded
        else:
            return False
    except Exception:
        return False


def convert_b64_uid_to_unified_uid(b64_uid: str) -> str:
    is_base64_unified_file_id = _is_base64_encoded_unified_file_id(b64_uid)
    if is_base64_unified_file_id:
        return is_base64_unified_file_id
    else:
        return b64_uid


def get_models_from_unified_file_id(unified_file_id: str) -> List[str]:
    """
    Extract model names from unified file ID.

    Example:
    unified_file_id = "litellm_proxy:application/octet-stream;unified_id,c4843482-b176-4901-8292-7523fd0f2c6e;target_model_names,gpt-4o-mini,gemini-2.0-flash"
    returns: ["gpt-4o-mini", "gemini-2.0-flash"]
    """
    try:
        # Ensure unified_file_id is a string and not a mock object
        if not isinstance(unified_file_id, str):
            return []
        match = re.search(r"target_model_names,([^;]+)", unified_file_id)
        if match:
            # Split on comma and strip whitespace from each model name
            return [model.strip() for model in match.group(1).split(",")]
        return []
    except Exception:
        return []


def get_model_id_from_unified_batch_id(file_id: str) -> Optional[str]:
    """
    Get the model_id from the file_id

    Expected format: litellm_proxy;model_id:{};llm_batch_id:{};llm_output_file_id:{}
    """
    ## use regex to get the model_id from the file_id
    try:
        # Ensure file_id is a string and not a mock object
        if not isinstance(file_id, str):
            return None
        return file_id.split("model_id:")[1].split(";")[0]
    except Exception:
        return None


def get_batch_id_from_unified_batch_id(file_id: str) -> str:
    ## use regex to get the batch_id from the file_id
    # Ensure file_id is a string and not a mock object
    if not isinstance(file_id, str):
        return ""
    if "llm_batch_id" in file_id:
        return file_id.split("llm_batch_id:")[1].split(",")[0]
    else:
        return file_id.split("generic_response_id:")[1].split(",")[0]


def encode_file_id_with_model(file_id: str, model: str) -> str:
    """
    Encode a file/batch ID with model routing information.
    
    Format: <prefix>-<base64(litellm:<original_id>;model,<model_name>)>
    The result preserves the original prefix (file-, batch_, etc.) for OpenAI compliance.
    
    Args:
        file_id: Original file/batch ID from the provider (e.g., "file-abc123", "batch_xyz")
        model: Model name from model_list (e.g., "gpt-4o-litellm")
    
    Returns:
        Encoded ID starting with appropriate prefix and containing routing information
    
    Examples:
        encode_file_id_with_model("file-abc123", "gpt-4o-litellm")
        -> "file-bGl0ZWxsbTpmaWxlLWFiYzEyMzttb2RlbCxncHQtNG8taWZvb2Q"
        
        encode_file_id_with_model("batch_abc123", "gpt-4o-test")
        -> "batch_bGl0ZWxsbTpiYXRjaF9hYmMxMjM7bW9kZWwsZ3B0LTRvLXRlc3Q"
    """
    encoded_str = f"litellm:{file_id};model,{model}"
    encoded_bytes = base64.urlsafe_b64encode(encoded_str.encode())
    encoded_b64 = encoded_bytes.decode().rstrip("=")
    
    # Detect the prefix from the original ID (file-, batch_, etc.)
    # Default to "file-" if no recognizable prefix
    if file_id.startswith("batch_"):
        prefix = "batch_"
    elif file_id.startswith("file-"):
        prefix = "file-"
    else:
        # Default to file- for backward compatibility
        prefix = "file-"
    
    return f"{prefix}{encoded_b64}"


def decode_model_from_file_id(encoded_id: str) -> Optional[str]:
    """
    Extract model name from an encoded file/batch ID.
    Handles IDs that start with "file-" or "batch_" prefix.
    """
    try:
        if not isinstance(encoded_id, str):
            return None
        
        # Remove prefix if present (file-, batch_, etc.)
        if encoded_id.startswith("file-"):
            b64_part = encoded_id[5:]  # Remove "file-"
        elif encoded_id.startswith("batch_"):
            b64_part = encoded_id[6:]  # Remove "batch_"
        else:
            b64_part = encoded_id
        
        padded = b64_part + "=" * (-len(b64_part) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode()   
        if decoded.startswith("litellm:") and ";model," in decoded:
            match = re.search(r";model,([^;]+)", decoded)
            if match:
                return match.group(1).strip()
        
        return None
    except Exception:
        return None


def get_original_file_id(encoded_id: str) -> str:
    """
    Extract the original provider file/batch ID from an encoded ID.
    Handles IDs that start with "file-" or "batch_" prefix.
    """
    try:
        if not isinstance(encoded_id, str):
            return encoded_id
        
        # Remove prefix if present (file-, batch_, etc.)
        if encoded_id.startswith("file-"):
            b64_part = encoded_id[5:]  # Remove "file-"
        elif encoded_id.startswith("batch_"):
            b64_part = encoded_id[6:]  # Remove "batch_"
        else:
            b64_part = encoded_id
        
        padded = b64_part + "=" * (-len(b64_part) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode()
        
        if decoded.startswith("litellm:") and ";model," in decoded:
            match = re.search(r"litellm:([^;]+);model,", decoded)
            if match:
                return match.group(1)
        
        return encoded_id
    except Exception:
        return encoded_id


def is_model_embedded_id(file_id: str) -> bool:
    """
    Check if a file/batch ID has model routing information embedded.
    """
    return decode_model_from_file_id(file_id) is not None


# ============================================================================
#                    MODEL-BASED CREDENTIAL ROUTING HELPERS
# ============================================================================


def extract_model_from_sources(
    file_id: str,
    request,  # FastAPI Request object
    data: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract model information from multiple sources in priority order:
    1. Embedded in file_id (highest priority)
    2. Request headers (x-litellm-model)
    3. Query parameters (?model=)
    4. Request body/data dict
    
    Args:
        file_id: File ID that may contain embedded model info
        request: FastAPI request object
        data: Optional request data dictionary
        
    Returns:
        Tuple of (model_from_id, model_from_param)
        - model_from_id: Model decoded from file ID (if embedded)
        - model_from_param: Model from header/query/body
    """
    if data is None:
        data = {}
    
    # Check if file_id has embedded model info
    model_from_id = decode_model_from_file_id(file_id)
    
    # Check other sources for model parameter
    model_from_param = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    
    return model_from_id, model_from_param


def get_credentials_for_model(
    llm_router,  # Router instance
    model_id: str,
    operation_context: str = "file operation",
):
    """
    Retrieve API credentials for a model from the LLM Router.
    
    Args:
        llm_router: LiteLLM Router instance
        model_id: Model name or deployment ID
        operation_context: Description for error messages (e.g., "file upload", "batch creation")
        
    Returns:
        Dictionary with credentials (api_key, api_base, custom_llm_provider, etc.)
        
    Raises:
        HTTPException: If router not initialized or model not found
    """
    from fastapi import HTTPException
    
    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Router not initialized. Cannot use model-based routing."},
        )
    
    credentials = llm_router.get_deployment_credentials_with_provider(model_id=model_id)
    
    if credentials is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Model '{model_id}' not found in model_list. Please check your config.yaml."
            },
        )
    
    return credentials


def prepare_data_with_credentials(
    data: dict,
    credentials: dict,
    file_id: Optional[str] = None,
) -> None:
    """
    Update data dictionary with model credentials (in-place).
    
    Args:
        data: Data dictionary to update
        credentials: Credentials from router
        file_id: Optional original file_id to set (for decoded file IDs)
    """
    data.update(credentials)
    data.pop("custom_llm_provider", None)
    
    if file_id is not None:
        data["file_id"] = file_id


def handle_model_based_routing(
    file_id: str,
    request,  # FastAPI Request object
    llm_router,  # Router instance
    data: dict,
    check_file_id_encoding: bool = True,
) -> tuple[bool, Optional[str], Optional[str], Optional[dict]]:
    """
    Orchestrate model-based credential routing for file operations.
    
    Args:
        file_id: File ID (may contain embedded model info)
        request: FastAPI request object
        llm_router: LiteLLM Router instance
        data: Request data dictionary
        check_file_id_encoding: Whether to check for embedded model in file_id
        
    Returns:
        Tuple of (should_use_model_routing, model_used, original_file_id, credentials)
        - should_use_model_routing: True if model-based routing should be used
        - model_used: The model name being used
        - original_file_id: Decoded file ID (if it was encoded)
        - credentials: Model credentials dict
        
    Raises:
        HTTPException: If router unavailable or model not found
    """
    model_from_id, model_from_param = extract_model_from_sources(
        file_id=file_id,
        request=request,
        data=data,
    )
    
    # Priority 1: Model embedded in file_id
    if check_file_id_encoding and model_from_id is not None:
        credentials = get_credentials_for_model(
            llm_router=llm_router,
            model_id=model_from_id,
            operation_context=f"file operation (file created with model '{model_from_id}')",
        )
        original_file_id = get_original_file_id(file_id)
        return True, model_from_id, original_file_id, credentials
    
    # Priority 2: Model from header/query/body
    elif model_from_param is not None:
        credentials = get_credentials_for_model(
            llm_router=llm_router,
            model_id=model_from_param,
            operation_context="file operation",
        )
        return True, model_from_param, None, credentials
    
    # No model-based routing needed
    return False, None, None, None
