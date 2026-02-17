import base64
import mimetypes
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Union

from litellm.types.utils import SpecialEnums

if TYPE_CHECKING:
    from fastapi import Request


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


# ============================================================================
#                    MIME TYPE DETECTION AND NORMALIZATION
# ============================================================================


# Gemini-supported image MIME types
GEMINI_SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}

# Gemini-supported video MIME types
GEMINI_SUPPORTED_VIDEO_TYPES = {
    "video/3gpp",
    "video/wmv",
    "video/webm",
    "video/mp4",
    "video/mpg",
    "video/mpegps",
    "video/mpeg",
    "video/quicktime",
    "video/x-flv",
}

# Gemini-supported audio MIME types
GEMINI_SUPPORTED_AUDIO_TYPES = {
    "audio/webm",
    "audio/wav",
    "audio/pcm",
    "audio/opus",
    "audio/mp4",
    "audio/mpga",
    "audio/mpeg",
    "audio/m4a",
    "audio/mp3",
    "audio/flac",
    "audio/aac",
}

# Gemini-supported document MIME types
GEMINI_SUPPORTED_DOCUMENT_TYPES = {
    "text/plain",
    "application/pdf",
}

# Mapping of common file extensions to MIME types
# This extends Python's mimetypes with custom mappings
EXTENSION_TO_MIME_TYPE = {
    ".jpg": "image/jpeg",  # Normalize jpg to jpeg
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
}


def detect_content_type_from_filename(filename: str) -> str:
    """
    Detect content type from filename using extension.
    
    Uses Python's mimetypes module with custom overrides for common cases.
    Normalizes jpg to jpeg for consistency.
    """
    if not filename:
        return "application/octet-stream"
    
    # Try custom mapping first
    filename_lower = filename.lower()
    for ext, mime_type in EXTENSION_TO_MIME_TYPE.items():
        if filename_lower.endswith(ext):
            return mime_type
    
    # Fall back to Python's mimetypes
    mime_type_guess, _ = mimetypes.guess_type(filename)
    if mime_type_guess is not None:
        return mime_type_guess
    
    return "application/octet-stream"


def normalize_mime_type_for_provider(
    mime_type: str, provider: Optional[str] = None
) -> str:
    """
    Normalize MIME type for specific provider requirements.
    
    Currently handles:
    - Gemini: Normalizes image/jpg to image/jpeg
    
    Args:
        mime_type: Original MIME type
        provider: Provider name (e.g., "gemini", "vertex_ai")
        
    Returns:
        str: Normalized MIME type
    """
    normalized = mime_type.lower().strip()
    
    # Gemini/Vertex AI requires image/jpeg, not image/jpg
    if provider and ("gemini" in provider.lower() or "vertex_ai" in provider.lower()):
        if normalized == "image/jpg":
            normalized = "image/jpeg"
    
    # General normalization: always normalize jpg to jpeg
    if normalized == "image/jpg":
        normalized = "image/jpeg"
    
    return normalized


def is_gemini_supported_mime_type(mime_type: str) -> bool:
    """
    Check if a MIME type is supported by Gemini multimodal models.
    
    Supported categories:
    - Images: image/png, image/jpeg, image/webp
    - Video: 3gpp, wmv, webm, mp4, mpg, mpegps, mpeg, quicktime, x-flv
    - Audio: webm, wav, pcm, opus, mp4, mpga, mpeg, m4a, mp3, flac, aac
    - Documents: text/plain, application/pdf
    
    Args:
        mime_type: MIME type to check
        
    Returns:
        bool: True if supported, False otherwise
    """
    normalized = normalize_mime_type_for_provider(mime_type, provider="gemini")
    return normalized in (
        GEMINI_SUPPORTED_IMAGE_TYPES
        | GEMINI_SUPPORTED_VIDEO_TYPES
        | GEMINI_SUPPORTED_AUDIO_TYPES
        | GEMINI_SUPPORTED_DOCUMENT_TYPES
    )


def get_content_type_from_file_object(file_object: Optional[dict]) -> str:
    """
    Determine content type from file object (from database or API response).
    
    Extracts filename from file object and uses detect_content_type_from_filename.
    Falls back to default if file object is invalid or filename not found.
    
    Args:
        file_object: File object dictionary (can be None)
        
    Returns:
        str: MIME type (defaults to "application/octet-stream" if cannot be determined)
    """
    if not file_object:
        return "application/octet-stream"
    
    # Handle JSON string
    if isinstance(file_object, str):
        import json
        try:
            file_object = json.loads(file_object)
        except json.JSONDecodeError:
            return "application/octet-stream"
    
    if not isinstance(file_object, dict):
        return "application/octet-stream"
    
    # Try to get filename
    filename = file_object.get("filename", "")
    if filename:
        return detect_content_type_from_filename(filename)
    
    return "application/octet-stream"


# ============================================================================
#                    REQUEST PARAMETER EXTRACTION
# ============================================================================


@dataclass
class FileCreationParams:
    """
    Structured parameters extracted from file creation requests.
    
    Attributes:
        target_storage: Storage backend name (e.g., "azure_storage", "default")
        target_model_names: List of model names for managed files
        model: Model parameter for multi-account routing
    """
    
    target_storage: str = "default"
    target_model_names: List[str] = field(default_factory=list)
    model: Optional[str] = None
    
    def __post_init__(self):
        """Normalize and validate parameters after initialization."""
        if self.target_model_names is None:
            self.target_model_names = []
        
        # Normalize target_storage
        if not self.target_storage:
            self.target_storage = "default"
        
        # Strip whitespace from model names
        self.target_model_names = [name.strip() for name in self.target_model_names if name.strip()]


async def extract_file_creation_params(
    request: "Request",
    request_body: Optional[dict] = None,
    target_model_names_form: Optional[str] = None,
    target_storage_form: Optional[str] = None,
) -> FileCreationParams:
    """
    Extract file creation parameters from request.
    
    Args:
        request: FastAPI request object
        request_body: Optional pre-parsed request body
        target_model_names_form: target_model_names from form field (comma-separated string)
        target_storage_form: target_storage from form field (defaults to "default")
        
    Returns:
        FileCreationParams: Structured parameters extracted from the request
    """
    from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
    
    if request_body is None:
        request_body = await _read_request_body(request=request) or {}
    
    # Extract target_storage (simplified - just use form parameter)
    target_storage = _extract_target_storage_simple(target_storage_form)
    
    # Extract target_model_names (simplified - just use form parameter)
    target_model_names = _extract_target_model_names_simple(target_model_names_form)
    
    # Extract model parameter
    model = _extract_model_param(request, request_body)
    
    return FileCreationParams(
        target_storage=target_storage,
        target_model_names=target_model_names,
        model=model,
    )


def _extract_target_storage_simple(target_storage_form: Optional[str] = None) -> str:
    """
    Extract target_storage parameter from form field.
    
    Args:
        target_storage_form: target_storage from form field
        
    Returns:
        str: Target storage backend name, or "default"
    """
    if target_storage_form:
        return target_storage_form.strip()
    return "default"


def _extract_target_model_names_simple(target_model_names_form: Optional[str] = None) -> List[str]:
    """
    Extract target_model_names parameter from form field.
    """
    if not target_model_names_form:
        return []
    
    # Parse comma-separated string into list
    if isinstance(target_model_names_form, str):
        return [name.strip() for name in target_model_names_form.split(",") if name.strip()]
    elif isinstance(target_model_names_form, list):
        return [str(name).strip() for name in target_model_names_form if name]
    
    return []


def _extract_model_param(request: "Request", request_body: dict) -> Optional[str]:
    """
    Extract model parameter from request.
    
    Priority:
    1. request_body.model
    2. Query parameter (?model=)
    3. Header (x-litellm-model)
    """
    return (
        request_body.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )


# ============================================================================
#                    BATCH DATABASE OPERATIONS
# ============================================================================


async def resolve_input_file_id_to_unified(response, prisma_client) -> None:
    """
    If the batch response contains a raw provider input_file_id (not already a
    unified ID), look up the corresponding unified file ID from the managed file
    table and replace it in-place.
    """
    if (
        hasattr(response, "input_file_id")
        and response.input_file_id
        and not _is_base64_encoded_unified_file_id(response.input_file_id)
        and prisma_client
    ):
        try:
            managed_file = await prisma_client.db.litellm_managedfiletable.find_first(
                where={"flat_model_file_ids": {"has": response.input_file_id}}
            )
            if managed_file:
                response.input_file_id = managed_file.unified_file_id
        except Exception:
            pass


async def get_batch_from_database(
    batch_id: str,
    unified_batch_id: Union[str, Literal[False]],
    managed_files_obj,
    prisma_client,
    verbose_proxy_logger,
):
    """
    Try to retrieve batch object from ManagedObjectTable for consistent state.
    
    Args:
        batch_id: The batch ID (may be unified/encoded)
        unified_batch_id: Result from _is_base64_encoded_unified_file_id()
        managed_files_obj: The managed_files proxy hook object
        prisma_client: Prisma database client
        verbose_proxy_logger: Logger instance
        
    Returns:
        Tuple of (db_batch_object, response_batch)
        - db_batch_object: Raw database object (or None)
        - response_batch: Parsed LiteLLMBatch object (or None)
    """
    import json
    from litellm.types.utils import LiteLLMBatch
    
    if managed_files_obj is None or not unified_batch_id:
        return None, None
    
    try:
        if not prisma_client:
            return None, None
            
        db_batch_object = await prisma_client.db.litellm_managedobjecttable.find_first(
            where={"unified_object_id": batch_id}
        )
        
        if not db_batch_object or not db_batch_object.file_object:
            return None, None
        
        # Parse the batch object from database
        batch_data = json.loads(db_batch_object.file_object) if isinstance(db_batch_object.file_object, str) else db_batch_object.file_object
        response = LiteLLMBatch(**batch_data)
        response.id = batch_id

        # The stored batch object has the raw provider input_file_id. Resolve to unified ID.
        await resolve_input_file_id_to_unified(response, prisma_client)
        
        verbose_proxy_logger.debug(
            f"Retrieved batch {batch_id} from ManagedObjectTable with status={response.status}"
        )
        
        return db_batch_object, response
        
    except Exception as e:
        verbose_proxy_logger.warning(
            f"Failed to retrieve batch from ManagedObjectTable: {e}, falling back to provider"
        )
        return None, None


async def update_batch_in_database(
    batch_id: str,
    unified_batch_id: Union[str, Literal[False]],
    response,
    managed_files_obj,
    prisma_client,
    verbose_proxy_logger,
    db_batch_object=None,
    operation: str = "update",
):
    """
    Update batch status and object in ManagedObjectTable.
    
    Args:
        batch_id: The batch ID (unified/encoded)
        unified_batch_id: Result from _is_base64_encoded_unified_file_id()
        response: The batch response object with updated state
        managed_files_obj: The managed_files proxy hook object
        prisma_client: Prisma database client
        verbose_proxy_logger: Logger instance
        db_batch_object: Optional existing database object (for comparison)
        operation: Description of operation ("update", "cancel", etc.)
    """
    import litellm.utils
    
    if managed_files_obj is None or not unified_batch_id:
        return
    
    try:
        if not prisma_client:
            return
        
        # Only update if status has changed (when db_batch_object is provided)
        if db_batch_object and response.status == db_batch_object.status:
            return
        
        if db_batch_object:
            verbose_proxy_logger.info(
                f"Updating batch {batch_id} status from {db_batch_object.status} to {response.status}"
            )
        else:
            verbose_proxy_logger.info(
                f"Updating batch {batch_id} status to {response.status} after {operation}"
            )
        
        # Normalize status for database storage
        db_status = response.status if response.status != "completed" else "complete"
        
        await prisma_client.db.litellm_managedobjecttable.update(
            where={"unified_object_id": batch_id},
            data={
                "status": db_status,
                "file_object": response.model_dump_json(),
                "updated_at": litellm.utils.get_utc_datetime(),
            },
        )
    except Exception as e:
        verbose_proxy_logger.error(
            f"Failed to update batch status in ManagedObjectTable: {e}"
        )
