import base64
import mimetypes
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Optional, Protocol, runtime_checkable

from litellm.repositories.table_repositories import (
    ManagedFileRepository,
    ManagedObjectRepository,
)
from litellm.types.utils import SpecialEnums

if TYPE_CHECKING:
    from fastapi import Request
    from prisma.models import LiteLLM_ManagedObjectTable

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
    from litellm.types.utils import LiteLLMBatch


@runtime_checkable
class ManagedResourceAccessChecker(Protocol):
    async def can_user_call_unified_file_id(
        self,
        unified_file_id: str,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool: ...

    async def can_user_call_unified_object_id(
        self,
        unified_object_id: str,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool: ...


def _is_base64_encoded_unified_file_id(b64_uid: str) -> str | Literal[False]:
    # Ensure b64_uid is a string and not a mock object
    if not isinstance(b64_uid, str):
        return False
    # Add padding back if needed
    padded: Final = b64_uid + "=" * (-len(b64_uid) % 4)
    # Decode from base64
    try:
        decoded: Final = base64.urlsafe_b64decode(padded).decode()
        if decoded.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return decoded
        else:
            return False
    except Exception:
        return False


def convert_b64_uid_to_unified_uid(b64_uid: str) -> str:
    is_base64_unified_file_id: Final = _is_base64_encoded_unified_file_id(b64_uid)
    if is_base64_unified_file_id:
        return is_base64_unified_file_id
    else:
        return b64_uid


def resolve_managed_output_file_model_name(
    unified_input_file_id: str | None, fallback_model_name: str | None
) -> str | None:
    if not unified_input_file_id:
        return fallback_model_name
    target_model_names: Final = get_models_from_unified_file_id(convert_b64_uid_to_unified_uid(unified_input_file_id))
    if target_model_names:
        return ",".join(target_model_names)
    return fallback_model_name


def get_models_from_unified_file_id(unified_file_id: str) -> list[str]:
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
        match: Final = re.search(r"target_model_names,([^;]+)", unified_file_id)
        if match:
            # Split on comma and strip whitespace from each model name
            return [model.strip() for model in match.group(1).split(",")]
        return []
    except Exception:
        return []


def get_model_id_from_unified_batch_id(file_id: str) -> str | None:
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
        batch_id = file_id.split("llm_batch_id:", 1)[1]
    else:
        batch_id = file_id.split("generic_response_id:", 1)[1]
    return re.split(r"[;,]", batch_id, maxsplit=1)[0]


def encode_file_id_with_model(file_id: str, model: str, id_type: Literal["file", "batch"] = "file") -> str:
    """
    Encode a file/batch ID with model routing information.

    Format: <prefix><base64(litellm:<original_id>;model,<model_name>)>
    The result preserves the original prefix (file-, batch_, etc.) for OpenAI compliance.

    Args:
        file_id: Original file/batch ID from the provider (e.g., "file-abc123", "batch_xyz")
        model: Model name from model_list (e.g., "gpt-4o-litellm")
        id_type: Type of ID being encoded. Used to determine the correct prefix when
                 the raw ID lacks a recognizable prefix (e.g., Vertex AI numeric IDs).
                 Defaults to "file" for backward compatibility.

    Returns:
        Encoded ID starting with appropriate prefix and containing routing information

    Examples:
        encode_file_id_with_model("file-abc123", "gpt-4o-litellm")
        -> "file-bGl0ZWxsbTpmaWxlLWFiYzEyMzttb2RlbCxncHQtNG8taWZvb2Q"

        encode_file_id_with_model("batch_abc123", "gpt-4o-test")
        -> "batch_bGl0ZWxsbTpiYXRjaF9hYmMxMjM7bW9kZWwsZ3B0LTRvLXRlc3Q"

        encode_file_id_with_model("3814889423749775360", "gemini-2.5-pro", id_type="batch")
        -> "batch_bGl0ZWxsbTozODE0ODg5NDIzNzQ5Nzc1MzYwO21vZGVsLGdlbWluaS0yLjUtcHJv"
    """
    encoded_str: Final = f"litellm:{file_id};model,{model}"
    encoded_bytes: Final = base64.urlsafe_b64encode(encoded_str.encode())
    encoded_b64: Final = encoded_bytes.decode().rstrip("=")

    # Detect the prefix from the original ID (file-, batch_, etc.)
    # For provider-specific IDs without a recognizable prefix (e.g., Vertex AI
    # numeric batch IDs), fall back to id_type to determine the correct prefix.
    if file_id.startswith("batch_"):
        prefix = "batch_"
    elif file_id.startswith("file-"):
        prefix = "file-"
    else:
        prefix = "batch_" if id_type == "batch" else "file-"

    return f"{prefix}{encoded_b64}"


def encode_batch_response_ids(response, model: str) -> None:
    """Encode all IDs in a batch response with model routing info (in-place)."""
    if not response or not hasattr(response, "id") or not response.id:
        return
    response.id = encode_file_id_with_model(file_id=response.id, model=model, id_type="batch")
    for attr in ("output_file_id", "error_file_id", "input_file_id"):
        if hasattr(response, attr) and getattr(response, attr):
            setattr(
                response,
                attr,
                encode_file_id_with_model(file_id=getattr(response, attr), model=model),
            )


def decode_model_from_file_id(encoded_id: str) -> str | None:
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

        padded: Final = b64_part + "=" * (-len(b64_part) % 4)
        decoded: Final = base64.urlsafe_b64decode(padded).decode()
        if decoded.startswith("litellm:") and ";model," in decoded:
            match: Final = re.search(r";model,([^;]+)", decoded)
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

        padded: Final = b64_part + "=" * (-len(b64_part) % 4)
        decoded: Final = base64.urlsafe_b64decode(padded).decode()

        if decoded.startswith("litellm:") and ";model," in decoded:
            match: Final = re.search(r"litellm:([^;]+);model,", decoded)
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
    data: dict | None = None,
) -> tuple[str | None, str | None]:
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
    model_from_id: Final = decode_model_from_file_id(file_id)

    # Check other sources for model parameter
    model_from_param = data.get("model") or request.query_params.get("model") or request.headers.get("x-litellm-model")

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

    credentials: Final = llm_router.get_deployment_credentials_with_provider(model_id=model_id)

    if credentials is None:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Model '{model_id}' not found in model_list. Please check your config.yaml."},
        )

    return credentials


def get_team_provider_credentials(
    llm_router: Optional["Router"],
    user_api_key_dict: "UserAPIKeyAuth",
    custom_llm_provider: str,
) -> dict | None:
    """
    Resolve upstream credentials for a provider-scoped file operation
    (e.g. GET /v1/files), which doesn't pin a model.

    Priority:
    1. The team's own (BYOK) deployment for this provider — a deployment whose
       ``model_info.team_id`` matches the caller's team. This keeps team-scoped
       listings on the team's own provider account/key instead of a shared
       global one.
    2. Fallback: any deployment the caller is granted access to for this
       provider, expanding wildcard routes and the all-proxy-models sentinel.

    Credential lookup is scoped to both the team's allowlist and the key's own
    model allowlist (``user_api_key_dict.models``), so neither a team nor a
    restricted key within a team can resolve a provider key for a deployment
    it isn't authorized to use. A key restricted to an explicit model list
    only narrows the team scope; sentinel-bearing keys (all-proxy-models /
    all-team-models) defer to the team scope instead of widening past it.
    Returns None when the router is unavailable or no authorized deployment
    matches, so the caller can fall back to default credential resolution.
    """
    if llm_router is None:
        return None

    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.model_checks import get_complete_model_list, get_key_models

    team_id: Final = user_api_key_dict.team_id
    team_models: Final = user_api_key_dict.team_models or []

    proxy_model_list: Final = llm_router.get_model_names(team_id=team_id)
    model_access_groups: Final = llm_router.get_model_access_groups()

    raw_key_models: Final = user_api_key_dict.models or []
    sentinel_values: Final = {
        SpecialModelNames.all_proxy_models.value,
        SpecialModelNames.all_team_models.value,
    }
    key_is_restricted: Final = bool(raw_key_models) and not (set(raw_key_models) & sentinel_values)
    key_model_allowlist: Final = (
        tuple(
            dict.fromkeys(
                get_key_models(
                    user_api_key_dict=user_api_key_dict,
                    proxy_model_list=proxy_model_list,
                    model_access_groups=model_access_groups,
                )
            )
        )
        if key_is_restricted
        else ()
    )
    key_model_allowlist_set: Final = frozenset(key_model_allowlist)

    def _key_may_use(public_model_name: str | None) -> bool:
        if not key_model_allowlist_set:
            return True
        return public_model_name is not None and public_model_name in key_model_allowlist_set

    def _provider_credentials(model_id: str) -> dict | None:
        credentials: Final = llm_router.get_deployment_credentials_with_provider(model_id=model_id, team_id=team_id)
        if credentials is not None and credentials.get("custom_llm_provider") == custom_llm_provider:
            return {key: value for key, value in credentials.items() if key != "model"}
        return None

    # 1. Prefer the team's own BYOK deployment, matched by model_info.team_id.
    if team_id is not None:
        for deployment in llm_router.model_list or []:
            model_info = deployment.get("model_info") or {}
            if model_info.get("team_id") != team_id:
                continue
            deployment_id = model_info.get("id")
            if deployment_id is None:
                continue
            if not _key_may_use(model_info.get("team_public_model_name") or deployment.get("model_name")):
                continue
            credentials = _provider_credentials(deployment_id)
            if credentials is not None:
                return credentials

    # 2. Fall back to deployments the caller is allowed to access. The key's
    #    effective allowlist (sentinels and access groups already expanded by
    #    get_key_models) wins when set; otherwise the team's allowlist applies.
    #    The all-proxy-models sentinel isn't expanded by
    #    get_complete_model_list, so normalize it to an empty allowlist, which
    #    defers to the team-scoped proxy model list. A team or key with a
    #    restricted allowlist (e.g. anthropic only) therefore never resolves
    #    another provider's key.
    grants_all_models: Final = SpecialModelNames.all_proxy_models.value in team_models
    effective_team_models: Final = [] if grants_all_models else team_models

    models_to_try: Final = list(
        dict.fromkeys(
            get_complete_model_list(
                key_models=list(key_model_allowlist),
                team_models=effective_team_models,
                proxy_model_list=proxy_model_list,
                user_model=None,
                infer_model_from_keys=False,
                return_wildcard_routes=True,
                llm_router=llm_router,
                model_access_groups=model_access_groups,
                include_model_access_groups=True,
                team_id=team_id,
            )
        )
    )
    for model_name in models_to_try:
        credentials = _provider_credentials(model_name)
        if credentials is not None:
            return credentials

    return None


def apply_team_provider_credentials(
    data: dict,  # mutable-ok: credentials are merged into the request payload in place, same contract as prepare_data_with_credentials
    llm_router: Optional["Router"],
    user_api_key_dict: "UserAPIKeyAuth",
    custom_llm_provider: str,
) -> None:
    """
    Resolve credentials for a provider-only request (no model pinned) via
    ``get_team_provider_credentials`` and merge them into ``data`` in-place.
    Leaves ``data`` untouched when no authorized deployment matches, so the
    caller falls back to environment-variable credentials exactly as before.
    """
    credentials: Final = get_team_provider_credentials(
        llm_router=llm_router,
        user_api_key_dict=user_api_key_dict,
        custom_llm_provider=custom_llm_provider,
    )
    if credentials is None:
        return
    prepare_data_with_credentials(data=data, credentials=credentials)


def prepare_data_with_credentials(
    data: dict,
    credentials: dict,
    file_id: str | None = None,
    include_internal_credentials: bool = False,
) -> None:
    """
    Update data dictionary with model credentials (in-place).

    Args:
        data: Data dictionary to update
        credentials: Credentials from router
        file_id: Optional original file_id to set (for decoded file IDs)
        include_internal_credentials: Preserve an immutable server-side snapshot
            for code paths that must distinguish proxy config from request params.
    """
    data.update(credentials)
    if include_internal_credentials:
        data["_litellm_internal_model_credentials"] = MappingProxyType(dict(credentials))
    data.pop("custom_llm_provider", None)

    if file_id is not None:
        data["file_id"] = file_id


def handle_model_based_routing(
    file_id: str,
    request,  # FastAPI Request object
    llm_router,  # Router instance
    data: dict,
    check_file_id_encoding: bool = True,
) -> tuple[bool, str | None, str | None, dict | None]:
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
        original_file_id: Final = get_original_file_id(file_id)
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
GEMINI_SUPPORTED_IMAGE_TYPES: Final = {
    "image/png",
    "image/jpeg",
    "image/webp",
}

# Gemini-supported video MIME types
GEMINI_SUPPORTED_VIDEO_TYPES: Final = {
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
GEMINI_SUPPORTED_AUDIO_TYPES: Final = {
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
GEMINI_SUPPORTED_DOCUMENT_TYPES: Final = {
    "text/plain",
    "application/pdf",
}

# Mapping of common file extensions to MIME types
# This extends Python's mimetypes with custom mappings
EXTENSION_TO_MIME_TYPE: Final = {
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
    filename_lower: Final = filename.lower()
    for ext, mime_type in EXTENSION_TO_MIME_TYPE.items():
        if filename_lower.endswith(ext):
            return mime_type

    # Fall back to Python's mimetypes
    mime_type_guess, _ = mimetypes.guess_type(filename)
    if mime_type_guess is not None:
        return mime_type_guess

    return "application/octet-stream"


def normalize_mime_type_for_provider(mime_type: str, provider: str | None = None) -> str:
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
    normalized: Final = normalize_mime_type_for_provider(mime_type, provider="gemini")
    return normalized in (
        GEMINI_SUPPORTED_IMAGE_TYPES
        | GEMINI_SUPPORTED_VIDEO_TYPES
        | GEMINI_SUPPORTED_AUDIO_TYPES
        | GEMINI_SUPPORTED_DOCUMENT_TYPES
    )


def get_content_type_from_file_object(file_object: dict | None) -> str:
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
    filename: Final = file_object.get("filename", "")
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
    target_model_names: list[str] = field(default_factory=list)
    model: str | None = None

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
    request_body: dict | None = None,
    target_model_names_form: str | None = None,
    target_storage_form: str | None = None,
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
    target_storage: Final = _extract_target_storage_simple(target_storage_form)

    # Extract target_model_names from the form field, then fall back to the raw form
    target_model_names = _extract_target_model_names_simple(target_model_names_form)
    if not target_model_names:
        target_model_names = await _extract_target_model_names_from_form(request)

    # Extract model parameter
    model: Final = _extract_model_param(request, request_body)

    return FileCreationParams(
        target_storage=target_storage,
        target_model_names=target_model_names,
        model=model,
    )


def _extract_target_storage_simple(target_storage_form: str | None = None) -> str:
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


def _extract_target_model_names_simple(
    target_model_names_form: str | None = None,
) -> list[str]:
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


def _is_target_model_names_key(key: str) -> bool:
    return key == "target_model_names" or (key.startswith("target_model_names[") and key.endswith("]"))


async def _extract_target_model_names_from_form(request: "Request") -> list[str]:
    """
    Collect target_model_names from the raw multipart form.

    Reads ``request.form()`` directly instead of the parsed request body, which is
    built via ``dict(form_data)`` and keeps only the last value for repeated keys.
    The OpenAI SDK sends a list ``extra_body`` as repeated ``target_model_names[]``
    fields, so reading the form preserves every value instead of truncating to one.
    Indexed keys like ``target_model_names[0]`` are handled the same way.
    """
    form_data: Final = await request.form()

    names: Final[list[str]] = []
    for key, value in form_data.multi_items():
        if _is_target_model_names_key(key) and isinstance(value, str):
            names.extend(_extract_target_model_names_simple(value))

    seen: Final = set()
    result: Final[list[str]] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def validate_managed_files_requirement(
    target_model_names: list[str],
    model: str | None = None,
) -> None:
    """
    Enforce proxy-level managed files when litellm.require_managed_files is enabled.

    Raises:
        HTTPException: 400 if the upload would bypass the managed-files flow, i.e.
            target_model_names is missing or a model parameter routes the request
            through the direct provider path instead of the managed-files hook.
    """
    from fastapi import HTTPException

    import litellm

    if litellm.require_managed_files is not True:
        return

    if not target_model_names:
        raise HTTPException(
            status_code=400,
            detail=(
                "target_model_names is required when require_managed_files is enabled "
                "in litellm_settings. Provide one or more model aliases via the "
                "target_model_names form field (e.g. target_model_names=my-model-alias)."
            ),
        )

    if model:
        raise HTTPException(
            status_code=400,
            detail=(
                "model is not allowed when require_managed_files is enabled in "
                "litellm_settings. Uploads must go through managed files using "
                "target_model_names instead of the model parameter."
            ),
        )


async def validate_managed_id_requirement(
    resource_id: str | None,
    resource_kind: Literal["file", "batch", "fine-tuning job"],
    user_api_key_dict: "UserAPIKeyAuth",
    managed_files_obj: object | None,
) -> None:
    """
    Enforce proxy-level managed resources on every route that accepts a provider-issued id
    when ``litellm.require_managed_files`` is enabled, and authenticate managed ids against
    the caller's stored ownership record.

    Ownership is only recorded for LiteLLM managed ids, so a raw provider id is forwarded to the
    provider under shared credentials without any tenant check; knowing another tenant's provider
    id would be enough to read, reuse, or destroy the object behind it.

    Raises:
        HTTPException: 400 for a raw id, 403 for an inaccessible managed id, or 500 when
            ownership validation is unavailable.
    """
    from fastapi import HTTPException

    import litellm

    if litellm.require_managed_files is not True:
        return

    if not resource_id:
        return

    if not _is_base64_encoded_unified_file_id(resource_id):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Raw provider {resource_kind} ids cannot be used when require_managed_files is enabled in "
                f"litellm_settings. Use the LiteLLM managed {resource_kind} id returned when the "
                f"{resource_kind} was created."
            ),
        )

    if not isinstance(managed_files_obj, ManagedResourceAccessChecker):
        raise HTTPException(
            status_code=500,
            detail="Managed resource ownership validation is unavailable.",
        )

    can_access: Final = (
        await managed_files_obj.can_user_call_unified_file_id(resource_id, user_api_key_dict)
        if resource_kind == "file"
        else await managed_files_obj.can_user_call_unified_object_id(resource_id, user_api_key_dict)
    )
    if can_access:
        return

    raise HTTPException(
        status_code=403,
        detail=f"The caller does not have access to this managed {resource_kind} id.",
    )


def _extract_model_param(request: "Request", request_body: dict) -> str | None:
    """
    Extract model parameter from request.

    Priority:
    1. request_body.model
    2. Query parameter (?model=)
    3. Header (x-litellm-model)
    """
    return request_body.get("model") or request.query_params.get("model") or request.headers.get("x-litellm-model")


# ============================================================================
#                    BATCH DATABASE OPERATIONS
# ============================================================================


def _batch_response_model_id_candidates(
    response,
    unified_batch_id: str | Literal[False] | None,
) -> tuple[str, ...]:
    response_id: Final = getattr(response, "id", None)
    decoded_response_id: Final = (
        _is_base64_encoded_unified_file_id(response_id) if isinstance(response_id, str) else False
    )
    return tuple(
        candidate
        for candidate in (
            unified_batch_id if isinstance(unified_batch_id, str) else None,
            decoded_response_id or None,
            response_id
            if isinstance(response_id, str)
            and not decoded_response_id
            and response_id.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value)
            else None,
        )
        if candidate
    )


def _model_id_for_batch_response(
    response: "LiteLLMBatch",
    unified_batch_id: str | Literal[False] | None,
) -> str | None:
    hidden_params: Final = getattr(response, "_hidden_params", None) or {}
    model_id: Final = hidden_params.get("model_id")
    if model_id:
        return model_id
    return next(
        (
            candidate_model_id
            for candidate in _batch_response_model_id_candidates(response, unified_batch_id)
            if (candidate_model_id := get_model_id_from_unified_batch_id(candidate))
        ),
        None,
    )


def _model_name_for_batch_response(response: "LiteLLMBatch") -> str | None:
    hidden_params: Final = getattr(response, "_hidden_params", None) or {}
    unified_file_id: Final = hidden_params.get("unified_file_id")
    return resolve_managed_output_file_model_name(
        unified_input_file_id=unified_file_id
        if isinstance(unified_file_id, str)
        else getattr(response, "input_file_id", None),
        fallback_model_name=hidden_params.get("model_name"),
    )


def _batch_owner_auth_from_db_object(db_batch_object: "LiteLLM_ManagedObjectTable") -> "UserAPIKeyAuth | None":
    from litellm.proxy._types import UserAPIKeyAuth

    created_by: Final = getattr(db_batch_object, "created_by", None)
    if not isinstance(created_by, str) or not created_by:
        return None
    raw_team_id: Final = getattr(db_batch_object, "team_id", None)
    return UserAPIKeyAuth(user_id=created_by, team_id=raw_team_id if isinstance(raw_team_id, str) else None)


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
            managed_file: Final = await ManagedFileRepository(prisma_client).table.find_first(
                where={"flat_model_file_ids": {"has": response.input_file_id}}
            )
            if managed_file:
                response.input_file_id = managed_file.unified_file_id
        except Exception:
            pass


async def resolve_output_file_ids_to_unified(response, prisma_client) -> None:
    """
    If the batch response contains raw provider output_file_id or error_file_id
    (not already unified IDs), look up the corresponding unified file IDs from
    the managed file table and replace them in-place.
    """
    if not prisma_client:
        return
    for attr in ("output_file_id", "error_file_id"):
        raw_id = getattr(response, attr, None)
        if not raw_id or _is_base64_encoded_unified_file_id(raw_id):
            continue
        try:
            managed_file = await ManagedFileRepository(prisma_client).table.find_first(
                where={"flat_model_file_ids": {"has": raw_id}}
            )
            if managed_file:
                setattr(response, attr, managed_file.unified_file_id)
        except Exception:
            pass


async def map_raw_file_ids_to_unified(
    raw_file_ids: frozenset[str], prisma_client: "PrismaClient | None"
) -> Mapping[str, str]:
    if not raw_file_ids or not prisma_client:
        return MappingProxyType({})
    managed_files: Final = await ManagedFileRepository(prisma_client).table.find_many(
        where={"flat_model_file_ids": {"hasSome": sorted(raw_file_ids)}}  # mutable-ok: prisma where is a plain dict
    )
    return MappingProxyType(
        {
            raw_id: managed_file.unified_file_id
            for managed_file in managed_files
            for raw_id in managed_file.flat_model_file_ids
            if raw_id in raw_file_ids
        }
    )


def apply_unified_file_ids(response: "LiteLLMBatch", unified_id_by_raw_id: Mapping[str, str]) -> None:
    for file_attr, raw_id in (
        ("input_file_id", getattr(response, "input_file_id", None)),
        ("output_file_id", getattr(response, "output_file_id", None)),
        ("error_file_id", getattr(response, "error_file_id", None)),
    ):
        if isinstance(raw_id, str) and raw_id in unified_id_by_raw_id:
            setattr(response, file_attr, unified_id_by_raw_id[raw_id])


async def ensure_batch_response_managed_file_ids(
    response,
    managed_files_obj,
    prisma_client,
    verbose_proxy_logger,
    user_api_key_dict=None,
    db_batch_object=None,
    unified_batch_id: str | Literal[False] | None = None,
) -> None:
    """Normalize batch file IDs to managed unified IDs before DB persistence."""
    await resolve_input_file_id_to_unified(response, prisma_client)
    await resolve_output_file_ids_to_unified(response, prisma_client)

    if managed_files_obj is None:
        return

    model_id: Final = _model_id_for_batch_response(response, unified_batch_id)
    if not model_id:
        return

    model_name: Final = _model_name_for_batch_response(response)

    owner_auth: Final = _batch_owner_auth_from_db_object(db_batch_object) if db_batch_object is not None else None
    effective_auth: Final = owner_auth if owner_auth is not None else user_api_key_dict
    if effective_auth is None:
        return

    for file_attr in ("output_file_id", "error_file_id"):
        raw_file_id = getattr(response, file_attr, None)
        if not raw_file_id or _is_base64_encoded_unified_file_id(raw_file_id):
            continue
        try:
            new_unified_file_id = managed_files_obj.get_unified_output_file_id(
                output_file_id=raw_file_id,
                model_id=model_id,
                model_name=model_name,
            )
            await managed_files_obj.store_unified_file_id(
                file_id=new_unified_file_id,
                file_object=None,
                litellm_parent_otel_span=getattr(effective_auth, "parent_otel_span", None),
                model_mappings={model_id: raw_file_id},
                user_api_key_dict=effective_auth,
            )
            setattr(response, file_attr, new_unified_file_id)
            verbose_proxy_logger.debug("Converted batch %s %r to managed ID before DB write", file_attr, raw_file_id)
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to convert batch %s=%r to managed ID before DB write: %s", file_attr, raw_file_id, e
            )


async def get_batch_from_database(
    batch_id: str,
    unified_batch_id: str | Literal[False],
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

        db_batch_object: Final = await ManagedObjectRepository(prisma_client).table.find_first(
            where={"unified_object_id": batch_id}
        )

        if not db_batch_object or not db_batch_object.file_object:
            return None, None

        # Parse the batch object from database
        batch_data: Final = (
            json.loads(db_batch_object.file_object)
            if isinstance(db_batch_object.file_object, str)
            else db_batch_object.file_object
        )
        response: Final = LiteLLMBatch.model_validate(batch_data)
        response.id = batch_id

        # The stored batch object may have raw provider file IDs. Register any missing
        # managed-file rows and normalize output/error IDs before returning.
        await ensure_batch_response_managed_file_ids(
            response=response,
            managed_files_obj=managed_files_obj,
            prisma_client=prisma_client,
            verbose_proxy_logger=verbose_proxy_logger,
            db_batch_object=db_batch_object,
            unified_batch_id=unified_batch_id,
        )

        verbose_proxy_logger.debug(
            "Retrieved batch %s from ManagedObjectTable with status=%s", batch_id, response.status
        )

        return db_batch_object, response

    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to retrieve batch from ManagedObjectTable: %s, falling back to provider", e
        )
        return None, None


async def update_batch_in_database(
    batch_id: str,
    unified_batch_id: str | Literal[False],
    response,
    managed_files_obj,
    prisma_client,
    verbose_proxy_logger,
    db_batch_object=None,
    operation: str = "update",
    user_api_key_dict=None,
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
        db_batch_object: Optional existing database object; fetched by unified_object_id when omitted
        operation: Description of operation ("update", "cancel", etc.)
        user_api_key_dict: Optional auth context for creating managed file IDs
    """
    import litellm.utils

    if managed_files_obj is None or not unified_batch_id:
        return

    try:
        if not prisma_client:
            return

        effective_db_batch_object: Final = (
            db_batch_object
            if db_batch_object is not None
            else await ManagedObjectRepository(prisma_client).table.find_first(where={"unified_object_id": batch_id})
        )

        # Always normalize the response's file IDs to unified managed IDs
        # (mutates in place) so the caller returns unified IDs to the user
        # even when we skip the DB update below for an unchanged status.
        await ensure_batch_response_managed_file_ids(
            response=response,
            managed_files_obj=managed_files_obj,
            prisma_client=prisma_client,
            verbose_proxy_logger=verbose_proxy_logger,
            user_api_key_dict=user_api_key_dict,
            db_batch_object=effective_db_batch_object,
            unified_batch_id=unified_batch_id,
        )

        # Only update if status has changed (when db_batch_object is provided)
        if effective_db_batch_object and response.status == effective_db_batch_object.status:
            return

        if effective_db_batch_object:
            verbose_proxy_logger.info(
                "Updating batch %s status from %s to %s", batch_id, effective_db_batch_object.status, response.status
            )
        else:
            verbose_proxy_logger.info("Updating batch %s status to %s after %s", batch_id, response.status, operation)

        # Normalize status for database storage
        db_status: Final = response.status if response.status != "completed" else "complete"

        update_data: Final[dict] = {
            "status": db_status,
            "file_object": response.model_dump_json(),
            "updated_at": litellm.utils.get_utc_datetime(),
        }

        # When a batch reaches completion, also mark batch_processed=True.
        # The cost callback is enqueued asynchronously during the
        # aretrieve_batch call that detected completion (via the @client
        # decorator).  It is not awaited, so there is a theoretical window
        # where the callback hasn't executed yet.  In practice the callback
        # completes reliably.  Setting the flag here unblocks file deletion
        # which queries batch_processed=False.  CheckBatchCost acts as a
        # safety net for the rare case where the callback fails.
        if db_status == "complete":
            update_data["batch_processed"] = True

        try:
            await ManagedObjectRepository(prisma_client).table.update(
                where={"unified_object_id": batch_id},
                data=update_data,
            )
        except Exception as col_err:
            # If the batch_processed column doesn't exist (old schema),
            # retry without it so the status update still succeeds.
            err_str: Final = str(col_err).lower()
            if "batch_processed" in err_str and update_data.get("batch_processed") is not None:
                verbose_proxy_logger.warning(
                    "batch_processed column not found, retrying update without it: %s", col_err
                )
                update_data.pop("batch_processed", None)
                await ManagedObjectRepository(prisma_client).table.update(
                    where={"unified_object_id": batch_id},
                    data=update_data,
                )
            else:
                raise
    except Exception as e:
        verbose_proxy_logger.error("Failed to update batch status in ManagedObjectTable: %s", e)
