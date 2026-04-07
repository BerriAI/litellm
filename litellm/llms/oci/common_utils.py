import base64
import datetime
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple
from urllib.parse import urlparse

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    _CRYPTOGRAPHY_AVAILABLE = False

try:
    from litellm._version import version as _litellm_version
except ImportError:
    _litellm_version = "0.0.0"


# OCI GenAI REST API version — stable since service launch, unlikely to change
OCI_API_VERSION = "20231130"


def _require_cryptography() -> None:
    if not _CRYPTOGRAPHY_AVAILABLE:
        raise ImportError(
            "cryptography package is required for OCI authentication. "
            "Please install it with: pip install cryptography"
        )


class OCIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[httpx.Headers] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


# ---------------------------------------------------------------------------
# OCI signing protocol and helpers
# ---------------------------------------------------------------------------


class OCISignerProtocol(Protocol):
    """
    Protocol for OCI request signers (e.g., oci.signer.Signer).

    Compatible with the OCI Python SDK's Signer class.
    See: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html
    """

    def do_request_sign(
        self, request: Any, *, enforce_content_headers: bool = False
    ) -> None:
        pass


@dataclass
class OCIRequestWrapper:
    """
    Wrapper for HTTP requests compatible with OCI signer interface.

    Wraps request data in the format expected by OCI SDK signers, which require
    objects with method, url, headers, body, and path_url attributes.
    """

    method: str
    url: str
    headers: dict
    body: bytes

    @property
    def path_url(self) -> str:
        """Returns the path + query string for OCI signing."""
        parsed = urlparse(self.url)
        return parsed.path + ("?" + parsed.query if parsed.query else "")


def sha256_base64(data: bytes) -> str:
    # SHA-256 is used here to compute the x-content-sha256 header required by the
    # OCI HTTP signing specification (RSA-SHA256 request signing), not for password
    # or secret hashing.  This is the correct and mandated algorithm for this purpose.
    # See: https://docs.oracle.com/en-us/iaas/Content/API/Concepts/signingrequests.htm
    digest = hashlib.sha256(data).digest()  # lgtm[py/weak-sensitive-data-hashing] # noqa: S324
    return base64.b64encode(digest).decode()


def build_signature_string(
    method: str, path: str, headers: dict, signed_headers: list
) -> str:
    lines = []
    for header in signed_headers:
        if header == "(request-target)":
            value = f"{method.lower()} {path}"
        else:
            value = headers[header]
        lines.append(f"{header}: {value}")
    return "\n".join(lines)


def load_private_key_from_str(key_str: str) -> Any:
    _require_cryptography()
    key = serialization.load_pem_private_key(  # type: ignore[union-attr]
        key_str.encode("utf-8"),
        password=None,
    )
    if not isinstance(key, rsa.RSAPrivateKey):  # type: ignore[union-attr]
        raise TypeError(
            "The provided private key is not an RSA key, which is required for OCI signing."
        )
    return key


def load_private_key_from_file(file_path: str) -> Any:
    """Loads a private key from a file path."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            key_str = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Private key file not found: {file_path}")
    except OSError as e:
        raise OSError(f"Failed to read private key file '{file_path}': {e}") from e

    if not key_str:
        raise ValueError(f"Private key file is empty: {file_path}")

    return load_private_key_from_str(key_str)


# ---------------------------------------------------------------------------
# Env-var credential resolution
# ---------------------------------------------------------------------------

_OCI_REGION_ENV = "OCI_REGION"
_OCI_USER_ENV = "OCI_USER"
_OCI_FINGERPRINT_ENV = "OCI_FINGERPRINT"
_OCI_TENANCY_ENV = "OCI_TENANCY"
_OCI_KEY_FILE_ENV = "OCI_KEY_FILE"
_OCI_KEY_ENV = "OCI_KEY"
_OCI_COMPARTMENT_ID_ENV = "OCI_COMPARTMENT_ID"


def resolve_oci_credentials(optional_params: dict) -> dict:
    """
    Merge OCI credentials from optional_params (explicit, always wins) and
    environment variables (fallback).

    Returns a dict with resolved values for:
        oci_region, oci_user, oci_fingerprint, oci_tenancy,
        oci_key, oci_key_file, oci_compartment_id
    """
    return {
        "oci_region": optional_params.get("oci_region")
        or os.environ.get(_OCI_REGION_ENV)
        or "us-ashburn-1",
        "oci_user": optional_params.get("oci_user") or os.environ.get(_OCI_USER_ENV),
        "oci_fingerprint": optional_params.get("oci_fingerprint")
        or os.environ.get(_OCI_FINGERPRINT_ENV),
        "oci_tenancy": optional_params.get("oci_tenancy")
        or os.environ.get(_OCI_TENANCY_ENV),
        "oci_key": optional_params.get("oci_key") or os.environ.get(_OCI_KEY_ENV),
        "oci_key_file": optional_params.get("oci_key_file")
        or os.environ.get(_OCI_KEY_FILE_ENV),
        "oci_compartment_id": optional_params.get("oci_compartment_id")
        or os.environ.get(_OCI_COMPARTMENT_ID_ENV),
    }


def get_oci_base_url(optional_params: dict, api_base: Optional[str] = None) -> str:
    """Return the OCI inference base URL, respecting any explicit api_base override."""
    if api_base:
        return api_base.rstrip("/")
    creds = resolve_oci_credentials(optional_params)
    region = creds["oci_region"]
    return f"https://inference.generativeai.{region}.oci.oraclecloud.com"


# ---------------------------------------------------------------------------
# Signing implementations (shared by chat, embed, and rerank configs)
# ---------------------------------------------------------------------------


def sign_with_oci_signer(
    headers: dict,
    optional_params: dict,
    request_data: dict,
    api_base: str,
) -> Tuple[dict, bytes]:
    """Sign a request using an OCI SDK Signer object passed in optional_params."""
    oci_signer = optional_params.get("oci_signer")
    body = json.dumps(request_data).encode("utf-8")
    method = str(optional_params.get("method", "POST")).upper()

    if method not in {"POST", "GET", "PUT", "DELETE", "PATCH"}:
        raise ValueError(f"Unsupported HTTP method: {method}")

    prepared_headers = {**headers}
    prepared_headers.setdefault("content-type", "application/json")
    prepared_headers.setdefault("content-length", str(len(body)))

    request_wrapper = OCIRequestWrapper(
        method=method, url=api_base, headers=prepared_headers, body=body
    )

    if oci_signer is None:
        raise ValueError("oci_signer cannot be None when calling sign_with_oci_signer")

    try:
        oci_signer.do_request_sign(request_wrapper, enforce_content_headers=True)
    except Exception as e:
        raise OCIError(
            status_code=500,
            message=(
                f"Failed to sign request with provided oci_signer: {str(e)}. "
                "The signer must implement the OCI SDK Signer interface with a "
                "do_request_sign(request, enforce_content_headers=True) method. "
                "See: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html"
            ),
        ) from e

    headers.update(request_wrapper.headers)
    return headers, body


def sign_with_manual_credentials(
    headers: dict,
    optional_params: dict,
    request_data: dict,
    api_base: str,
) -> Tuple[dict, bytes]:
    """Sign a request using manually provided OCI credentials (user/fingerprint/tenancy/key)."""
    creds = resolve_oci_credentials(optional_params)
    oci_user = creds["oci_user"]
    oci_fingerprint = creds["oci_fingerprint"]
    oci_tenancy = creds["oci_tenancy"]
    oci_key = creds["oci_key"]
    oci_key_file = creds["oci_key_file"]

    if (
        not oci_user
        or not oci_fingerprint
        or not oci_tenancy
        or not (oci_key or oci_key_file)
    ):
        raise OCIError(
            status_code=401,
            message=(
                "Missing required OCI credentials: oci_user, oci_fingerprint, oci_tenancy, "
                "and at least one of oci_key or oci_key_file. "
                "These can also be supplied via environment variables: "
                f"{_OCI_USER_ENV}, {_OCI_FINGERPRINT_ENV}, {_OCI_TENANCY_ENV}, {_OCI_KEY_ENV} (or {_OCI_KEY_FILE_ENV}). "
                "Alternatively, provide an oci_signer object from the OCI SDK."
            ),
        )

    method = str(optional_params.get("method", "POST")).upper()
    body = json.dumps(request_data).encode("utf-8")
    parsed = urlparse(api_base)
    path = parsed.path or "/"
    host = parsed.netloc

    date = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    content_type = headers.get("content-type", "application/json")
    content_length = str(len(body))
    x_content_sha256 = sha256_base64(body)

    headers_to_sign: Dict[str, str] = {
        "date": date,
        "host": host,
        "content-type": content_type,
        "content-length": content_length,
        "x-content-sha256": x_content_sha256,
    }

    signed_header_names = [
        "date",
        "(request-target)",
        "host",
        "content-length",
        "content-type",
        "x-content-sha256",
    ]
    signing_string = build_signature_string(
        method, path, headers_to_sign, signed_header_names
    )

    _require_cryptography()

    # Resolve the private key — prefer inline PEM content over file path
    oci_key_content: Optional[str] = None
    if oci_key:
        if not isinstance(oci_key, str):
            raise OCIError(
                status_code=400,
                message=(
                    f"oci_key must be a string containing the PEM private key content. "
                    f"Got type: {type(oci_key).__name__}"
                ),
            )
        oci_key_content = oci_key.replace("\\n", "\n").replace("\r\n", "\n")

    private_key = (
        load_private_key_from_str(oci_key_content)
        if oci_key_content
        else load_private_key_from_file(oci_key_file) if oci_key_file else None
    )

    if private_key is None:
        raise OCIError(
            status_code=400,
            message="Private key is required for OCI authentication. Provide either oci_key or oci_key_file.",
        )

    signature = private_key.sign(
        signing_string.encode("utf-8"),
        padding.PKCS1v15(),  # type: ignore[union-attr]
        hashes.SHA256(),  # type: ignore[union-attr]
    )
    signature_b64 = base64.b64encode(signature).decode()

    key_id = f"{oci_tenancy}/{oci_user}/{oci_fingerprint}"
    authorization = (
        'Signature version="1",'
        f'keyId="{key_id}",'
        'algorithm="rsa-sha256",'
        f'headers="{" ".join(signed_header_names)}",'
        f'signature="{signature_b64}"'
    )

    headers.update(
        {
            "authorization": authorization,
            "date": date,
            "host": host,
            "content-type": content_type,
            "content-length": content_length,
            "x-content-sha256": x_content_sha256,
        }
    )
    return headers, body


def sign_oci_request(
    headers: dict,
    optional_params: dict,
    request_data: dict,
    api_base: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    stream: Optional[bool] = None,
    fake_stream: Optional[bool] = None,
) -> Tuple[dict, bytes]:
    """
    Route to the appropriate OCI signing method based on what credentials are present.

    If ``oci_signer`` is in optional_params, use the OCI SDK signer object.
    Otherwise use manual RSA-SHA256 signing with explicit credentials (which can
    also be supplied via OCI_* environment variables).

    Returns:
        Tuple of (signed_headers, signed_body_bytes)
    """
    if optional_params.get("oci_signer") is not None:
        return sign_with_oci_signer(headers, optional_params, request_data, api_base)
    return sign_with_manual_credentials(
        headers, optional_params, request_data, api_base
    )


def validate_oci_environment(
    headers: dict,
    optional_params: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Populate common OCI request headers (content-type, user-agent).

    Full credential validation is deferred to signing time so that credentials
    supplied via environment variables are resolved at call time rather than
    at construction time.
    """
    headers.setdefault("content-type", "application/json")
    headers.setdefault("user-agent", f"litellm/{_litellm_version}")
    return headers


# ---------------------------------------------------------------------------
# JSON schema utilities for OCI tool definitions
#
# OCI Generative AI does not support JSON Schema extensions ($ref, $defs,
# anyOf).  Pydantic v2 emits all three for models with Optional fields or
# nested schemas.  The helpers below are ported from the official
# langchain-oracle reference implementation so that tool schemas are always
# valid before they reach the OCI endpoint.
# ---------------------------------------------------------------------------

# Mapping from JSON Schema type names to Python type names, as expected by
# the OCI Cohere API's CohereParameterDefinition.type field.
OCI_JSON_TO_PYTHON_TYPES: Dict[str, str] = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "integer": "int",
    "array": "List",
    "object": "Dict",
    "any": "any",
}


def resolve_oci_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline all ``$ref``/``$defs`` references — OCI does not support JSON Schema ``$ref``."""
    defs = schema.get("$defs", {})
    resolving_stack: set = set()

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                if ref.startswith("#/$defs/"):
                    key = ref.split("/")[-1]
                    if key in resolving_stack:
                        return {"type": "object"}  # break cycles
                    resolving_stack.add(key)
                    try:
                        return _resolve(defs.get(key, obj))
                    finally:
                        resolving_stack.discard(key)
                return obj  # external $ref — leave unchanged
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    resolved = _resolve(schema)
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
    return resolved


def resolve_oci_schema_anyof(obj: Any) -> Any:
    """Resolve Pydantic v2 ``Optional[T]`` → ``anyOf`` patterns.

    Pydantic v2 emits ``{"anyOf": [{"type": "T"}, {"type": "null"}]}`` for
    ``Optional[T]``.  OCI models don't understand ``anyOf``, so we pick the
    first non-null branch and merge top-level metadata into it.
    """
    if isinstance(obj, dict):
        if "anyOf" in obj and "type" not in obj:
            non_null = [
                t
                for t in obj["anyOf"]
                if not (isinstance(t, dict) and t.get("type") == "null")
            ]
            if non_null:
                resolved = {**obj, **non_null[0]}
                resolved.pop("anyOf", None)
                return resolve_oci_schema_anyof(resolved)
        return {k: resolve_oci_schema_anyof(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_oci_schema_anyof(item) for item in obj]
    return obj


def sanitize_oci_schema(schema: Any) -> Any:
    """Recursively remove OCI-incompatible fields from a JSON schema.

    Strips ``title`` keys, removes ``None``-valued ``default`` entries,
    normalises ``type: [T, "null"]`` list types, and ensures arrays carry an
    ``items`` definition.
    """
    if isinstance(schema, list):
        return [sanitize_oci_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    sanitized: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key == "default" and value is None:
            continue
        if key == "type":
            if value == "any":
                sanitized[key] = "object"
                continue
            if isinstance(value, list):
                non_null = [t for t in value if t != "null"]
                sanitized[key] = non_null[0] if non_null else "string"
                continue
        sanitized[key] = sanitize_oci_schema(value)

    if sanitized.get("type") == "array" and "items" not in sanitized:
        sanitized["items"] = {"type": "object"}

    required = sanitized.get("required")
    properties = sanitized.get("properties")
    if "required" in sanitized:
        if isinstance(required, list) and isinstance(properties, dict):
            sanitized["required"] = [
                f for f in required if isinstance(f, str) and f in properties
            ]
        elif not isinstance(required, list):
            sanitized["required"] = []

    return sanitized


def enrich_cohere_param_description(
    description: str, param_schema: Dict[str, Any]
) -> str:
    """Embed schema constraints into a Cohere parameter description.

    ``CohereParameterDefinition`` only has ``type``, ``description``, and
    ``isRequired``.  Rich constraints (``enum``, ``format``, ``minimum``,
    ``maximum``, ``pattern``) are appended to the description string so the
    model can still see and respect them.
    """
    parts = [description] if description else []
    if "enum" in param_schema:
        parts.append(f"Allowed values: {param_schema['enum']}")
    if "format" in param_schema:
        parts.append(f"Format: {param_schema['format']}")
    if "minimum" in param_schema or "maximum" in param_schema:
        range_parts = []
        if "minimum" in param_schema:
            range_parts.append(f"min={param_schema['minimum']}")
        if "maximum" in param_schema:
            range_parts.append(f"max={param_schema['maximum']}")
        parts.append(f"Range: {', '.join(range_parts)}")
    if "pattern" in param_schema:
        parts.append(f"Pattern: {param_schema['pattern']}")
    return ". ".join(parts) if parts else ""
