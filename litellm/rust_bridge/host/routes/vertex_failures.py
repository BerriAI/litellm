"""The exceptions Python's Vertex AI token acquisition raises, built from the stage at which Rust's failed."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import TypeAdapter, ValidationError

from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import GOOGLE_IMPORT_ERROR_MESSAGE, VertexBase

_Stage = Literal["no_default_credentials", "token_rejected", "token_unreachable", "unloadable"]
_GCP_CREDENTIAL_FAILURE: Final = TypeAdapter(tuple[_Stage, str])


def credential_failure(marker: object, credentials_configured: bool) -> Exception | None:
    """Python loads configured `vertex_credentials` itself and raises `ValueError` when they do not load,
    and leaves anything else to `google.auth.default`, which raises `DefaultCredentialsError`."""
    try:
        stage, text = _GCP_CREDENTIAL_FAILURE.validate_python(marker)
    except ValidationError:
        return None
    try:
        from google.auth import exceptions as google_auth
        from google.auth._default import (  # pyright: ignore[reportPrivateUsage]  # google.auth.default's own text
            _CLOUD_SDK_MISSING_CREDENTIALS,
        )
    except ImportError:
        return ImportError(GOOGLE_IMPORT_ERROR_MESSAGE)
    match stage:
        case "no_default_credentials":
            return google_auth.DefaultCredentialsError(_CLOUD_SDK_MISSING_CREDENTIALS)
        case "token_rejected":
            return google_auth.RefreshError(text)
        case "token_unreachable":
            return google_auth.TransportError(text)
        case "unloadable":
            return ValueError(text) if credentials_configured else google_auth.DefaultCredentialsError(text)


def default_base(model: str, location: str | None) -> str:
    """The base Python sends a Vertex AI partner call to when the call names no `api_base`."""
    return get_vertex_base_url(VertexBase.get_vertex_region(location, model))
