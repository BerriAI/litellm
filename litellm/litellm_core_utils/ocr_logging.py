from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal

import httpx

from litellm.constants import request_timeout
from litellm.rust_bridge.timeouts import timeout_to_seconds


@dataclass(frozen=True, slots=True)
class OCRRequestMetadata:
    model: str
    custom_llm_provider: str | None
    litellm_call_id: str | None
    document_type: Literal["document_url", "image_url", "file", "unknown"]
    timeout: float | None

    @classmethod
    def from_request(
        cls, model: str, provider: object, call_id: object, document: object, timeout: object
    ) -> "OCRRequestMetadata":
        kind: Final = document.get("type") if isinstance(document, Mapping) else None
        return cls(
            model=model,
            custom_llm_provider=provider if isinstance(provider, str) else None,
            litellm_call_id=call_id if isinstance(call_id, str) else None,
            document_type=(
                "document_url"
                if kind == "document_url"
                else "image_url"
                if kind == "image_url"
                else "file"
                if kind == "file"
                else "unknown"
            ),
            timeout=timeout_to_seconds(timeout)
            if isinstance(timeout, (int, float, httpx.Timeout))
            else request_timeout,
        )

    def as_dict(self) -> dict[str, str | float | None]:
        return {
            "model": self.model,
            "custom_llm_provider": self.custom_llm_provider,
            "litellm_call_id": self.litellm_call_id,
            "document_type": self.document_type,
            "timeout": self.timeout,
        }


@dataclass(frozen=True, slots=True)
class OCRLoggingError(Exception):
    error_type: str
    status_code: int | None

    def __str__(self) -> str:
        return f"{self.error_type}: OCR request failed"


def ocr_error_for_logging(error: Exception) -> OCRLoggingError:
    status: Final = getattr(error, "status_code", None)
    return OCRLoggingError(type(error).__name__, status if isinstance(status, int) else None)


_OCR_IDENTITY_FIELDS: Final = frozenset(
    {
        "user_api_key_hash",
        "user_api_key_alias",
        "user_api_key_user_id",
        "user_api_key_team_id",
        "user_api_key_team_alias",
        "user_api_key_org_id",
        "user_api_key_org_alias",
        "user_api_key_project_id",
        "user_api_key_project_alias",
        "user_api_key_end_user_id",
        "model_group",
        "deployment",
        "team_id",
        "team_alias",
    }
)


def ocr_logging_context(params: Mapping[str, object]) -> Mapping[str, object]:
    from litellm.types.utils import CustomPricingLiteLLMParams

    metadata: Final = params.get("metadata") or params.get("litellm_metadata")
    model_info: Final = metadata.get("model_info") if isinstance(metadata, Mapping) else None
    model_id: Final = model_info.get("id") if isinstance(model_info, Mapping) else None
    identity: Final = (
        dict((key, value) for key, value in metadata.items() if key in _OCR_IDENTITY_FIELDS and isinstance(value, str))
        if isinstance(metadata, Mapping)
        else MappingProxyType({})
    )
    return MappingProxyType(
        dict(
            chain(
                (
                    ("metadata", {**identity, "model_info": {"id": model_id}})
                    if isinstance(model_id, str)
                    else ("metadata", identity),
                )
                if identity or isinstance(model_id, str)
                else (),
                (
                    (key, value)
                    for key, value in params.items()
                    if key in CustomPricingLiteLLMParams.model_fields and isinstance(value, (int, float))
                ),
                (("no_log", params["no_log"]),) if isinstance(params.get("no_log"), bool) else (),
            )
        )
    )
