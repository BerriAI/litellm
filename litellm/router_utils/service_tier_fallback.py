from collections.abc import Mapping
from typing import Final

from litellm.exceptions import RateLimitError, Timeout, UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.hidden_params import get_hidden_params, set_hidden_params


class FlexTierFallback:
    pending_key: Final = "_service_tier_fallback_pending"

    @staticmethod
    def applies(request: Mapping[str, object], enabled: bool) -> bool:
        return enabled and request.get("service_tier") == "flex"

    @staticmethod
    def is_trigger(error: object) -> bool:
        return isinstance(error, RateLimitError | Timeout) or (
            isinstance(error, UnsupportedParamsError) and "service_tier" in str(error)
        )

    @classmethod
    def pending(cls, request: Mapping[str, object], field: str) -> Mapping[str, object]:
        metadata: Final = as_str_mapping(request.get(field)) or {}
        return {**request, field: {**metadata, cls.pending_key: {"active": True}}}

    @classmethod
    def finish(cls, request: Mapping[str, object], field: str) -> None:
        metadata: Final = as_str_mapping(request.get(field)) or {}
        marker: Final = metadata.get(cls.pending_key)
        if isinstance(marker, dict):
            marker.clear()

    @classmethod
    def downgraded(cls, request: Mapping[str, object], field: str) -> Mapping[str, object]:
        metadata: Final = as_str_mapping(request.get(field)) or {}
        return {
            **{key: value for key, value in request.items() if key not in ("service_tier", "litellm_logging_obj")},
            field: {
                **{key: value for key, value in metadata.items() if key != cls.pending_key},
                "service_tier_fallback": {"from": "flex", "to": "default"},
            },
        }

    @classmethod
    def suppress_failure(cls, logged: Mapping[str, object], error: object) -> bool:
        params: Final = as_str_mapping(logged.get("litellm_params")) or {}
        metadata: Final = (
            as_str_mapping(params.get("litellm_metadata"))
            or as_str_mapping(params.get("metadata"))
            or as_str_mapping(logged.get("litellm_metadata"))
            or as_str_mapping(logged.get("metadata"))
            or {}
        )
        marker: Final = as_str_mapping(metadata.get(cls.pending_key)) or {}
        return marker.get("active") is True and cls.is_trigger(error)

    @staticmethod
    def add_header(response: object) -> object:
        hidden: Final = get_hidden_params(response) or {}
        headers: Final = as_str_mapping(hidden.get("additional_headers")) or {}
        set_hidden_params(response, {**hidden, "additional_headers": {**headers, "x-litellm-service-tier": "default"}})
        return response
