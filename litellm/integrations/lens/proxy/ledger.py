import sys
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.batch_utils import BATCH_COST_REQUEST_ID_SUFFIX, charged_poll_cost
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.videos.utils import VIDEO_COST_REQUEST_ID_SUFFIX

_SETTINGS: Final = TypeAdapter(dict[str, object])
_UNSECURED_RESPONSE_READS: Final = frozenset({"aget_responses", "acancel_responses"})


def _proxy_settings() -> Mapping[str, object] | None:
    proxy_server: Final = sys.modules.get("litellm.proxy.proxy_server")
    if proxy_server is None or getattr(proxy_server, "prisma_client", None) is None:
        return None
    try:
        return _SETTINGS.validate_python(getattr(proxy_server, "general_settings", None) or {})
    except ValidationError:
        return None


def ledger_request_id(kwargs: Mapping[str, object], response_obj: object) -> str | None:
    """The spend log id the ledger commits this request under, or None when no ledger commit will follow."""
    from litellm.proxy.spend_tracking.spend_logs_ledger_setting import spend_logs_ledger_enabled
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_spend_logs_id
    from litellm.proxy.utils import ProxyUpdateSpend

    settings: Final = _proxy_settings()
    if settings is None or not spend_logs_ledger_enabled(settings) or ProxyUpdateSpend.disable_spend_updates():
        return None
    call_type: Final = str(kwargs.get("call_type") or "")
    if (
        call_type in _UNSECURED_RESPONSE_READS
        and settings.get("disable_responses_id_security") is True
        and not kwargs.get("response_cost")
    ):
        return None
    litellm_params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
    metadata: Final = as_str_mapping(litellm_params.get("metadata")) or {}
    response_id: Final[object] = getattr(response_obj, "id", None)
    response: Final = as_str_mapping(response_obj) or {"id": response_id}
    request_id: Final = get_spend_logs_id(call_type, response, kwargs, metadata)
    if request_id is None:
        return None
    settles_once: Final = request_id.endswith((BATCH_COST_REQUEST_ID_SUFFIX, VIDEO_COST_REQUEST_ID_SUFFIX))
    return None if settles_once and not charged_poll_cost(kwargs) else request_id
