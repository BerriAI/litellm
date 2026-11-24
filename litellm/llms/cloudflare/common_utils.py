from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.secret_managers.main import (
    get_secret_str,
    normalize_nonempty_secret_str,
)

CLOUDFLARE_MISSING_API_BASE_MESSAGE: Final = (
    "Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID or pass api_base explicitly"
)
_RESPONSE_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def workers_ai_default_api_base() -> str | None:
    account_id: Final = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
    if account_id is None:
        return None
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"


def workers_ai_canonical_model(model: str) -> str:
    if model.startswith("@cf/"):
        return model
    return f"@cf/cloudflare/{model}"


def workers_ai_request_model(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1]


def workers_ai_run_url(api_base: str, model: str) -> str:
    normalized_api_base: Final = api_base.rstrip("/")
    if normalized_api_base.endswith("/ai/v1"):
        return join_api_base(normalized_api_base.removesuffix("/ai/v1"), f"ai/run/{model}")
    if normalized_api_base.endswith("/ai/run"):
        return f"{normalized_api_base}/{model}"
    return join_api_base(normalized_api_base, f"ai/run/{model}")


def unwrap_workers_ai_result(payload: object) -> object:
    try:
        response_mapping: Final = _RESPONSE_MAPPING_ADAPTER.validate_python(payload)
    except ValidationError:
        return payload
    if "answers" in response_mapping:
        return response_mapping
    try:
        return _RESPONSE_MAPPING_ADAPTER.validate_python(response_mapping.get("result"))
    except ValidationError:
        return response_mapping
