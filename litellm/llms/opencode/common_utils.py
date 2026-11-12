import re
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import provider_reported_usage_cost, set_response_cost_in_hidden_params
from litellm.litellm_core_utils.provider_affinity import add_provider_affinity_header
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse


def is_opencode_api_base(api_base: str | None) -> bool:
    return bool(
        api_base
        and re.fullmatch(
            r"https://opencode\.ai/(?:zen(?:/go)?|inference/(?:go/)?(?:openai|anthropic))(?:/v1)?/?", api_base
        )
    )


def go_model_info(model: str) -> Mapping[str, object]:
    import litellm

    raw: Final[object] = litellm.model_cost.get(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy mutable catalog, validated below
        f"opencode_go/{model.removeprefix('opencode_go/')}", {}
    )
    return TypeAdapter(Mapping[str, object]).validate_python(raw)


def go_endpoint(model: str) -> str:
    endpoints: Final = TypeAdapter(tuple[str, ...]).validate_python(
        go_model_info(model).get("supported_endpoints") or ()
    )
    return next(iter(endpoints), "/v1/chat/completions")


def go_api_base(api_base: str | None, model: str = "") -> str:
    endpoint: Final = go_endpoint(model)
    family: Final = (
        "anthropic/v1"
        if endpoint == "/v1/messages"
        else ("google/v1beta" if endpoint == "/v1beta/models" else "openai/v1")
    )
    return api_base or get_secret_str("OPENCODE_GO_API_BASE") or f"https://opencode.ai/inference/go/{family}"


def go_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("OPENCODE_GO_API_KEY") or get_secret_str("OPENCODE_API_KEY")


def go_headers(
    headers: Mapping[str, object],
    params: object,
    api_key: str | None,
    model: str,
) -> dict[str, object]:  # mutable-ok: protocol handlers mutate the returned request and headers
    import litellm

    resolved_key: Final = go_api_key(api_key)
    if not resolved_key:
        raise litellm.AuthenticationError(
            message="Set OPENCODE_API_KEY or pass api_key for OpenCode Go.",
            llm_provider="opencode_go",
            model=model,
        )
    return add_provider_affinity_header(
        {"Content-Type": "application/json", **headers, "Authorization": f"Bearer {resolved_key}"},
        params,
        "x-opencode-session",
    )


def go_reported_cost(usage: object) -> float | None:
    return provider_reported_usage_cost(usage, "opencode_go")


def attach_go_reported_cost(response: ModelResponse | ResponsesAPIResponse, usage: object) -> None:
    set_response_cost_in_hidden_params(response, go_reported_cost(usage))
