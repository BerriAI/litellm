from collections.abc import Mapping
from typing import Final

import litellm
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import optional_str
from litellm.types.tokenize import TokenizeResponse
from litellm.types.utils import is_litellm_owned_kwarg

_BODY_FIELDS: Final = frozenset(
    {
        "add_special_tokens",
        "project_id",
        "watsonx_project",
        "space_id",
        "api_version",
        "token",
        "zen_api_key",
    }
)


_PROJECTED_FIELDS: Final = frozenset({"model", "prompt", "timeout", "base_url", "extra_headers"})


def body_params(request: Mapping[str, object]) -> Mapping[str, object]:
    return {
        name: value
        for name, value in request.items()
        if value is not None
        and (name in _BODY_FIELDS or (name not in _PROJECTED_FIELDS and not is_litellm_owned_kwarg(name)))
    }


def response(value: Mapping[str, object]) -> TokenizeResponse:
    return TokenizeResponse.model_validate(value)


def map_failure(error: Exception, request: Mapping[str, object]) -> Exception:
    provider: Final = optional_str(request.get("custom_llm_provider")) or "tokenize"
    if isinstance(error, (TypeError, ValueError)):
        return litellm.BadRequestError(message=str(error), model=str(request["model"]), llm_provider=provider)
    return failures.map_native_failure(
        error,
        str(request["model"]),
        provider,
        request,
        optional_str(request.get("api_base")),
    )
