from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Final, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator

from litellm._logging import verbose_router_logger

CHAT_ENDPOINT_BY_CALL: Final[Mapping[str, str]] = MappingProxyType(
    {
        "completion": "/v1/chat/completions",
        "acompletion": "/v1/chat/completions",
        "_completion": "/v1/chat/completions",
        "_acompletion": "/v1/chat/completions",
        "text_completion": "/v1/completions",
        "atext_completion": "/v1/completions",
        "_atext_completion": "/v1/completions",
        "anthropic_messages": "/v1/messages",
        "aresponses": "/v1/responses",
        "agenerate_content": "generateContent",
        "agenerate_content_stream": "streamGenerateContent",
        "acreate_interaction": "/v1beta/interactions",
        "litellm.interactions.main.acreate": "/v1beta/interactions",
        "litellm.interactions.main.create": "/v1beta/interactions",
        "_aresponses_websocket": "/v1/responses",
        "acompact_responses": "/v1/responses/compact",
    }
)


def _endpoint_path(endpoint: str) -> str:
    rooted: Final = "/" + endpoint.strip().strip("/").casefold()
    path: Final = rooted.removeprefix("/openai").removeprefix("/v1beta").removeprefix("/v1")
    return "/" + path.rsplit(":", 1)[-1] if ":" in path else path


CHAT_ENDPOINTS: Final = frozenset(_endpoint_path(endpoint) for endpoint in CHAT_ENDPOINT_BY_CALL.values())
_Deployment: Final = TypeVar("_Deployment", bound=Mapping[str, object])
_ENFORCE_FLAG: Final = TypeAdapter(bool | None)
_REQUEST_SELECTION: Final[ContextVar[tuple[str | None, str | None]]] = ContextVar(
    "router_request_endpoint", default=(None, None)
)


class _EndpointDeclaration(BaseModel):
    model_config = ConfigDict(frozen=True)

    enforce_supported_endpoints: bool | None = None
    supported_endpoints: tuple[str, ...] | None = None

    @field_validator("supported_endpoints", mode="before")
    @classmethod
    def _one_endpoint_as_a_list(cls, value: object) -> object:
        return (value,) if isinstance(value, str) else value


def endpoint_for_call(call_type: object, module: object = None) -> str | None:
    qualified_call: Final = f"{module}.{call_type}" if module == "litellm.interactions.main" else call_type
    return CHAT_ENDPOINT_BY_CALL.get(qualified_call) if isinstance(qualified_call, str) else None


@contextmanager
def endpoint_selection(call_type: object, module: object = None, model: object = None) -> Generator[None]:
    endpoint: Final = endpoint_for_call(call_type, module)
    token: Final = _REQUEST_SELECTION.set((endpoint, model if isinstance(model, str) else None))
    try:
        yield
    finally:
        _REQUEST_SELECTION.reset(token)


def selected_endpoint() -> str | None:
    return _REQUEST_SELECTION.get()[0]


def _enforced_endpoints(model_info: object) -> tuple[str, ...] | None:
    try:
        declaration: Final = _EndpointDeclaration.model_validate(model_info)
    except ValidationError:
        return None
    return declaration.supported_endpoints if declaration.enforce_supported_endpoints else None


def warn_on_unenforced_endpoint_flag(model_name: str, model_info: Mapping[str, object]) -> str | None:
    try:
        flagged: Final = _ENFORCE_FLAG.validate_python(model_info.get("enforce_supported_endpoints"))
    except ValidationError:
        return None
    if not flagged or _enforced_endpoints(model_info) is not None:
        return None
    warning: Final = (
        f"Deployment '{model_name}' sets enforce_supported_endpoints without a supported_endpoints list of "
        'strings, so litellm enforces nothing for it. Set supported_endpoints, e.g. ["/v1/embeddings"].'
    )
    verbose_router_logger.warning(warning)
    return warning
