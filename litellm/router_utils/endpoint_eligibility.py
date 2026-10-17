from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Final, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.exceptions import BadRequestError

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
CHAT_ENDPOINTS: Final = frozenset((*CHAT_ENDPOINT_BY_CALL.values(), "/chat/completions", "/completions", "/responses"))
_Deployment: Final = TypeVar("_Deployment", bound=Mapping[str, object])
_REQUEST_SELECTION: Final[ContextVar[tuple[str | None, str | None]]] = ContextVar(
    "router_request_endpoint", default=(None, None)
)


class _EndpointDeclaration(BaseModel):
    model_config = ConfigDict(frozen=True)

    enforce_supported_endpoints: bool = False
    supported_endpoints: tuple[str, ...] | None = None


class UnsupportedEndpointError(BadRequestError):
    def __init__(self, message: str, model: str, llm_provider: str = "") -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # inherited public exception constructor is untyped
            message=message, model=model, llm_provider=llm_provider
        )
        self.message = message
        self.type = "invalid_request_error"
        self.param = "model"
        self.code = "400"


@contextmanager
def endpoint_selection(call_type: object, module: object = None, model: object = None) -> Generator[None]:
    qualified_call: Final = f"{module}.{call_type}" if module == "litellm.interactions.main" else call_type
    endpoint: Final = CHAT_ENDPOINT_BY_CALL.get(qualified_call) if isinstance(qualified_call, str) else None
    token: Final = _REQUEST_SELECTION.set((endpoint, model if isinstance(model, str) else None))
    try:
        yield
    finally:
        _REQUEST_SELECTION.reset(token)


def non_chat_endpoints(deployment: Mapping[str, object]) -> frozenset[str] | None:
    try:
        declaration: Final = _EndpointDeclaration.model_validate(deployment.get("model_info"))
    except ValidationError:
        return None
    if not declaration.enforce_supported_endpoints or declaration.supported_endpoints is None:
        return None
    endpoints: Final = frozenset(declaration.supported_endpoints)
    return endpoints if endpoints.isdisjoint(CHAT_ENDPOINTS) else None


def eligible_deployments(
    model: str, deployments: Sequence[_Deployment]
) -> tuple[_Deployment, ...] | UnsupportedEndpointError:
    endpoint, requested_model = _REQUEST_SELECTION.get()
    if endpoint is None or not deployments:
        return tuple(deployments)
    eligible: Final = tuple(deployment for deployment in deployments if non_chat_endpoints(deployment) is None)
    if eligible:
        return eligible
    requested: Final = requested_model or model
    supported: Final = ", ".join(
        sorted(frozenset[str]().union(*(non_chat_endpoints(deployment) or frozenset() for deployment in deployments)))
    )
    return UnsupportedEndpointError(
        message=f"{endpoint} is not supported by model={requested}. Supported endpoints: {supported or 'none'}",
        model=model,
        llm_provider="",
    )
