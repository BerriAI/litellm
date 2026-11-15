from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.common_utils import is_claude_model_target
from litellm.types.utils import CallTypes

from .types import ClientSurface, ObservationPath, ProviderFamily, Runtime, UnassessedReason, WireShape

_PROVIDERS: Final = MappingProxyType({provider.value: provider for provider in ProviderFamily})
_SURFACES: Final = MappingProxyType(
    {
        CallTypes.anthropic_messages.value: ClientSurface.MESSAGES,
        CallTypes.aanthropic_messages.value: ClientSurface.MESSAGES,
        CallTypes.completion.value: ClientSurface.CHAT,
        CallTypes.acompletion.value: ClientSurface.CHAT,
        CallTypes.responses.value: ClientSurface.RESPONSES,
        CallTypes.aresponses.value: ClientSurface.RESPONSES,
    }
)
_BEDROCK_OPERATIONS: Final = MappingProxyType(
    {
        "converse": WireShape.BEDROCK_CONVERSE,
        "converse-stream": WireShape.BEDROCK_CONVERSE,
        "invoke": WireShape.ANTHROPIC_MESSAGES,
        "invoke-with-response-stream": WireShape.ANTHROPIC_MESSAGES,
    }
)
_RUST_RESPONSE_HEADER: Final = "x-litellm-rust"


def _wire(provider: ProviderFamily, api_base: str | None, body: Mapping[str, JsonValue]) -> WireShape | None:
    if provider is ProviderFamily.BEDROCK:
        operation: Final = (api_base or "").split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
        return _BEDROCK_OPERATIONS.get(operation)
    return WireShape.ANTHROPIC_MESSAGES if isinstance(body.get("messages"), list) else None


def _runtime(response: object, served_runtime: object) -> Runtime | None:
    """`Logging.pre_call` stamps every attempt Python sends and the Rust host stamps the attempts Rust sends, so
    an attempt without a stamp is unknown rather than Python. Rust's response header, when a callback can see
    it, agrees with the stamp."""
    headers: Final = as_str_mapping(get_hidden_params_dict(response).get("additional_headers"))
    if served_runtime == "rust" or (headers is not None and headers.get(_RUST_RESPONSE_HEADER) == "true"):
        return Runtime.RUST
    return Runtime.PYTHON if served_runtime == "python" else None


def classify_path(
    provider: str,
    provider_model: str,
    call_type: str,
    api_base: str | None,
    body: Mapping[str, JsonValue],
    stream: bool,
    response: object,
    served_runtime: object,
) -> ObservationPath | UnassessedReason:
    family: Final = _PROVIDERS.get(provider)
    if family is None:
        return UnassessedReason.PROVIDER_OUT_OF_SCOPE
    if not is_claude_model_target(provider, provider_model):
        return UnassessedReason.MODEL_OUT_OF_SCOPE
    surface: Final = _SURFACES.get(call_type)
    if surface is None:
        return UnassessedReason.SURFACE_OUT_OF_SCOPE
    wire: Final = _wire(family, api_base, body)
    if wire is None:
        return UnassessedReason.WIRE_SHAPE_UNRECOGNIZED
    runtime: Final = _runtime(response, served_runtime)
    if runtime is None:
        return UnassessedReason.RUNTIME_UNKNOWN
    return ObservationPath(provider=family, wire=wire, surface=surface, stream=stream, runtime=runtime)
