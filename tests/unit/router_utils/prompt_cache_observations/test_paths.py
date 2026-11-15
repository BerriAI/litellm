import pytest
from pydantic import JsonValue

from litellm.router_utils.prompt_cache_observations.paths import classify_path
from litellm.router_utils.prompt_cache_observations.types import (
    ClientSurface,
    ObservationPath,
    Runtime,
    UnassessedReason,
)
from litellm.rust_bridge.catalog import Route
from litellm.rust_bridge.response_metadata import mark_rust_response
from litellm.types.utils import ModelResponse

_BODY: dict[str, JsonValue] = {"messages": [{"role": "user", "content": "hi"}]}
_DICT_RESPONSE = {"id": "msg"}


def _classify(
    provider: str = "anthropic",
    model: str = "claude-test",
    call_type: str = "acompletion",
    body: dict[str, JsonValue] = _BODY,
    stream: bool = False,
    response: object = None,
    served_runtime: object = "python",
) -> ObservationPath | UnassessedReason:
    return classify_path(
        provider=provider,
        provider_model=model,
        call_type=call_type,
        body=body,
        stream=stream,
        response=ModelResponse() if response is None else response,
        served_runtime=served_runtime,
    )


@pytest.mark.parametrize(
    ("call_type", "surface"),
    [
        ("anthropic_messages", ClientSurface.MESSAGES),
        ("acompletion", ClientSurface.CHAT),
        ("aresponses", ClientSurface.RESPONSES),
    ],
)
def test_each_client_surface_on_native_anthropic_is_classified(call_type: str, surface: ClientSurface) -> None:
    assert _classify(call_type=call_type, stream=True) == ObservationPath(
        surface=surface, stream=True, runtime=Runtime.PYTHON
    )


@pytest.mark.parametrize(
    ("provider", "model", "call_type", "body", "reason"),
    [
        ("bedrock", "us.anthropic.claude-test-v1:0", "acompletion", _BODY, UnassessedReason.PROVIDER_OUT_OF_SCOPE),
        ("vertex_ai", "claude-test", "acompletion", _BODY, UnassessedReason.PROVIDER_OUT_OF_SCOPE),
        ("azure_ai", "claude-test", "acompletion", _BODY, UnassessedReason.PROVIDER_OUT_OF_SCOPE),
        ("openai", "gpt-test", "acompletion", _BODY, UnassessedReason.PROVIDER_OUT_OF_SCOPE),
        ("anthropic", "not-a-claude-model", "acompletion", _BODY, UnassessedReason.MODEL_OUT_OF_SCOPE),
        ("anthropic", "claude-test", "aembedding", _BODY, UnassessedReason.SURFACE_OUT_OF_SCOPE),
        ("anthropic", "claude-test", "acompletion", {"contents": []}, UnassessedReason.WIRE_SHAPE_UNRECOGNIZED),
    ],
)
def test_paths_that_cannot_be_classified_are_unassessed(
    provider: str, model: str, call_type: str, body: dict[str, JsonValue], reason: UnassessedReason
) -> None:
    assert _classify(provider, model, call_type, body) == reason


@pytest.mark.parametrize(
    ("call_type", "response", "served_runtime", "expected"),
    [
        ("acompletion", ModelResponse(), "python", Runtime.PYTHON),
        ("anthropic_messages", _DICT_RESPONSE, "rust", Runtime.RUST),
        ("acompletion", mark_rust_response(ModelResponse(), Route.CHAT_COMPLETIONS), None, Runtime.RUST),
        ("acompletion", ModelResponse(), None, UnassessedReason.RUNTIME_UNKNOWN),
        ("anthropic_messages", _DICT_RESPONSE, None, UnassessedReason.RUNTIME_UNKNOWN),
    ],
)
def test_runtime_comes_from_the_attempts_stamp_and_absence_is_never_python(
    call_type: str, response: object, served_runtime: object, expected: Runtime | UnassessedReason
) -> None:
    path = _classify(call_type=call_type, response=response, served_runtime=served_runtime)
    assert (path.runtime if isinstance(path, ObservationPath) else path) == expected
