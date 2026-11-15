import pytest
from pydantic import JsonValue

from litellm.router_utils.prompt_cache_observations.paths import classify_path
from litellm.router_utils.prompt_cache_observations.types import (
    ClientSurface,
    ObservationPath,
    ProviderFamily,
    Runtime,
    UnassessedReason,
    WireShape,
)
from litellm.rust_bridge.catalog import Route
from litellm.rust_bridge.response_metadata import mark_rust_response
from litellm.types.utils import ModelResponse

_BODY: dict[str, JsonValue] = {"messages": [{"role": "user", "content": "hi"}]}
_BEDROCK_BASE = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-test-v1:0"


def _classify(
    provider: str = "anthropic",
    model: str = "claude-test",
    call_type: str = "acompletion",
    api_base: str | None = None,
    body: dict[str, JsonValue] = _BODY,
    stream: bool = False,
    response: object = None,
    served_runtime: object = "python",
) -> ObservationPath | UnassessedReason:
    return classify_path(
        provider=provider,
        provider_model=model,
        call_type=call_type,
        api_base=api_base,
        body=body,
        stream=stream,
        response=ModelResponse() if response is None else response,
        served_runtime=served_runtime,
    )


@pytest.mark.parametrize(
    ("api_base", "wire"),
    [
        (f"{_BEDROCK_BASE}/converse", WireShape.BEDROCK_CONVERSE),
        (f"{_BEDROCK_BASE}/converse-stream", WireShape.BEDROCK_CONVERSE),
        (f"{_BEDROCK_BASE}/invoke", WireShape.ANTHROPIC_MESSAGES),
        (f"{_BEDROCK_BASE}/invoke-with-response-stream", WireShape.ANTHROPIC_MESSAGES),
    ],
)
def test_bedrock_wire_shape_follows_the_operation_actually_called(api_base: str, wire: WireShape) -> None:
    assert _classify(provider="bedrock", model="us.anthropic.claude-test-v1:0", api_base=api_base) == ObservationPath(
        provider=ProviderFamily.BEDROCK, wire=wire, surface=ClientSurface.CHAT, stream=False, runtime=Runtime.PYTHON
    )


_CONVERSE_COUNT = f"{_BEDROCK_BASE}/count-tokens"
_DICT_RESPONSE = {"id": "msg"}


@pytest.mark.parametrize(
    ("provider", "model", "call_type", "api_base", "body", "response", "reason"),
    [
        ("openai", "gpt-test", "acompletion", None, _BODY, None, UnassessedReason.PROVIDER_OUT_OF_SCOPE),
        ("azure_ai", "mistral-large", "acompletion", None, _BODY, None, UnassessedReason.MODEL_OUT_OF_SCOPE),
        ("vertex_ai", "gemini-test", "acompletion", None, _BODY, None, UnassessedReason.MODEL_OUT_OF_SCOPE),
        ("anthropic", "claude-test", "aembedding", None, _BODY, None, UnassessedReason.SURFACE_OUT_OF_SCOPE),
        (
            "bedrock",
            "us.anthropic.claude-test-v1:0",
            "acompletion",
            _CONVERSE_COUNT,
            _BODY,
            None,
            UnassessedReason.WIRE_SHAPE_UNRECOGNIZED,
        ),
        (
            "vertex_ai",
            "claude-test",
            "acompletion",
            None,
            {"contents": []},
            None,
            UnassessedReason.WIRE_SHAPE_UNRECOGNIZED,
        ),
    ],
)
def test_paths_that_cannot_be_classified_are_unassessed(
    provider: str,
    model: str,
    call_type: str,
    api_base: str | None,
    body: dict[str, JsonValue],
    response: object,
    reason: UnassessedReason,
) -> None:
    assert _classify(provider, model, call_type, api_base, body, response=response) == reason


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
