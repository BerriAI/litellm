import asyncio
import inspect
from collections.abc import Callable, Iterator, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest

import litellm
from litellm.llms.anthropic.pass_through.messages import handler as python_messages
from litellm.messages.dispatch import ANTHROPIC_MESSAGES, ANTHROPIC_MESSAGES_HANDLER
from litellm.proxy import proxy_server
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_RESPONSE, MESSAGES_TEXT
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, recording_service
from tests.test_litellm_rust.support.served import assert_served_natively

pytest.importorskip("litellm.rust_bridge._native")

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

RUST: Final = (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)
PYTHON_TEXT: Final = "served by python"
CLAUDE: Final = "anthropic/claude-sonnet-5"
DANGEROUS_TOOL_USE: Final = [{"type": "dangerous_tool_use", "classifier_context": {"permission_mode": "auto"}}]
CLASSIFIER_SETTINGS: Final = {"safeguards_classifier_model": "gpt-oss-120b"}


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    with recording_service() as recording:
        recording.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
        yield recording


def served_text(response: object) -> object:
    assert isinstance(response, dict)
    content: Final = cast(list[Mapping[str, object]], response["content"])
    return content[0]["text"]


def call(model: str, **kwargs: object) -> Mapping[str, object]:
    return {"model": model, "messages": list(MESSAGES), "max_tokens": 16, **kwargs}


def upstream(server: RecordingServer, **kwargs: object) -> Mapping[str, object]:
    return {"api_key": "test-key", "api_base": server.base_url, **kwargs}


def assert_served_by(server: RecordingServer, response: object, *, python: bool) -> None:
    if python:
        server.expected_requests = 0
        assert served_text(response) == PYTHON_TEXT
        return
    assert served_text(response) == MESSAGES_TEXT
    assert_served_natively(server, response)


SERVED: Final = pytest.mark.parametrize(
    ("args", "kwargs", "python"),
    (
        pytest.param((16, list(MESSAGES), CLAUDE), {}, False, id="positional-claude"),
        pytest.param((), call(CLAUDE), False, id="keyword-claude"),
        pytest.param((), call("vertex_ai/gemini-2.5-pro"), True, id="vertex-gemini"),
        pytest.param((), call("gemini-2.5-pro", custom_llm_provider="vertex_ai"), True, id="vertex-provider"),
    ),
)


@SERVED
def test_messages_handler_serves_each_call_from_the_expected_backend(
    server: RecordingServer, args: tuple[object, ...], kwargs: Mapping[str, object], python: bool
) -> None:
    extra: Final = {"mock_response": PYTHON_TEXT} if python else {}
    response: Final = ANTHROPIC_MESSAGES_HANDLER.run(args, upstream(server, **kwargs, **extra), rules=RUST)
    assert_served_by(server, response, python=python)


@pytest.mark.asyncio
@SERVED
async def test_messages_serves_each_call_from_the_expected_backend(
    server: RecordingServer, args: tuple[object, ...], kwargs: Mapping[str, object], python: bool
) -> None:
    extra: Final = {"mock_response": PYTHON_TEXT} if python else {}
    response: Final = await ANTHROPIC_MESSAGES.arun(args, upstream(server, **kwargs, **extra), rules=RUST)
    assert_served_by(server, response, python=python)


CONTAINER: Final = "container_positional"


@pytest.mark.asyncio
async def test_handler_with_the_is_async_flag_binds_positional_arguments_by_the_sync_signature(
    server: RecordingServer,
) -> None:
    positional: Final = (16, list(MESSAGES), CLAUDE, *(None,) * 10, CONTAINER)

    pending: Final = ANTHROPIC_MESSAGES_HANDLER.run(positional, upstream(server, is_async=True), rules=RUST)

    assert asyncio.iscoroutine(pending)
    assert_served_by(server, await pending, python=False)
    body: Final = server.requests[0].body
    assert isinstance(body, dict)
    assert body["container"] == CONTAINER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "kwargs", "general_settings", "python"),
    (
        ("deepseek/deepseek-v4-flash", {"safeguards": DANGEROUS_TOOL_USE}, CLASSIFIER_SETTINGS, True),
        (
            "deepseek-v4-flash",
            {"safeguards": DANGEROUS_TOOL_USE, "custom_llm_provider": "deepseek"},
            CLASSIFIER_SETTINGS,
            True,
        ),
        ("deepseek/deepseek-v4-flash", {"safeguards": DANGEROUS_TOOL_USE}, {}, False),
        ("deepseek/deepseek-v4-flash", {}, CLASSIFIER_SETTINGS, False),
        ("deepseek/deepseek-v4-flash", {"safeguards": [{"type": "other"}]}, CLASSIFIER_SETTINGS, False),
        (CLAUDE, {"safeguards": DANGEROUS_TOOL_USE}, CLASSIFIER_SETTINGS, False),
    ),
)
async def test_python_keeps_the_requests_it_answers_with_safeguard_results(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    kwargs: Mapping[str, object],
    general_settings: Mapping[str, object],
    python: bool,
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", general_settings)
    extra: Final = {"mock_response": PYTHON_TEXT} if python else {}
    response: Final = await ANTHROPIC_MESSAGES.arun((), upstream(server, **call(model, **kwargs), **extra), rules=RUST)
    assert_served_by(server, response, python=python)


def test_public_signature_is_the_legacy_signature() -> None:
    public_messages: Final = cast(Callable[..., object], litellm.anthropic_messages_handler)
    legacy_messages: Final = cast(Callable[..., object], python_messages.anthropic_messages_handler)
    public_amessages: Final = cast(Callable[..., object], litellm.anthropic_messages)
    legacy_amessages: Final = cast(Callable[..., object], python_messages.anthropic_messages)
    assert inspect.signature(public_messages) == inspect.signature(legacy_messages)
    assert inspect.signature(public_amessages) == inspect.signature(legacy_amessages)
