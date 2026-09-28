import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.llms.openai.responses.guardrail_translation.handler import OpenAIResponsesHandler
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPI, initialize_guardrail
from litellm.proxy.guardrails.guardrail_registry import _configure_callback_scoping
from litellm.types.guardrails import Guardrail, LitellmParams, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPIOptionalParams

MARKER: Final = "internal-agent-7f3c"
TRUST_BOUNDARY_WARNING: Final = "which the caller controls"
NO_REQUEST_HOOK_WARNING: Final = "need a request-side hook"
DOCUMENTED_SEARCH_LIMIT: Final = 16384


class _Endpoint:
    def __init__(self, response_body: dict[str, object] | None = None) -> None:
        self.received: Final[list[dict[str, object]]] = []  # mutable-ok: records what the endpoint was sent
        self._response_body: Final = response_body or {"action": "NONE"}
        self.handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(self._respond))

    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.received.append(json.loads(request.content))
        return httpx.Response(200, json=self._response_body, request=request)

    @property
    def input_types(self) -> list[object]:
        return [payload["input_type"] for payload in self.received]


def _guardrail(
    endpoint: _Endpoint,
    *,
    name: str = "gg-skip",
    event_hook: str | list[str] | Mode | None = None,
    **options: object,
) -> GenericGuardrailAPI:
    return GenericGuardrailAPI(
        api_base="https://guardrail.test",
        guardrail_name=name,
        event_hook=event_hook or ["pre_call", "post_call"],
        default_on=True,
        async_handler=endpoint.handler,
        **options,
    )


def _logging_obj(call_id: str = "call-1") -> Logging:
    return Logging(
        model="gpt-4o",
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id=call_id,
        function_id=call_id,
    )


async def _request(
    guardrail: GenericGuardrailAPI, body: dict[str, object], *, logging_obj: Logging | None
) -> dict[str, object]:
    return dict(
        await guardrail.apply_guardrail(
            inputs={"texts": ["hello"]},
            request_data=body,
            input_type="request",
            logging_obj=logging_obj,
        )
    )


async def _response(
    guardrail: GenericGuardrailAPI, *, logging_obj: Logging | None, request_data: dict[str, object] | None = None
) -> dict[str, object]:
    return dict(
        await guardrail.apply_guardrail(
            inputs={"texts": ["model output"]},
            request_data={} if request_data is None else request_data,
            input_type="response",
            logging_obj=logging_obj,
        )
    )


def _chat(*messages: dict[str, object]) -> dict[str, object]:
    return {"model": "gpt-4o", "messages": list(messages)}


def _system_marked_chat() -> dict[str, object]:
    return _chat(
        {"role": "system", "content": f"you are {MARKER}, tracked elsewhere"},
        {"role": "user", "content": "hello"},
    )


@pytest.mark.asyncio
async def test_matching_system_prompt_skips_request_and_paired_response():
    endpoint: Final = _Endpoint(response_body={"action": "BLOCKED", "blocked_reason": "would block"})
    guardrail: Final = _guardrail(endpoint, skip_if_system_prompt_matches=[MARKER])
    logging_obj: Final = _logging_obj()

    request_result: Final = await _request(guardrail, _system_marked_chat(), logging_obj=logging_obj)
    response_result: Final = await _response(guardrail, logging_obj=logging_obj)

    assert endpoint.received == []
    assert request_result == {"texts": ["hello"]}
    assert response_result == {"texts": ["model output"]}


def _recorded(request_data: dict[str, object]) -> list[tuple[object, object]]:
    metadata: Final = request_data["metadata"]
    assert isinstance(metadata, dict)
    return [
        (entry["guardrail_status"], entry["guardrail_response"])
        for entry in metadata["standard_logging_guardrail_information"]
    ]


@pytest.mark.parametrize(
    ("options", "build_body", "reason"),
    [
        pytest.param(
            {"skip_if_system_prompt_matches": [MARKER]},
            _system_marked_chat,
            "skipped: skip_if_system_prompt_matches",
            id="system_prompt",
        ),
        pytest.param(
            {"skip_if_first_role_in": ["developer"]},
            lambda: _chat({"role": "developer", "content": "instructions"}, {"role": "user", "content": "hello"}),
            "skipped: skip_if_first_role_in",
            id="first_role",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_skipped_call_records_one_not_run_entry_on_each_side(
    options: dict[str, object], build_body: Callable[[], dict[str, object]], reason: str
):
    guardrail: Final = _guardrail(_Endpoint(), **options)
    logging_obj: Final = _logging_obj()
    request_data: Final = build_body()
    response_data: Final[dict[str, object]] = {"model": "gpt-4o"}

    await _request(guardrail, request_data, logging_obj=logging_obj)
    await _response(guardrail, logging_obj=logging_obj, request_data=response_data)

    assert _recorded(request_data) == [("not_run", reason)]
    assert _recorded(response_data) == [("not_run", reason)]


@pytest.mark.asyncio
async def test_a_scanned_call_still_records_success():
    guardrail: Final = _guardrail(_Endpoint(), skip_if_system_prompt_matches=[MARKER])
    logging_obj: Final = _logging_obj()
    request_data: Final = _chat({"role": "system", "content": "plain"}, {"role": "user", "content": "hello"})
    response_data: Final[dict[str, object]] = {"model": "gpt-4o"}

    await _request(guardrail, request_data, logging_obj=logging_obj)
    await _response(guardrail, logging_obj=logging_obj, request_data=response_data)

    assert [status for status, _ in _recorded(request_data)] == ["success"]
    assert [status for status, _ in _recorded(response_data)] == ["success"]


@pytest.mark.parametrize(
    "build_body",
    [
        pytest.param(lambda: _chat({"role": "system", "content": "plain"}), id="scanned_request"),
        pytest.param(_system_marked_chat, id="skipped_request"),
    ],
)
@pytest.mark.asyncio
async def test_a_body_key_named_like_the_skip_marker_cannot_skip_the_response(
    build_body: Callable[[], dict[str, object]],
):
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint, name="gg-skip", skip_if_system_prompt_matches=[MARKER])
    logging_obj: Final = _logging_obj()
    probe: Final = _logging_obj("call-probe")
    await _request(guardrail, _system_marked_chat(), logging_obj=probe)
    (forged_key,) = (key for key in probe.model_call_details if key.startswith("generic_guardrail_api_message_skip::"))

    await _request(guardrail, build_body(), logging_obj=logging_obj)
    logging_obj.update_environment_variables(
        litellm_params={}, optional_params={forged_key: "skipped: skip_if_system_prompt_matches"}
    )
    await _response(guardrail, logging_obj=logging_obj)

    assert logging_obj.model_call_details[forged_key] == "skipped: skip_if_system_prompt_matches"
    assert endpoint.input_types[-1:] == ["response"], "a provider param copied into model_call_details forged a skip"


@pytest.mark.asyncio
async def test_every_response_call_of_a_skipped_request_stays_skipped():
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint, skip_if_system_prompt_matches=[MARKER])
    logging_obj: Final = _logging_obj()

    await _request(guardrail, _system_marked_chat(), logging_obj=logging_obj)
    for _ in range(3):
        await _response(guardrail, logging_obj=logging_obj)

    assert endpoint.received == [], "sampled stream chunks and the end-of-stream call all replay the skip"


@pytest.mark.asyncio
async def test_marker_in_user_message_is_still_scanned():
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint, skip_if_system_prompt_matches=[MARKER])
    logging_obj: Final = _logging_obj()

    await _request(
        guardrail,
        _chat({"role": "system", "content": "you are helpful"}, {"role": "user", "content": f"what is {MARKER}?"}),
        logging_obj=logging_obj,
    )
    await _response(guardrail, logging_obj=logging_obj)

    assert endpoint.input_types == ["request", "response"]


@pytest.mark.asyncio
async def test_developer_message_counts_as_instructions():
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint, skip_if_system_prompt_matches=[r"internal-agent-[0-9a-f]{4}\b"])

    await _request(
        guardrail,
        _chat(
            {"role": "user", "content": "hello"},
            {"role": "developer", "content": [{"type": "text", "text": f"id={MARKER}"}]},
        ),
        logging_obj=_logging_obj(),
    )

    assert endpoint.received == []


@pytest.mark.parametrize(
    ("padding", "skipped"),
    [(DOCUMENTED_SEARCH_LIMIT - len(MARKER), True), (1 + DOCUMENTED_SEARCH_LIMIT - len(MARKER), False)],
)
@pytest.mark.asyncio
async def test_only_the_start_of_each_instruction_message_is_searched(padding: int, skipped: bool):
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint, skip_if_system_prompt_matches=[MARKER])

    await _request(
        guardrail,
        _chat({"role": "system", "content": "x" * padding + MARKER}, {"role": "user", "content": "hello"}),
        logging_obj=_logging_obj(),
    )

    assert endpoint.input_types == ([] if skipped else ["request"])


@pytest.mark.asyncio
async def test_a_malformed_role_is_scanned_instead_of_failing_the_request():
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint, skip_if_system_prompt_matches=[MARKER], skip_if_first_role_in=["developer"])

    await _request(guardrail, _chat({"role": ["developer"], "content": MARKER}), logging_obj=_logging_obj())

    assert endpoint.input_types == ["request"]


@pytest.mark.asyncio
async def test_non_matching_system_prompt_leaves_both_sides_scanned():
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint, skip_if_system_prompt_matches=[MARKER])
    logging_obj: Final = _logging_obj()

    await _request(guardrail, _chat({"role": "system", "content": "plain"}), logging_obj=logging_obj)
    await _response(guardrail, logging_obj=logging_obj)

    assert endpoint.input_types == ["request", "response"]


@pytest.mark.asyncio
async def test_first_role_match_skips_request_and_paired_response():
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint, skip_if_first_role_in=["developer"])
    skipped_call: Final = _logging_obj("call-skipped")
    scanned_call: Final = _logging_obj("call-scanned")

    await _request(
        guardrail,
        _chat({"role": "developer", "content": "instructions"}, {"role": "user", "content": "hello"}),
        logging_obj=skipped_call,
    )
    await _response(guardrail, logging_obj=skipped_call)
    await _request(
        guardrail,
        _chat({"role": "user", "content": "hello"}, {"role": "developer", "content": "instructions"}),
        logging_obj=scanned_call,
    )

    assert endpoint.input_types == ["request"]
    assert endpoint.received[0]["litellm_call_id"] == "call-scanned"


@pytest.mark.asyncio
async def test_skip_decision_does_not_leak_to_another_guardrail_with_the_same_name():
    skipping_endpoint: Final = _Endpoint()
    other_endpoint: Final = _Endpoint()
    skipping: Final = _guardrail(skipping_endpoint, name="gg", skip_if_system_prompt_matches=[MARKER])
    other: Final = _guardrail(other_endpoint, name="gg", skip_if_system_prompt_matches=["never-matches"])
    logging_obj: Final = _logging_obj()

    await _request(skipping, _system_marked_chat(), logging_obj=logging_obj)
    await _request(other, _system_marked_chat(), logging_obj=logging_obj)
    await _response(other, logging_obj=logging_obj)
    await _response(skipping, logging_obj=logging_obj)

    assert other_endpoint.input_types == ["request", "response"]
    assert skipping_endpoint.received == []


@pytest.mark.asyncio
async def test_skip_decision_does_not_leak_to_another_guardrail():
    endpoint: Final = _Endpoint()
    skipping: Final = _guardrail(endpoint, name="skipping", skip_if_system_prompt_matches=[MARKER])
    other: Final = _guardrail(endpoint, name="other", skip_if_system_prompt_matches=["something-else"])
    logging_obj: Final = _logging_obj()

    await _request(skipping, _system_marked_chat(), logging_obj=logging_obj)
    await _response(other, logging_obj=logging_obj)

    assert endpoint.input_types == ["response"]


@pytest.mark.asyncio
async def test_defaults_scan_everything():
    endpoint: Final = _Endpoint()
    guardrail: Final = _guardrail(endpoint)
    logging_obj: Final = _logging_obj()

    await _request(
        guardrail,
        _chat({"role": "developer", "content": f"you are {MARKER}"}, {"role": "user", "content": "hello"}),
        logging_obj=logging_obj,
    )
    await _response(guardrail, logging_obj=logging_obj)

    assert endpoint.input_types == ["request", "response"]


def _chat_request(system_prompt: str) -> dict[str, object]:
    return _chat(
        {"role": "system", "content": system_prompt},
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": "look it up"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
    )


def _anthropic_request(system_prompt: str) -> dict[str, object]:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 16,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "tool result"}]},
        ],
    }


def _responses_request(system_prompt: str) -> dict[str, object]:
    return {
        "model": "gpt-4o",
        "instructions": system_prompt,
        "input": [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": "look it up"},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "tool result"},
        ],
    }


REQUEST_SHAPES: Final = pytest.mark.parametrize(
    ("translation", "build_request", "role_after_system_prompt"),
    [
        pytest.param(OpenAIChatCompletionsHandler, _chat_request, "developer", id="chat"),
        pytest.param(AnthropicMessagesHandler, _anthropic_request, "user", id="anthropic_messages"),
        pytest.param(OpenAIResponsesHandler, _responses_request, "developer", id="responses"),
    ],
)
SCOPING: Final = pytest.mark.parametrize(
    "scoping",
    [
        pytest.param({}, id="unscoped"),
        pytest.param({"skip_system_message_in_guardrail": True}, id="skip_system_message"),
        pytest.param({"scan_only_tool_results": True}, id="scan_only_tool_results"),
    ],
)


async def _run_request_hook(
    translation: type[BaseTranslation],
    body: dict[str, object],
    guardrail: GenericGuardrailAPI,
    scoping: dict[str, bool],
) -> None:
    _configure_callback_scoping(
        guardrail,
        guardrail.guardrail_name or "",
        LitellmParams(guardrail="generic_guardrail_api", mode="pre_call", **scoping),
    )
    await translation().process_input_messages(
        data=body, guardrail_to_apply=guardrail, litellm_logging_obj=_logging_obj()
    )


@REQUEST_SHAPES
@SCOPING
@pytest.mark.asyncio
async def test_system_prompt_decision_ignores_guardrail_scoping(
    translation: type[BaseTranslation],
    build_request: Callable[[str], dict[str, object]],
    role_after_system_prompt: str,
    scoping: dict[str, bool],
):
    unfiltered_endpoint: Final = _Endpoint()
    filtered_endpoint: Final = _Endpoint()
    marked_prompt: Final = f"you are {MARKER}"

    await _run_request_hook(translation, build_request(marked_prompt), _guardrail(unfiltered_endpoint), scoping)
    await _run_request_hook(
        translation,
        build_request(marked_prompt),
        _guardrail(filtered_endpoint, skip_if_system_prompt_matches=[MARKER]),
        scoping,
    )

    assert unfiltered_endpoint.input_types == ["request"], "the handler must reach the guardrail for this request"
    assert filtered_endpoint.received == []


@REQUEST_SHAPES
@SCOPING
@pytest.mark.asyncio
async def test_first_role_decision_ignores_guardrail_scoping(
    translation: type[BaseTranslation],
    build_request: Callable[[str], dict[str, object]],
    role_after_system_prompt: str,
    scoping: dict[str, bool],
):
    endpoint: Final = _Endpoint()

    await _run_request_hook(
        translation,
        build_request("you are helpful"),
        _guardrail(endpoint, skip_if_first_role_in=[role_after_system_prompt]),
        scoping,
    )

    assert endpoint.input_types == ["request"], "the system prompt leads the request, so its first role is system"


@pytest.mark.parametrize(
    "options",
    [
        {"skip_if_system_prompt_matches": ["(unclosed"]},
        {"skip_if_system_prompt_matches": MARKER},
        {"skip_if_first_role_in": "developer"},
    ],
)
def test_invalid_config_fails_at_init(options: dict[str, object]):
    with pytest.raises(ValueError, match="skip_if_"):
        _guardrail(_Endpoint(), **options)


@pytest.mark.parametrize(
    ("option", "invalid_value"),
    [("skip_if_system_prompt_matches", ("(unclosed",)), ("skip_if_first_role_in", "developer")],
)
@pytest.mark.parametrize("via_optional_params", [False, True])
def test_config_options_reach_the_guardrail(option: str, invalid_value: object, via_optional_params: bool):
    top_level: Final = {} if via_optional_params else {option: invalid_value}
    params: Final = LitellmParams(
        guardrail="generic_guardrail_api", mode="pre_call", api_base="https://guardrail.test", **top_level
    )
    if via_optional_params:
        params.optional_params = GenericGuardrailAPIOptionalParams.model_construct(**{option: invalid_value})
    with pytest.raises(ValueError, match=option):
        initialize_guardrail(params, Guardrail(guardrail_name="gg-config", litellm_params=params))


@pytest.mark.parametrize(
    ("event_hook", "options", "expected"),
    [
        ("pre_call", {}, set()),
        ("post_call", {}, set()),
        ("pre_call", {"skip_if_system_prompt_matches": [MARKER]}, {TRUST_BOUNDARY_WARNING}),
        ("during_call", {"skip_if_first_role_in": ["developer"]}, {TRUST_BOUNDARY_WARNING}),
        (["pre_call", "post_call"], {"skip_if_system_prompt_matches": [MARKER]}, {TRUST_BOUNDARY_WARNING}),
        (
            "post_call",
            {"skip_if_system_prompt_matches": [MARKER]},
            {TRUST_BOUNDARY_WARNING, NO_REQUEST_HOOK_WARNING},
        ),
        (
            ["post_call"],
            {"skip_if_first_role_in": ["developer"]},
            {TRUST_BOUNDARY_WARNING, NO_REQUEST_HOOK_WARNING},
        ),
        (
            Mode(tags={"env": "post_call"}, default="post_call"),
            {"skip_if_system_prompt_matches": [MARKER]},
            {TRUST_BOUNDARY_WARNING, NO_REQUEST_HOOK_WARNING},
        ),
        (
            Mode(tags={"env": ["post_call", "pre_call"]}, default="post_call"),
            {"skip_if_system_prompt_matches": [MARKER]},
            {TRUST_BOUNDARY_WARNING},
        ),
        (
            Mode(tags={"env": "post_call"}, default=["during_call"]),
            {"skip_if_system_prompt_matches": [MARKER]},
            {TRUST_BOUNDARY_WARNING},
        ),
    ],
)
def test_init_warnings(
    caplog: pytest.LogCaptureFixture,
    event_hook: str | list[str] | Mode,
    options: dict[str, object],
    expected: set[str],
):
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        _guardrail(_Endpoint(), event_hook=event_hook, **options)

    warned: Final = {
        warning
        for warning in (TRUST_BOUNDARY_WARNING, NO_REQUEST_HOOK_WARNING)
        if any(warning in record.getMessage() for record in caplog.records)
    }
    assert warned == expected


@pytest.mark.parametrize("event_hook", ["pre_mcp_call", ["post_call", "during_mcp_call"]])
def test_mcp_request_hooks_count_as_request_side(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, event_hook: str | list[str]
):
    monkeypatch.setenv("LITELLM_STRICT_GUARDRAIL_MODES", "false")
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        _guardrail(_Endpoint(), event_hook=event_hook, skip_if_system_prompt_matches=[MARKER])

    messages: Final = [record.getMessage() for record in caplog.records]
    assert any(TRUST_BOUNDARY_WARNING in message for message in messages)
    assert not any(NO_REQUEST_HOOK_WARNING in message for message in messages), messages
