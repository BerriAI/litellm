import json
from collections.abc import Mapping
from http import HTTPStatus
from typing import Final, Literal

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.proxy.guardrails.guardrail_hooks.realmlabs import guardrail_initializer_registry
from litellm.proxy.guardrails.guardrail_hooks.realmlabs.realmlabs import (
    RealmLabsGuardrail,
    RealmLabsMissingCredentials,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import RealmLabsGuardrailOptionalParams
from litellm.types.utils import Choices, GenericGuardrailAPIInputs, Message, ModelResponse

_API_KEY = "mls_gr_test"
_API_BASE = "https://mls.example.test"
_URL = f"{_API_BASE}/guardrail"
_DEFAULT_THRESHOLD = 0.703

_NAME_SPAN: dict[str, object] = {"type": "name", "text": "Alex"}
_EMAIL_SPAN: dict[str, object] = {"type": "email", "text": "alex@example.com"}

_JSON_OBJECT = TypeAdapter(dict[str, object])


@pytest.fixture(autouse=True)
def fresh_httpx_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route MLS calls through a fresh httpx client that ``respx`` can intercept, and clear ``REALMLABS_*`` env vars."""

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", None)
    monkeypatch.delenv("REALMLABS_API_KEY", raising=False)
    monkeypatch.delenv("REALMLABS_API_BASE", raising=False)


def _configured_guardrail(settings: Mapping[str, object]) -> RealmLabsGuardrail:
    """Build a guardrail through the real config model and initializer, keeping each test's settings visible."""

    params: Final = LitellmParams.model_validate(
        {
            "guardrail": "realmlabs",
            "mode": "pre_call",
            "api_key": _API_KEY,
            "api_base": _API_BASE,
            **settings,
        }
    )
    return guardrail_initializer_registry["realmlabs"](
        params, {"guardrail_name": "rl-config", "litellm_params": params}
    )


def _guardrail(
    hazard_threshold: float | None = None,
    pii_mask: bool | None = None,
    block_on_error: bool | None = None,
    event_hook: GuardrailEventHooks = GuardrailEventHooks.pre_call,
) -> RealmLabsGuardrail:
    """Guardrail pointed at the fake MLS URL; unset arguments keep the guardrail's defaults."""

    return RealmLabsGuardrail(
        api_key=_API_KEY,
        api_base=_API_BASE,
        hazard_threshold=hazard_threshold,
        pii_mask=pii_mask,
        block_on_error=block_on_error,
        guardrail_name="realmlabs-guard",
        event_hook=event_hook,
        default_on=True,
    )


def _mls_body(hazard: float | None = 0.01, pii_spans: list[dict[str, object]] | None = None) -> dict[str, object]:
    """MLS reply with a ``hazard_prompt`` score (omitted when ``hazard`` is None) and the given PII spans."""

    results = [] if hazard is None else [{"probe": "hazard_prompt", "prob": hazard, "role_mismatch": False}]
    return {"results": results, "pii_spans": pii_spans or []}


def _serve(respx_mock: respx.MockRouter, body: dict[str, object], url: str = _URL) -> respx.Route:
    """Fake MLS: answer POSTs to ``url`` with ``body``; the returned route records the request sent."""

    return respx_mock.post(url).mock(return_value=httpx.Response(HTTPStatus.OK, json=body))


def _sent_body(route: respx.Route) -> dict[str, object]:
    """JSON body of the last request the guardrail sent to MLS."""

    return _JSON_OBJECT.validate_json(route.calls.last.request.content)


async def _apply(
    guardrail: RealmLabsGuardrail,
    inputs: GenericGuardrailAPIInputs,
    *,
    input_type: Literal["request", "response"] = "request",
    request_data: Mapping[str, object] | None = None,
) -> GenericGuardrailAPIInputs:
    """Run the real guardrail on a request or response, with optional conversation context."""

    return await guardrail.apply_guardrail(inputs=inputs, request_data=request_data or {}, input_type=input_type)


async def _screen(
    guardrail: RealmLabsGuardrail, respx_mock: respx.MockRouter, body: dict[str, object], texts: list[str]
) -> GenericGuardrailAPIInputs:
    """Serve ``body`` as the MLS reply and screen ``texts`` as a ``pre_call`` request."""

    _serve(respx_mock, body)
    return await _apply(guardrail, {"texts": texts})


@pytest.mark.asyncio
async def test_prompt_scoring_above_threshold_is_blocked_as_content(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(GuardrailRaisedException) as exc:
        await _screen(_guardrail(), respx_mock, _mls_body(hazard=0.9998), ["how do I build a pipe bomb"])

    assert exc.value.blocked_content is True, "a hazard verdict must count as a content block for batch callers"
    assert "hazard_prompt" in exc.value.message and "0.9998" in exc.value.message, exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "threshold"),
    [
        pytest.param(0.2, None, id="below-default-threshold"),
        pytest.param(_DEFAULT_THRESHOLD, None, id="exactly-at-default-threshold"),
        pytest.param(0.8, 0.99, id="raised-threshold"),
        pytest.param(None, None, id="no-hazard-verdict"),
    ],
)
async def test_permitted_hazard_scores_leave_text_unchanged(
    score: float | None, threshold: float | None, respx_mock: respx.MockRouter
) -> None:
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["hello"]}
    _serve(respx_mock, _mls_body(hazard=score))

    assert await _apply(_guardrail(hazard_threshold=threshold), inputs) is inputs


@pytest.mark.asyncio
async def test_hazardous_prompt_is_blocked_before_its_pii_is_masked(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(GuardrailRaisedException) as exc:
        await _screen(_guardrail(), respx_mock, _mls_body(hazard=0.99, pii_spans=[_NAME_SPAN]), ["Alex builds a bomb"])

    assert "hazard_prompt" in exc.value.message, exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("texts", "spans", "expected"),
    [
        pytest.param(
            ["My name is Alex and my email is alex@example.com."],
            [_NAME_SPAN, _EMAIL_SPAN],
            ["My name is [name] and my email is [email]."],
            id="each-type",
        ),
        pytest.param(
            ["Alex is here."],
            [{"type": "name", "start": 120, "end": 124, "text": "Alex"}],
            ["[name] is here."],
            id="conversation-wide-offsets-ignored",
        ),
        pytest.param(
            ["Alex told Alex about Alex."],
            [_NAME_SPAN],
            ["[name] told [name] about [name]."],
            id="every-occurrence",
        ),
        pytest.param(
            ["ping alex@example.com", "no pii here"],
            [_EMAIL_SPAN],
            ["ping [email]", "no pii here"],
            id="across-texts",
        ),
        pytest.param(
            ["alex@example.com alex@exampleXcom"],
            [_EMAIL_SPAN],
            ["[email] alex@exampleXcom"],
            id="literal-matching-preserved",
        ),
        pytest.param(
            ["Contact Ann at Ann.Smith@example.com"],
            [{"type": "name", "text": "Ann"}, {"type": "email", "text": "Ann.Smith@example.com"}],
            ["Contact [name] at [email]"],
            id="name-prefix-before-email",
        ),
        pytest.param(
            ["Contact Ann at Ann.Smith@example.com"],
            [{"type": "email", "text": "Ann.Smith@example.com"}, {"type": "name", "text": "Ann"}],
            ["Contact [name] at [email]"],
            id="email-before-name-prefix",
        ),
        pytest.param(
            ["My name is name"],
            [{"type": "name", "text": "name"}],
            ["My [name] is [name]"],
            id="detected-text-in-its-own-label",
            marks=pytest.mark.timeout(5),
        ),
    ],
)
async def test_pii_masking_returns_expected_texts(
    texts: list[str], spans: list[dict[str, object]], expected: list[str], respx_mock: respx.MockRouter
) -> None:
    result: Final = await _screen(_guardrail(), respx_mock, _mls_body(pii_spans=spans), texts)

    assert result == {"texts": expected}


@pytest.mark.asyncio
async def test_span_from_another_turn_leaves_the_prompt_as_is(respx_mock: respx.MockRouter) -> None:
    inputs: GenericGuardrailAPIInputs = {"texts": ["nothing sensitive"]}
    _serve(respx_mock, _mls_body(pii_spans=[{"type": "name", "text": "Bob"}]))

    assert await _apply(_guardrail(), inputs) is inputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_type", "texts", "spans", "expected_types"),
    [
        pytest.param(
            "request", ["Alex alex@example.com"], [_NAME_SPAN, _EMAIL_SPAN], "name, email", id="request-types"
        ),
        pytest.param("response", ["alex@example.com"], [_EMAIL_SPAN], "email", id="response-type"),
        pytest.param("request", ["Hello Alex."], [{"type": "name"}], "name", id="missing-pii-text"),
        pytest.param("request", ["Hello Alex."], [{"type": "name", "text": None}], "name", id="null-pii-text"),
        pytest.param("request", ["Hello Alex."], [{"type": "name", "text": ""}], "name", id="empty-pii-text"),
    ],
)
async def test_detected_pii_blocks_as_content_when_masking_is_disabled(
    input_type: Literal["request", "response"],
    texts: list[str],
    spans: list[dict[str, object]],
    expected_types: str,
    respx_mock: respx.MockRouter,
) -> None:
    _serve(respx_mock, _mls_body(pii_spans=spans))

    with pytest.raises(GuardrailRaisedException) as exc:
        await _apply(_guardrail(pii_mask=False, block_on_error=True), {"texts": texts}, input_type=input_type)

    assert exc.value.blocked_content is True
    assert f"PII detected in the {input_type} ({expected_types})" in exc.value.message


@pytest.mark.asyncio
async def test_request_carries_the_conversation_and_settings_but_not_the_model(respx_mock: respx.MockRouter) -> None:
    conversation: list[AllMessageValues] = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    route = _serve(respx_mock, _mls_body())

    await _apply(_guardrail(), {"texts": ["hello"], "structured_messages": conversation, "model": "openai/gpt-4o-mini"})

    assert route.calls.last.request.headers["Authorization"] == f"Bearer {_API_KEY}"
    assert _sent_body(route) == {
        "messages": conversation,
        "probes": ["hazard_prompt"],
        "pii": True,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
async def test_plain_texts_are_sent_as_user_turns(respx_mock: respx.MockRouter) -> None:
    route = _serve(respx_mock, _mls_body())

    await _apply(_guardrail(), {"texts": ["a", "b"]})

    assert _sent_body(route) == {
        "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        "probes": ["hazard_prompt"],
        "pii": True,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_empty_inputs_are_returned_without_calling_mls(
    input_type: Literal["request", "response"], respx_mock: respx.MockRouter
) -> None:
    route = _serve(respx_mock, _mls_body())
    inputs: GenericGuardrailAPIInputs = {"texts": []}

    assert await _apply(_guardrail(), inputs, input_type=input_type) is inputs
    assert route.call_count == 0, "empty input must not be billed as an MLS call"


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
@pytest.mark.parametrize("block_on_error", [None, True], ids=["default-fail-open", "fail-closed"])
@pytest.mark.parametrize(
    ("mls_reply", "reason"),
    [
        pytest.param(httpx.ConnectError("connection refused"), "connection refused", id="connection-error"),
        pytest.param(
            httpx.Response(HTTPStatus.SERVICE_UNAVAILABLE),
            f"{HTTPStatus.SERVICE_UNAVAILABLE.value} {HTTPStatus.SERVICE_UNAVAILABLE.phrase}",
            id="server-error",
        ),
        pytest.param(
            httpx.Response(HTTPStatus.OK, json={"results": "not a list"}),
            "Invalid RealmLabs guardrail response",
            id="invalid-body",
        ),
    ],
)
async def test_mls_failures_follow_the_error_policy(
    mls_reply: httpx.Response | httpx.HTTPError,
    reason: str,
    block_on_error: bool | None,
    input_type: Literal["request", "response"],
    respx_mock: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    route: Final = respx_mock.post(_URL)
    if isinstance(mls_reply, httpx.HTTPError):
        route.mock(side_effect=mls_reply)
    else:
        route.mock(return_value=mls_reply)
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["hello"]}
    guardrail: Final = _guardrail(block_on_error=block_on_error)

    if block_on_error:
        with pytest.raises(GuardrailRaisedException) as exc:
            await _apply(guardrail, inputs, input_type=input_type)
        assert exc.value.blocked_content is False
        assert reason in exc.value.message
    else:
        assert await _apply(guardrail, inputs, input_type=input_type) is inputs
    assert reason in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
@pytest.mark.parametrize("block_on_error", [False, True], ids=["fail-open", "fail-closed"])
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="empty-object"),
        pytest.param({"choices": [{"message": {"content": "hello"}}]}, id="chat-envelope"),
        pytest.param({"pii_spans": []}, id="missing-results"),
        pytest.param({"results": []}, id="missing-pii-spans"),
        pytest.param(
            {"results": [{"probe": "hazard_prompt", "probability": 0.99}], "pii_spans": []}, id="renamed-score"
        ),
        pytest.param({"results": [{"prob": 0.99}], "pii_spans": []}, id="missing-probe"),
        pytest.param(
            {"results": [{"probe": "hazard_prompt", "prob": 0.99, "role_mismatch": "true"}], "pii_spans": []},
            id="invalid-role-mismatch",
        ),
        pytest.param(_mls_body(pii_spans=[{"text": "Alex"}]), id="missing-pii-type"),
        pytest.param(_mls_body(pii_spans=[{"type": "name"}]), id="missing-pii-text"),
        pytest.param(_mls_body(pii_spans=[{"type": "name", "text": None}]), id="null-pii-text"),
        pytest.param(_mls_body(pii_spans=[{"type": "name", "text": ""}]), id="empty-pii-text"),
    ],
)
async def test_incomplete_verdicts_follow_the_error_policy(
    body: dict[str, object],
    block_on_error: bool,
    input_type: Literal["request", "response"],
    respx_mock: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _serve(respx_mock, body)
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["Hello Alex."]}
    guardrail: Final = _guardrail(block_on_error=block_on_error)

    if block_on_error:
        with pytest.raises(GuardrailRaisedException) as exc:
            await _apply(guardrail, inputs, input_type=input_type)
        assert exc.value.blocked_content is False, exc.value.message
        assert "Invalid RealmLabs guardrail response" in exc.value.message, exc.value.message
    else:
        assert await _apply(guardrail, inputs, input_type=input_type) is inputs
    assert "Invalid RealmLabs guardrail response" in caplog.text, caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prob",
    [
        pytest.param(None, id="null"),
        pytest.param("0.99", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(-0.1, id="negative"),
        pytest.param(1.1, id="above-one"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
async def test_invalid_probabilities_cannot_bypass_fail_closed(prob: object, respx_mock: respx.MockRouter) -> None:
    body: Final = {"results": [{"probe": "hazard_prompt", "prob": prob}], "pii_spans": []}
    respx_mock.post(_URL).mock(return_value=httpx.Response(HTTPStatus.OK, content=json.dumps(body)))

    with pytest.raises(GuardrailRaisedException) as exc:
        await _apply(_guardrail(block_on_error=True), {"texts": ["hello"]})

    assert exc.value.blocked_content is False, exc.value.message
    assert "Invalid RealmLabs guardrail response" in exc.value.message, exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_empty_verdict_arrays_are_valid_in_fail_closed_mode(
    input_type: Literal["request", "response"], respx_mock: respx.MockRouter
) -> None:
    _serve(respx_mock, {"results": [], "pii_spans": []})
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["hello"]}

    assert await _apply(_guardrail(block_on_error=True), inputs, input_type=input_type) is inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("hazard", [False, True], ids=["mask-pii", "block-hazard"])
async def test_additional_response_fields_preserve_policy_enforcement(
    hazard: bool, respx_mock: respx.MockRouter
) -> None:
    _serve(
        respx_mock,
        {
            "future_metadata": {"version": 2},
            "results": [{"probe": "hazard_prompt", "prob": 1.0 if hazard else 0.0, "future_field": [1, 2]}],
            "pii_spans": [{"type": "name", "text": "Alex", "future_field": {"source": "test"}}],
        },
    )
    guardrail: Final = _guardrail(hazard_threshold=0.5, block_on_error=True)
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["Hello Alex."]}

    if hazard:
        with pytest.raises(GuardrailRaisedException) as exc:
            await _apply(guardrail, inputs)
        assert exc.value.blocked_content is True, exc.value.message
        assert "hazard_prompt" in exc.value.message, exc.value.message
    else:
        assert await _apply(guardrail, inputs) == {"texts": ["Hello [name]."]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_key", "api_base", "expected_key", "expected_base"),
    [
        pytest.param(None, None, "env_key", "https://env.example.test", id="environment-fallback"),
        pytest.param(_API_KEY, f"{_API_BASE}/", _API_KEY, _API_BASE, id="config-overrides-environment"),
    ],
)
async def test_credentials_resolve_config_before_environment(
    api_key: str | None,
    api_base: str | None,
    expected_key: str,
    expected_base: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("REALMLABS_API_KEY", "env_key")
    monkeypatch.setenv("REALMLABS_API_BASE", "https://env.example.test/")
    route: Final = _serve(respx_mock, _mls_body(), f"{expected_base}/guardrail")

    await _apply(RealmLabsGuardrail(api_key=api_key, api_base=api_base), {"texts": ["hello"]})

    assert route.call_count == 1
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {expected_key}"


def test_missing_api_key_is_rejected_at_startup() -> None:
    with pytest.raises(RealmLabsMissingCredentials):
        RealmLabsGuardrail()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "expected_probes", "expected_thinking"),
    [
        pytest.param(
            {"probes": ["hazard_prompt", "dispute"], "pii": False, "enable_thinking": True},
            ["hazard_prompt", "dispute"],
            True,
            id="top-level",
        ),
        pytest.param(
            {"optional_params": {"probes": "all", "pii": False, "enable_thinking": True}},
            "all",
            True,
            id="nested",
        ),
        pytest.param(
            {
                "probes": ["hazard_prompt"],
                "pii": True,
                "enable_thinking": True,
                "optional_params": {"probes": [], "pii": False, "enable_thinking": False},
            },
            [],
            False,
            id="nested-false-and-empty-list-win",
        ),
        pytest.param(
            {
                "probes": "all",
                "pii": False,
                "enable_thinking": True,
                "optional_params": {"probes": None, "pii": None, "enable_thinking": None},
            },
            "all",
            True,
            id="nested-null-falls-back",
        ),
        pytest.param(
            {"probes": "all", "pii": False, "enable_thinking": True, "optional_params": {}},
            "all",
            True,
            id="empty-options-keep-top-level",
        ),
    ],
)
async def test_config_yaml_settings_reach_mls(
    settings: dict[str, object],
    expected_probes: list[str] | str,
    expected_thinking: bool,
    respx_mock: respx.MockRouter,
) -> None:
    guardrail: Final = _configured_guardrail(settings)
    route: Final = _serve(respx_mock, _mls_body())

    await _apply(guardrail, {"texts": ["hello"]})

    assert route.calls.last.request.headers["Authorization"] == f"Bearer {_API_KEY}"
    assert _sent_body(route) == {
        "messages": [{"role": "user", "content": "hello"}],
        "probes": expected_probes,
        "pii": False,
        "enable_thinking": expected_thinking,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"hazard_threshold": 0.0}, id="top-level-zero"),
        pytest.param({"optional_params": {"hazard_threshold": 0.0}}, id="nested-zero"),
        pytest.param({"hazard_threshold": 0.99, "optional_params": {"hazard_threshold": 0.0}}, id="nested-zero-wins"),
    ],
)
async def test_configured_zero_hazard_threshold_blocks_a_positive_score(
    settings: dict[str, object], respx_mock: respx.MockRouter
) -> None:
    guardrail: Final = _configured_guardrail(settings)
    _serve(respx_mock, _mls_body(hazard=0.01))

    with pytest.raises(GuardrailRaisedException) as exc:
        await _apply(guardrail, {"texts": ["hello"]})

    assert exc.value.blocked_content is True
    assert "threshold=0.0" in exc.value.message, exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"hazard_threshold": 0.99, "optional_params": {}}, id="omitted-nested"),
        pytest.param({"hazard_threshold": 0.99, "optional_params": {"hazard_threshold": None}}, id="null-nested"),
        pytest.param({"hazard_threshold": 0.0, "optional_params": {"hazard_threshold": 0.99}}, id="nested-wins"),
    ],
)
async def test_configured_higher_hazard_threshold_allows_the_request(
    settings: dict[str, object], respx_mock: respx.MockRouter
) -> None:
    guardrail: Final = _configured_guardrail(settings)
    _serve(respx_mock, _mls_body(hazard=0.9))
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["hello"]}

    assert await _apply(guardrail, inputs) is inputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"pii_mask": False}, id="top-level"),
        pytest.param({"optional_params": {"pii_mask": False}}, id="nested"),
        pytest.param({"pii_mask": True, "optional_params": {"pii_mask": False}}, id="nested-false-wins"),
        pytest.param({"pii_mask": False, "optional_params": {"pii_mask": None}}, id="null-nested"),
    ],
)
async def test_configured_masking_disabled_blocks_detected_pii(
    settings: dict[str, object], respx_mock: respx.MockRouter
) -> None:
    guardrail: Final = _configured_guardrail(settings)
    _serve(respx_mock, _mls_body(pii_spans=[_NAME_SPAN]))

    with pytest.raises(GuardrailRaisedException) as exc:
        await _apply(guardrail, {"texts": ["Hello Alex."]})

    assert exc.value.blocked_content is True
    assert "PII detected in the request (name)" in exc.value.message, exc.value.message


@pytest.mark.asyncio
async def test_nested_masking_enabled_overrides_top_level_blocking(respx_mock: respx.MockRouter) -> None:
    guardrail: Final = _configured_guardrail({"pii_mask": False, "optional_params": {"pii_mask": True}})
    _serve(respx_mock, _mls_body(pii_spans=[_NAME_SPAN]))

    result: Final = await _apply(guardrail, {"texts": ["Hello Alex."]})

    assert result == {"texts": ["Hello [name]."]}, result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"block_on_error": True}, id="top-level"),
        pytest.param({"optional_params": {"block_on_error": True}}, id="nested"),
        pytest.param({"block_on_error": False, "optional_params": {"block_on_error": True}}, id="nested-wins"),
        pytest.param({"block_on_error": True, "optional_params": {"block_on_error": None}}, id="null-nested"),
    ],
)
async def test_configured_fail_closed_blocks_an_mls_outage(
    settings: dict[str, object], respx_mock: respx.MockRouter
) -> None:
    guardrail: Final = _configured_guardrail(settings)
    respx_mock.post(_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    with pytest.raises(GuardrailRaisedException) as exc:
        await _apply(guardrail, {"texts": ["hello"]})

    assert exc.value.blocked_content is False
    assert "block_on_error=True" in exc.value.message, exc.value.message


@pytest.mark.asyncio
async def test_nested_fail_open_overrides_top_level_fail_closed(respx_mock: respx.MockRouter) -> None:
    guardrail: Final = _configured_guardrail({"block_on_error": True, "optional_params": {"block_on_error": False}})
    respx_mock.post(_URL).mock(side_effect=httpx.ConnectError("connection refused"))
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["hello"]}

    assert await _apply(guardrail, inputs) is inputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "expected_timeout"),
    [
        pytest.param({"optional_params": None}, None, id="default-no-options"),
        pytest.param({"optional_params": {}}, None, id="default-empty-options"),
        pytest.param({"optional_params": {"enable_thinking": True}}, None, id="default-thinking-enabled"),
        pytest.param({"optional_params": {"timeout": None}}, None, id="default-null-timeout"),
        pytest.param({"timeout": 2.0, "optional_params": None}, 2.0, id="top-level-no-options"),
        pytest.param({"timeout": 2.0, "optional_params": {}}, 2.0, id="top-level-empty-options"),
        pytest.param(
            {"timeout": 2.0, "optional_params": {"enable_thinking": True}}, 2.0, id="top-level-thinking-enabled"
        ),
        pytest.param({"timeout": 2.0, "optional_params": {"timeout": None}}, 2.0, id="top-level-null-timeout"),
        pytest.param({"timeout": 0.0, "optional_params": None}, 0.0, id="zero-no-options"),
        pytest.param({"timeout": 0.0, "optional_params": {}}, 0.0, id="zero-empty-options"),
        pytest.param({"timeout": 0.0, "optional_params": {"enable_thinking": True}}, 0.0, id="zero-thinking-enabled"),
        pytest.param({"timeout": 0.0, "optional_params": {"timeout": None}}, 0.0, id="zero-null-timeout"),
        pytest.param({"timeout": 2.0, "optional_params": {"timeout": 3.0}}, 3.0, id="nested-overrides-top-level"),
        pytest.param({"timeout": 2.0, "optional_params": {"timeout": 10.0}}, 10.0, id="explicit-nested-ten-seconds"),
    ],
)
async def test_configured_timeout_reaches_mls(
    settings: dict[str, object], expected_timeout: float | None, respx_mock: respx.MockRouter
) -> None:
    guardrail: Final = _configured_guardrail(settings)
    route: Final = _serve(respx_mock, _mls_body())

    await _apply(guardrail, {"texts": ["hello"]})

    expected: Final = RealmLabsGuardrailOptionalParams().timeout if expected_timeout is None else expected_timeout
    extensions: Final = _JSON_OBJECT.validate_python(route.calls.last.request.extensions)
    timeouts: Final = _JSON_OBJECT.validate_python(extensions["timeout"])
    assert timeouts["read"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_structured_messages", [False, True], ids=["request-fallback", "structured-takes-priority"]
)
async def test_request_uses_structured_messages_before_conversation_fallback(
    use_structured_messages: bool, respx_mock: respx.MockRouter
) -> None:
    conversation: Final[list[AllMessageValues]] = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello."},
    ]
    inputs: Final[GenericGuardrailAPIInputs] = (
        {"texts": ["Hello."], "structured_messages": conversation} if use_structured_messages else {"texts": ["Hello."]}
    )
    request_data: Final = {
        "messages": [{"role": "user", "content": "Outside the selected scope."}]
        if use_structured_messages
        else conversation
    }
    route: Final = _serve(respx_mock, _mls_body())

    result: Final = await _apply(_guardrail(), inputs, request_data=request_data)

    assert result is inputs
    assert _sent_body(route) == {
        "messages": conversation,
        "probes": ["hazard_prompt"],
        "pii": True,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
async def test_role_mismatch_skips_request_hazard_but_still_masks_pii(respx_mock: respx.MockRouter) -> None:
    _serve(
        respx_mock,
        {
            "results": [{"probe": "hazard_prompt", "prob": 0.99, "role_mismatch": True}],
            "pii_spans": [_NAME_SPAN],
        },
    )

    result: Final = await _apply(_guardrail(), {"texts": ["Hello Alex."]})

    assert result == {"texts": ["Hello [name]."]}, result


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch_fields", [{}, {"role_mismatch": None}], ids=["omitted", "null"])
async def test_missing_role_mismatch_still_enforces_request_hazard(
    mismatch_fields: dict[str, object], respx_mock: respx.MockRouter
) -> None:
    _serve(respx_mock, {"results": [{"probe": "hazard_prompt", "prob": 0.99, **mismatch_fields}], "pii_spans": []})

    with pytest.raises(GuardrailRaisedException) as exc:
        await _apply(_guardrail(), {"texts": ["a hazardous request"]})

    assert exc.value.blocked_content is True
    assert "hazard_prompt" in exc.value.message, exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Alex", "[name]"], ids=["unmasked-name", "existing-placeholder"])
async def test_post_call_masks_the_reply_in_conversation_context(name: str, respx_mock: respx.MockRouter) -> None:
    guardrail: Final = _guardrail(event_hook=GuardrailEventHooks.post_call)
    conversation: Final[list[AllMessageValues]] = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "My name is Alex."},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is my name?"},
    ]
    route: Final = _serve(respx_mock, _mls_body(pii_spans=[{"type": "name", "text": name.removesuffix("]")}]))
    response: Final = ModelResponse(
        choices=[
            Choices(index=0, message=Message(content=f"Your name is {name}.", role="assistant"), finish_reason="stop")
        ]
    )

    result: Final = await OpenAIChatCompletionsHandler().process_output_response(  # pyright: ignore[reportUnknownMemberType]  # upstream request_data parameter uses an unparameterized dict
        response=response, guardrail_to_apply=guardrail, request_data={"messages": conversation}
    )

    assert result.choices == [
        Choices(index=0, message=Message(content="Your name is [name].", role="assistant"), finish_reason="stop")
    ], result.choices
    assert conversation == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "My name is Alex."},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is my name?"},
    ], conversation
    assert _sent_body(route) == {
        "messages": [*conversation, {"role": "assistant", "content": f"Your name is {name}."}],
        "probes": ["hazard_prompt"],
        "pii": True,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_conversation_text_parts_use_blank_lines_without_changing_internal_paragraphs(
    input_type: Literal["request", "response"], respx_mock: respx.MockRouter
) -> None:
    conversation: Final = [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this.\n\nKeep this paragraph."},
                {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                {"type": "text", "text": "In one sentence."},
            ],
        },
        {"role": "assistant", "content": None},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/other.png"}}]},
        {"content": "No role."},
        None,
    ]
    route: Final = _serve(respx_mock, _mls_body())
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["A landscape."]}

    result: Final = await _apply(_guardrail(), inputs, input_type=input_type, request_data={"messages": conversation})

    expected_history: Final = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Describe this.\n\nKeep this paragraph.\n\nIn one sentence."},
    ]
    expected_messages: Final = (
        [*expected_history, {"role": "assistant", "content": "A landscape."}]
        if input_type == "response"
        else expected_history
    )

    assert result is inputs
    assert _sent_body(route) == {
        "messages": expected_messages,
        "probes": ["hazard_prompt"],
        "pii": True,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("request_data", [{}, {"messages": []}, {"messages": None}], ids=["missing", "empty", "null"])
async def test_response_without_history_is_still_scanned_as_assistant_text(
    request_data: dict[str, object], respx_mock: respx.MockRouter
) -> None:
    route: Final = _serve(respx_mock, _mls_body(pii_spans=[_NAME_SPAN]))

    result: Final = await _apply(
        _guardrail(), {"texts": ["Alex was here."]}, input_type="response", request_data=request_data
    )

    assert result == {"texts": ["[name] was here."]}, result
    assert _sent_body(route) == {
        "messages": [{"role": "assistant", "content": "Alex was here."}],
        "probes": ["hazard_prompt"],
        "pii": True,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("role_mismatch", [False, True, None], ids=["matching-role", "mismatched-role", "null-role"])
async def test_response_ignores_hazard_scores_and_merges_overlapping_pii(
    role_mismatch: bool | None, respx_mock: respx.MockRouter
) -> None:
    _serve(
        respx_mock,
        {
            "results": [{"probe": "hazard_prompt", "prob": 0.99, "role_mismatch": role_mismatch}],
            "pii_spans": [{"type": "name", "text": "Ann"}, {"type": "email", "text": "Ann.Smith@example.com"}],
        },
    )
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["Contact Ann at Ann.Smith@example.com.", "No PII here."]}

    result: Final = await _apply(_guardrail(), inputs, input_type="response")

    assert result == {"texts": ["Contact [name] at [email].", "No PII here."]}, result
    assert inputs == {"texts": ["Contact Ann at Ann.Smith@example.com.", "No PII here."]}, inputs


@pytest.mark.asyncio
async def test_pii_only_in_history_leaves_the_reply_unchanged(respx_mock: respx.MockRouter) -> None:
    _serve(respx_mock, _mls_body(pii_spans=[_NAME_SPAN]))
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["Hello there."]}
    conversation: Final = [{"role": "user", "content": "My name is Alex."}]

    result: Final = await _apply(_guardrail(), inputs, input_type="response", request_data={"messages": conversation})

    assert result is inputs
    assert conversation == [{"role": "user", "content": "My name is Alex."}], conversation
