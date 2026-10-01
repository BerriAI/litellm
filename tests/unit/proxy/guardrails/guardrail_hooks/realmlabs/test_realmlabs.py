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
from litellm.proxy.guardrails.guardrail_hooks.realmlabs import guardrail_initializer_registry
from litellm.proxy.guardrails.guardrail_hooks.realmlabs.realmlabs import (
    RealmLabsGuardrail,
    RealmLabsMissingCredentials,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import RealmLabsGuardrailOptionalParams
from litellm.types.utils import GenericGuardrailAPIInputs

_API_KEY = "mls_gr_test"
_API_BASE = "https://mls.example.test"
_URL = f"{_API_BASE}/guardrail"
_DEFAULT_THRESHOLD = 0.703


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
    block_on_error: bool | None = None,
    event_hook: GuardrailEventHooks = GuardrailEventHooks.pre_call,
) -> RealmLabsGuardrail:
    """Guardrail pointed at the fake MLS URL; unset arguments keep the guardrail's defaults."""

    return RealmLabsGuardrail(
        api_key=_API_KEY,
        api_base=_API_BASE,
        hazard_threshold=hazard_threshold,
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
        "pii": False,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
async def test_plain_texts_are_sent_as_user_turns(respx_mock: respx.MockRouter) -> None:
    route = _serve(respx_mock, _mls_body())

    await _apply(_guardrail(), {"texts": ["a", "b"]})

    assert _sent_body(route) == {
        "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        "probes": ["hazard_prompt"],
        "pii": False,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request"])
async def test_empty_inputs_are_returned_without_calling_mls(
    input_type: Literal["request", "response"], respx_mock: respx.MockRouter
) -> None:
    route = _serve(respx_mock, _mls_body())
    inputs: GenericGuardrailAPIInputs = {"texts": []}

    assert await _apply(_guardrail(), inputs, input_type=input_type) is inputs
    assert route.call_count == 0, "empty input must not be billed as an MLS call"


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request"])
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
@pytest.mark.parametrize("input_type", ["request"])
@pytest.mark.parametrize("block_on_error", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="empty-object"),
        pytest.param({"choices": [{"message": {"content": "hello"}}]}, id="chat-envelope"),
        pytest.param({"pii_spans": []}, id="missing-results"),
        pytest.param(
            {"results": [{"probe": "hazard_prompt", "probability": 0.99}], "pii_spans": []}, id="renamed-score"
        ),
        pytest.param({"results": [{"prob": 0.99}], "pii_spans": []}, id="missing-probe"),
        pytest.param(
            {"results": [{"probe": "hazard_prompt", "prob": 0.99, "role_mismatch": "true"}], "pii_spans": []},
            id="invalid-role-mismatch",
        ),
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
@pytest.mark.parametrize("input_type", ["request"])
async def test_empty_verdict_arrays_are_valid_in_fail_closed_mode(
    input_type: Literal["request", "response"], respx_mock: respx.MockRouter
) -> None:
    _serve(respx_mock, {"results": [], "pii_spans": []})
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["hello"]}

    assert await _apply(_guardrail(block_on_error=True), inputs, input_type=input_type) is inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("hazard", [False, True])
async def test_additional_response_fields_preserve_policy_enforcement(
    hazard: bool, respx_mock: respx.MockRouter
) -> None:
    _serve(
        respx_mock,
        {
            "future_metadata": {"version": 2},
            "results": [{"probe": "hazard_prompt", "prob": 1.0 if hazard else 0.0, "future_field": [1, 2]}],
            "pii_spans": [],
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
        assert await _apply(guardrail, inputs) is inputs


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
            {"probes": ["hazard_prompt", "dispute"], "enable_thinking": True},
            ["hazard_prompt", "dispute"],
            True,
            id="top-level",
        ),
        pytest.param(
            {"optional_params": {"probes": "all", "enable_thinking": True}},
            "all",
            True,
            id="nested",
        ),
        pytest.param(
            {
                "probes": ["hazard_prompt"],
                "enable_thinking": True,
                "optional_params": {"probes": [], "enable_thinking": False},
            },
            [],
            False,
            id="nested-false-and-empty-list-win",
        ),
        pytest.param(
            {
                "probes": "all",
                "enable_thinking": True,
                "optional_params": {"probes": None, "enable_thinking": None},
            },
            "all",
            True,
            id="nested-null-falls-back",
        ),
        pytest.param(
            {"probes": "all", "enable_thinking": True, "optional_params": {}},
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
async def test_response_text_is_returned_without_calling_mls(respx_mock: respx.MockRouter) -> None:
    route: Final = _serve(respx_mock, _mls_body())
    inputs: Final[GenericGuardrailAPIInputs] = {"texts": ["hello"]}

    assert await _apply(_guardrail(), inputs, input_type="response") is inputs
    assert route.call_count == 0
