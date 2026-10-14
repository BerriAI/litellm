from typing import Final

import httpx
import pytest
import respx

import litellm
from tests.unit.llms.sail.helpers import MODEL, SAIL_API_BASE, SpendCapture, cost_at, responses_body, sent_body

INPUT: Final = "hi"


@pytest.fixture
def responses_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{SAIL_API_BASE}/responses").mock(return_value=httpx.Response(200, json=responses_body()))


@pytest.mark.parametrize(
    ("service_tier", "metadata", "background", "wire_metadata", "column_suffix"),
    [
        pytest.param(None, None, None, None, "", id="no-tier"),
        pytest.param("auto", None, None, None, "", id="auto"),
        pytest.param("default", None, None, {"completion_window": "asap"}, "", id="default"),
        pytest.param("priority", None, None, {"completion_window": "asap"}, "", id="priority"),
        pytest.param("flex", None, True, {"completion_window": "flex"}, "_flex", id="flex-background"),
        pytest.param("balanced", None, None, {"completion_window": "balanced"}, "_balanced", id="balanced"),
        pytest.param("balanced", None, True, {"completion_window": "balanced"}, "_balanced", id="balanced-background"),
        pytest.param("Balanced", None, None, {"completion_window": "balanced"}, "_balanced", id="balanced-any-case"),
        pytest.param(
            "flex",
            {"user_tag": "a"},
            True,
            {"user_tag": "a", "completion_window": "flex"},
            "_flex",
            id="tier-keeps-metadata",
        ),
        pytest.param(
            None, {"completion_window": "flex"}, True, {"completion_window": "flex"}, "_flex", id="caller-window"
        ),
        pytest.param(
            None,
            {"completion_window": "standard"},
            None,
            {"completion_window": "standard"},
            "_balanced",
            id="standard-window",
        ),
        pytest.param(
            None, {"completion_window": "FLEX"}, True, {"completion_window": "flex"}, "_flex", id="window-any-case"
        ),
        pytest.param(
            "priority",
            {"completion_window": "asap"},
            False,
            {"completion_window": "asap"},
            "",
            id="agreeing-tier-and-window",
        ),
        pytest.param(None, {"user_tag": "a"}, None, {"user_tag": "a"}, "", id="metadata-without-window"),
    ],
)
@pytest.mark.asyncio
async def test_sail_responses_send_the_window_and_bill_its_price_columns(
    sail_env: None,
    responses_route: respx.Route,
    spend_capture: SpendCapture,
    service_tier: str | None,
    metadata: dict[str, str] | None,
    background: bool | None,
    wire_metadata: dict[str, str] | None,
    column_suffix: str,
) -> None:
    await litellm.aresponses(
        model=MODEL,
        input=INPUT,
        service_tier=service_tier,
        metadata=metadata,
        background=background,
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(responses_route)
    assert "service_tier" not in body
    assert body.get("metadata") == wire_metadata
    assert body.get("background") == background
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(column_suffix))


FLEX_NEEDS_BACKGROUND: Final = "only to Responses requests with background=true"
ASAP_CANNOT_RUN_IN_BACKGROUND: Final = "does not serve .* to Responses requests with background=true"


@pytest.mark.parametrize(
    ("service_tier", "metadata", "background", "message"),
    [
        pytest.param("scale", None, None, "service_tier='scale'", id="unknown-tier"),
        pytest.param(5, None, None, "service_tier=5", id="non-string-tier"),
        pytest.param(None, {"completion_window": "soon"}, None, "completion_window='soon'", id="unknown-window"),
        pytest.param("flex", {"completion_window": "asap"}, None, "select different completion windows", id="conflict"),
        pytest.param("flex", None, None, FLEX_NEEDS_BACKGROUND, id="flex-tier-in-foreground"),
        pytest.param("FLEX", None, False, FLEX_NEEDS_BACKGROUND, id="flex-tier-with-background-false"),
        pytest.param(None, {"completion_window": "flex"}, None, FLEX_NEEDS_BACKGROUND, id="flex-window-in-foreground"),
        pytest.param("priority", None, True, ASAP_CANNOT_RUN_IN_BACKGROUND, id="priority-tier-in-background"),
        pytest.param("default", None, True, ASAP_CANNOT_RUN_IN_BACKGROUND, id="default-tier-in-background"),
        pytest.param(
            None, {"completion_window": "asap"}, True, ASAP_CANNOT_RUN_IN_BACKGROUND, id="asap-window-in-background"
        ),
    ],
)
@pytest.mark.asyncio
async def test_sail_responses_reject_before_sending(
    sail_env: None,
    responses_route: respx.Route,
    service_tier: object,
    metadata: dict[str, str] | None,
    background: bool | None,
    message: str,
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match=message):
        await litellm.aresponses(
            model=MODEL, input=INPUT, service_tier=service_tier, metadata=metadata, background=background
        )

    assert not responses_route.called


@pytest.mark.parametrize(
    ("service_tier", "metadata", "background", "wire_metadata"),
    [
        pytest.param("flex", None, None, None, id="flex-tier-in-foreground"),
        pytest.param(None, {"completion_window": "flex", "user_tag": "a"}, None, {"user_tag": "a"}, id="flex-window"),
        pytest.param("priority", None, True, None, id="priority-tier-in-background"),
    ],
)
@pytest.mark.asyncio
async def test_sail_responses_drop_a_window_sail_cannot_serve_under_drop_params_and_bill_base(
    sail_env: None,
    responses_route: respx.Route,
    spend_capture: SpendCapture,
    service_tier: str | None,
    metadata: dict[str, str] | None,
    background: bool | None,
    wire_metadata: dict[str, str] | None,
) -> None:
    await litellm.aresponses(
        model=MODEL,
        input=INPUT,
        service_tier=service_tier,
        metadata=metadata,
        background=background,
        drop_params=True,
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(responses_route)
    assert "service_tier" not in body
    assert body.get("metadata") == wire_metadata
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(""))


@pytest.mark.asyncio
async def test_sail_responses_drop_an_unknown_tier_and_window_under_drop_params(
    sail_env: None, responses_route: respx.Route, spend_capture: SpendCapture
) -> None:
    await litellm.aresponses(
        model=MODEL,
        input=INPUT,
        service_tier="scale",
        metadata={"completion_window": "soon", "user_tag": "a"},
        drop_params=True,
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(responses_route)
    assert "service_tier" not in body
    assert body["metadata"] == {"user_tag": "a"}
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(""))


@pytest.mark.parametrize(
    ("service_tier", "wire_metadata", "column_suffix"),
    [
        pytest.param("balanced", {"trace_id": "t-1", "completion_window": "balanced"}, "_balanced", id="balanced"),
        pytest.param(None, {"trace_id": "t-1"}, "", id="no-tier"),
    ],
)
@pytest.mark.asyncio
async def test_sail_responses_merge_caller_extra_body_metadata_with_the_tier_window(
    sail_env: None,
    responses_route: respx.Route,
    spend_capture: SpendCapture,
    service_tier: str | None,
    wire_metadata: dict[str, str],
    column_suffix: str,
) -> None:
    await litellm.aresponses(
        model=MODEL,
        input=INPUT,
        service_tier=service_tier,
        extra_body={"metadata": {"trace_id": "t-1"}, "foo": 1},
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(responses_route)
    assert body["metadata"] == wire_metadata
    assert body["foo"] == 1
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(column_suffix))


@pytest.mark.parametrize(
    ("extra_body", "message"),
    [
        pytest.param(
            {"metadata": {"completion_window": "flex"}},
            "extra_body.metadata.completion_window",
            id="extra-body-window",
        ),
        pytest.param({"service_tier": "flex"}, "service_tier inside extra_body", id="extra-body-tier"),
    ],
)
@pytest.mark.asyncio
async def test_sail_responses_reject_a_window_billing_cannot_see_before_sending(
    sail_env: None, responses_route: respx.Route, extra_body: dict[str, object], message: str
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match=message):
        await litellm.aresponses(model=MODEL, input=INPUT, extra_body=extra_body)

    assert not responses_route.called


def test_sail_sync_responses_drop_a_window_billing_cannot_see_under_drop_params(
    sail_env: None, responses_route: respx.Route
) -> None:
    litellm.responses(
        model=MODEL,
        input=INPUT,
        service_tier="balanced",
        extra_body={"service_tier": "flex", "metadata": {"trace_id": "t-1", "completion_window": "flex"}},
        drop_params=True,
    )

    body: Final = sent_body(responses_route)
    assert "service_tier" not in body
    assert body["metadata"] == {"trace_id": "t-1", "completion_window": "balanced"}


@pytest.mark.asyncio
async def test_sail_responses_drop_a_lone_caller_window_under_drop_params_and_bill_asap(
    sail_env: None, responses_route: respx.Route, spend_capture: SpendCapture
) -> None:
    await litellm.aresponses(
        model=MODEL,
        input=INPUT,
        extra_body={"metadata": {"completion_window": "flex"}},
        drop_params=True,
        litellm_call_id=spend_capture.call_id,
    )

    assert "completion_window" not in (sent_body(responses_route).get("metadata") or {})
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(""))


def test_sail_responses_pass_a_non_mapping_extra_body_metadata_through_untouched(
    sail_env: None, responses_route: respx.Route
) -> None:
    litellm.responses(model=MODEL, input=INPUT, extra_body={"metadata": None, "foo": 1})

    body: Final = sent_body(responses_route)
    assert "metadata" in body
    assert body["metadata"] is None
    assert body["foo"] == 1


@pytest.mark.asyncio
async def test_sail_responses_use_sail_api_base_env_and_key(
    sail_env: None, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAIL_API_BASE", "https://sail-gateway.invalid/v1")
    route: Final = respx_mock.post("https://sail-gateway.invalid/v1/responses").mock(
        return_value=httpx.Response(200, json=responses_body())
    )

    await litellm.aresponses(model=MODEL, input=INPUT)

    assert route.calls.last.request.headers["Authorization"] == "Bearer sail-test-key"
