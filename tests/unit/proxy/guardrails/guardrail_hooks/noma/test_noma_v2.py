import json

import httpx
import pytest
import respx

import litellm
from litellm.proxy.guardrails.guardrail_hooks.noma import (
    NomaV2Guardrail,
    guardrail_initializer_registry,
)
from litellm.types.guardrails import LitellmParams

_API_BASE = "https://noma.example.test"


@pytest.fixture(autouse=True)
def _fresh_httpx_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", None)
    monkeypatch.delenv("NOMA_GATEWAY_NAME", raising=False)


def _guardrail(gateway_name: str | None) -> NomaV2Guardrail:
    return NomaV2Guardrail(
        api_base=_API_BASE,
        gateway_name=gateway_name,
        guardrail_name="noma-guard",
        event_hook="pre_call",
        default_on=True,
    )


async def _scan_body(guardrail: NomaV2Guardrail, respx_mock: respx.MockRouter) -> dict[str, object]:
    route = respx_mock.post(f"{_API_BASE}/litellm/guardrail").respond(json={"action": "NONE"})
    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={"metadata": {}}, input_type="request")
    assert route.call_count == 1
    return json.loads(route.calls.last.request.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(("guardrail_type", "extra_params"), [("noma_v2", {}), ("noma", {"use_v2": True})])
async def test_gateway_name_from_guardrail_config_reaches_noma(
    guardrail_type: str, extra_params: dict[str, bool], respx_mock: respx.MockRouter
) -> None:
    litellm_params = LitellmParams(
        guardrail=guardrail_type,
        mode="pre_call",
        api_base=_API_BASE,
        gateway_name="prod-us-east",
        **extra_params,
    )
    guardrail = guardrail_initializer_registry[guardrail_type](litellm_params, {"guardrail_name": "noma-guard"})

    assert (await _scan_body(guardrail, respx_mock))["gateway_name"] == "prod-us-east"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "env_value", "expected"),
    [
        (None, "env-gateway", "env-gateway"),
        ("config-gateway", "env-gateway", "config-gateway"),
        ("  config-gateway  ", None, "config-gateway"),
    ],
)
async def test_gateway_name_resolution(
    configured: str | None,
    env_value: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    if env_value is not None:
        monkeypatch.setenv("NOMA_GATEWAY_NAME", env_value)

    assert (await _scan_body(_guardrail(configured), respx_mock))["gateway_name"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, "", "   "])
async def test_unset_or_blank_gateway_name_is_left_out(configured: str | None, respx_mock: respx.MockRouter) -> None:
    assert "gateway_name" not in await _scan_body(_guardrail(configured), respx_mock)


@pytest.mark.asyncio
async def test_positional_args_keep_their_meaning_after_gateway_name_was_added(respx_mock: respx.MockRouter) -> None:
    guardrail = NomaV2Guardrail("test-api-key", _API_BASE, "test-app", False, True)

    body = await _scan_body(guardrail, respx_mock)

    assert body["monitor_mode"] is False
    assert body["application_id"] == "test-app"
    assert "gateway_name" not in body
    respx_mock.post(f"{_API_BASE}/litellm/guardrail").respond(status_code=503)
    with pytest.raises(httpx.HTTPStatusError):
        await guardrail.apply_guardrail(
            inputs={"texts": ["hello"]}, request_data={"metadata": {}}, input_type="request"
        )
