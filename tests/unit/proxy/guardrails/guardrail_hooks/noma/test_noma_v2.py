import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

import httpx
import pytest
import respx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_hooks.noma import (
    NomaV2Guardrail,
    guardrail_initializer_registry,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import UnifiedLLMGuardrails
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import LitellmParams

_API_BASE = "https://noma.example.test"
_STREAMING_KEYS = (
    "streaming_buffer_until_moderated",
    "streaming_end_of_stream_only",
    "streaming_sampling_rate",
    "streaming_buffer_release_on_scan",
)


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


_Loader: TypeAlias = Callable[[Path, Mapping[str, object]], CustomGuardrail | None]
_SUBCLASS_MODULE = (
    "from litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2 import NomaV2Guardrail\n\n\n"
    "class SubclassedNoma(NomaV2Guardrail):\n"
    "    pass\n"
)


def _initializer(guardrail_type: str, fixed: Mapping[str, object]) -> _Loader:
    def load(_tmp_path: Path, params: Mapping[str, object]) -> CustomGuardrail | None:
        litellm_params = LitellmParams(
            guardrail=guardrail_type, mode="post_call", api_base=_API_BASE, **fixed, **params
        )
        return guardrail_initializer_registry[guardrail_type](litellm_params, {"guardrail_name": "noma-guard"})

    return load


def _custom_subclass(tmp_path: Path, params: Mapping[str, object]) -> CustomGuardrail | None:
    (tmp_path / "noma_subclass.py").write_text(_SUBCLASS_MODULE)
    guardrail_type = "noma_subclass.SubclassedNoma"
    return InMemoryGuardrailHandler().initialize_custom_guardrail(
        guardrail={"guardrail_name": "noma-guard"},
        guardrail_type=guardrail_type,
        litellm_params=LitellmParams(guardrail=guardrail_type, mode="post_call", api_base=_API_BASE, **params),
        config_file_path=str(tmp_path / "config.yaml"),
    )


_LOADERS = pytest.mark.parametrize(
    "load",
    [_initializer("noma_v2", {}), _initializer("noma", {"use_v2": True}), _custom_subclass],
    ids=["noma_v2", "noma-use_v2", "custom-subclass"],
)


def _streaming_flags(load: _Loader, tmp_path: Path, params: Mapping[str, object]) -> list[object]:
    guardrail = load(tmp_path, params)
    return [UnifiedLLMGuardrails().resolve_streaming_flag(guardrail, key, "hook-default") for key in _STREAMING_KEYS]


@_LOADERS
@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (
            {
                "streaming_buffer_until_moderated": True,
                "streaming_end_of_stream_only": True,
                "streaming_sampling_rate": 3,
                "streaming_buffer_release_on_scan": True,
            },
            [True, True, 3, True],
        ),
        (
            {
                "streaming_buffer_until_moderated": False,
                "streaming_end_of_stream_only": False,
                "streaming_sampling_rate": 1,
                "streaming_buffer_release_on_scan": False,
            },
            [False, False, 1, False],
        ),
        ({"streaming_sampling_rate": 7}, ["hook-default", "hook-default", 7, "hook-default"]),
        (
            {"streaming_buffer_until_moderated": "false", "streaming_sampling_rate": "2"},
            [False, "hook-default", 2, "hook-default"],
        ),
        ({}, ["hook-default"] * 4),
    ],
    ids=["all-on", "all-off", "one-key", "string-values", "unset"],
)
def test_streaming_settings_in_guardrail_config_reach_the_streaming_hook(
    load: _Loader, tmp_path: Path, configured: dict[str, object], expected: list[object]
) -> None:
    assert _streaming_flags(load, tmp_path, configured) == expected


@_LOADERS
@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({"streaming_sampling_rate": 0}, ["hook-default"] * 4),
        ({"streaming_sampling_rate": -1}, ["hook-default"] * 4),
        ({"streaming_sampling_rate": "often"}, ["hook-default"] * 4),
        ({"streaming_buffer_until_moderated": "sometimes"}, ["hook-default"] * 4),
        (
            {"streaming_buffer_until_moderated": True, "streaming_sampling_rate": 0},
            [True, "hook-default", "hook-default", "hook-default"],
        ),
    ],
    ids=["rate-zero", "rate-negative", "rate-word", "buffer-word", "valid-key-kept"],
)
def test_invalid_streaming_value_is_ignored_and_the_guardrail_still_loads(
    load: _Loader,
    tmp_path: Path,
    configured: dict[str, object],
    expected: list[object],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verbose_proxy_logger, "propagate", True)
    assert _streaming_flags(load, tmp_path, configured) == expected
    warnings = [record.getMessage() for record in caplog.records if "not a valid value" in record.getMessage()]
    assert warnings, caplog.text
    assert all(message.startswith("Noma v2 guardrail noma-guard: ignoring ") for message in warnings), warnings
