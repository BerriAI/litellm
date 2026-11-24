from collections.abc import Mapping
from typing import Final

import pytest

from litellm.router_config import build_router_config
from litellm.router_rust.core import RustCore
from tests.unit.test_router.core.fallbacks_helpers import complete, headers
from tests.unit.test_router.core.scripted_host import ScriptedHost, rust_core

_HEADER: Final = "x-litellm-ignored-cache-controls"
_OFF: Final = {"no-cache": True, "no-store": True}


def _deployment(name: str, **cache_params: object) -> Mapping[str, object]:  # kwargs-ok: litellm_params
    return {
        "model_name": name,
        "litellm_params": {"model": f"openai/{name}", **cache_params},
        "model_info": {"id": name},
    }


@pytest.mark.parametrize(
    ("cache_params", "request_cache", "sent_cache", "ignored"),
    [
        pytest.param(
            {"cache": _OFF, "allowed_cache_controls": []},
            {"ttl": 600, "no-cache": False, "no-store": False},
            _OFF,
            "ttl,no-cache,no-store",
            id="a locked off model stays off and names every ignored control",
        ),
        pytest.param(
            {"cache": {"ttl": 60}, "allowed_cache_controls": ["no-cache", "no-store"]},
            {"ttl": 5, "no-cache": True},
            {"ttl": 60, "no-cache": True},
            "ttl",
            id="a locked ttl stays and the allowed no-cache goes through",
        ),
        pytest.param(
            {"cache": {"ttl": 60}, "allowed_cache_controls": ["ttl"]},
            {"ttl": 5},
            {"ttl": 5},
            None,
            id="an allowed ttl overrides the model's",
        ),
        pytest.param(
            {"cache": _OFF},
            {"ttl": 600},
            {"no-cache": True, "no-store": True, "ttl": 600},
            None,
            id="without a list a request ttl leaves the model's no-store in place",
        ),
        pytest.param({}, {"ttl": 5}, {"ttl": 5}, None, id="a model with no rule sends the request's controls"),
    ],
)
@pytest.mark.asyncio
async def test_attempt_sends_the_merged_cache_controls_and_reports_ignored_ones(
    cache_params: dict[str, object],
    request_cache: dict[str, object],
    sent_cache: dict[str, object],
    ignored: str | None,
) -> None:
    host: Final = ScriptedHost()
    core: Final = rust_core(host, model_list=[_deployment("a", **cache_params)], num_retries=0)

    response: Final = await complete(core, cache=request_cache)

    (call,) = host.calls_to("acompletion")
    assert call.kwargs["cache"] == sent_cache
    assert "allowed_cache_controls" not in call.kwargs
    assert headers(response).get(_HEADER) == ignored


@pytest.mark.asyncio
async def test_fallback_response_reports_only_the_controls_its_own_deployment_ignored() -> None:
    host: Final = ScriptedHost()
    core: Final = rust_core(
        host,
        model_list=[_deployment("a", cache={"ttl": 60}, allowed_cache_controls=[]), _deployment("b")],
        num_retries=0,
        fallbacks=[{"a": ["b"]}],
    )
    host.answer("openai/a", RuntimeError("down"))

    response: Final = await complete(core, cache={"ttl": 5})

    primary, fallback = host.calls_to("acompletion")
    assert primary.kwargs["cache"] == {"ttl": 60}
    assert fallback.kwargs["cache"] == {"ttl": 5}
    assert _HEADER not in headers(response)


@pytest.mark.parametrize("request_cache", [None, {"ttl": 600, "no-store": False}])
@pytest.mark.asyncio
async def test_a_lock_without_a_model_rule_keeps_the_router_default_cache_controls(
    request_cache: dict[str, object] | None,
) -> None:
    host: Final = ScriptedHost()
    core: Final = rust_core(
        host,
        model_list=[_deployment("a", allowed_cache_controls=[])],
        num_retries=0,
        default_litellm_params={"cache": _OFF},
    )

    await (complete(core) if request_cache is None else complete(core, cache=request_cache))

    (call,) = host.calls_to("acompletion")
    assert call.kwargs["cache"] == _OFF


@pytest.mark.parametrize(
    ("cache_params", "request_cache", "sent_cache"),
    [
        pytest.param(
            {"cache": _OFF, "allowed_cache_controls": []},
            {"ttl": 600, "no-cache": False, "no-store": False},
            _OFF,
            id="a locked off model stays off",
        ),
        pytest.param(
            {"cache": _OFF}, {"ttl": 600}, {**_OFF, "ttl": 600}, id="a request ttl leaves the model's opt out"
        ),
    ],
)
def test_sync_text_completion_sends_the_merged_cache_controls(
    cache_params: Mapping[str, object], request_cache: Mapping[str, object], sent_cache: Mapping[str, object]
) -> None:
    host: Final = ScriptedHost()
    _ = RustCore.build(
        build_router_config({"model_list": [_deployment("a", **cache_params)], "num_retries": 0}), host
    ).text_completion(model="a", prompt="hi", cache=request_cache)

    (call,) = host.calls_to("text_completion")
    assert call.kwargs["cache"] == sent_cache
    assert "allowed_cache_controls" not in call.kwargs
