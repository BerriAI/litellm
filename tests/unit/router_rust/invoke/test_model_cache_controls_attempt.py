from collections.abc import Mapping
from typing import Final

import pytest

from litellm.router_config import build_router_config
from litellm.router_rust.core import RustCore
from tests.unit.test_router.core.fallbacks_helpers import complete, headers
from tests.unit.test_router.core.retries_health_helpers import fresh, not_found
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


@pytest.mark.parametrize(
    ("primary_params", "fallback_params", "primary_cache", "fallback_cache", "ignored"),
    [
        pytest.param(
            {"cache": {"ttl": 60}, "allowed_cache_controls": []},
            {},
            {"ttl": 60},
            {"ttl": 60},
            "ttl",
            id="a fallback with no rule follows the requested model's lock",
        ),
        pytest.param(
            {"cache": _OFF, "allowed_cache_controls": []},
            {"cache": {"ttl": 600}, "allowed_cache_controls": ["ttl"]},
            _OFF,
            _OFF,
            "ttl",
            id="the requested model's lock replaces the fallback's own rule and list",
        ),
        pytest.param(
            {"cache": {"ttl": 60}},
            {"cache": {"ttl": 600}, "allowed_cache_controls": []},
            {"ttl": 5},
            {"ttl": 600},
            "ttl",
            id="without a lock on the requested model the fallback keeps its own",
        ),
        pytest.param({"cache": {"ttl": 60}}, {}, {"ttl": 5}, {"ttl": 5}, None, id="no lock on either model"),
    ],
)
@pytest.mark.asyncio
async def test_a_fallback_follows_the_cache_lock_of_the_requested_model(
    primary_params: dict[str, object],
    fallback_params: dict[str, object],
    primary_cache: dict[str, object],
    fallback_cache: dict[str, object],
    ignored: str | None,
) -> None:
    host: Final = ScriptedHost()
    core: Final = rust_core(
        host,
        model_list=[_deployment("a", **primary_params), _deployment("b", **fallback_params)],
        num_retries=0,
        fallbacks=[{"a": ["b"]}],
    )
    host.answer("openai/a", RuntimeError("down"))

    response: Final = await complete(core, cache={"ttl": 5})

    primary, fallback = host.calls_to("acompletion")
    assert primary.kwargs["cache"] == primary_cache
    assert fallback.kwargs["cache"] == fallback_cache
    assert "allowed_cache_controls" not in fallback.kwargs
    assert headers(response).get(_HEADER) == ignored


@pytest.mark.asyncio
async def test_a_fallback_follows_the_cache_lock_of_a_requested_model_that_is_cooling_down() -> None:
    host: Final = ScriptedHost()
    core: Final = rust_core(
        host,
        model_list=[_deployment("a", cache=_OFF, allowed_cache_controls=[]), _deployment("b")],
        num_retries=0,
        fallbacks=[{"a": ["b"]}],
    )
    host.answer("openai/a", fresh(not_found))
    _ = await complete(core, cache={"ttl": 5})

    response: Final = await complete(core, cache={"ttl": 5})

    _, _, straight_to_fallback = host.calls_to("acompletion")
    assert straight_to_fallback.kwargs["model"] == "openai/b"
    assert straight_to_fallback.kwargs["cache"] == _OFF
    assert headers(response)[_HEADER] == "ttl"


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


def test_sync_text_completion_reports_the_ignored_cache_controls() -> None:
    response: Final = RustCore.build(
        build_router_config(
            {"model_list": [_deployment("a", cache=_OFF, allowed_cache_controls=[])], "num_retries": 0}
        ),
        ScriptedHost(),
    ).text_completion(model="a", prompt="hi", cache={"ttl": 600, "no-store": False})

    assert headers(response)[_HEADER] == "ttl,no-store"


class _RenderedPrompt:
    def __init__(self, model: str) -> None:
        self._model = model

    def get_chat_completion_prompt(self, **_: object) -> tuple[str, object, Mapping[str, object]]:  # kwargs-ok: stub
        return self._model, [{"role": "user", "content": "hi"}], {}


@pytest.mark.parametrize(
    "rendered_model",
    [
        pytest.param("openai/rendered", id="rendered model called directly"),
        pytest.param("b", id="rendered model is a group"),
    ],
)
@pytest.mark.asyncio
async def test_a_prompt_managed_model_sends_the_merged_cache_controls_and_reports_ignored_ones(
    rendered_model: str,
) -> None:
    host: Final = ScriptedHost()
    prompt_params: Final = {"model": "langfuse/a", "prompt_id": "p", "cache": _OFF, "allowed_cache_controls": []}
    core: Final = rust_core(
        host,
        model_list=[
            {"model_name": "a", "litellm_params": prompt_params, "model_info": {"id": "a"}},
            _deployment("b"),
        ],
        num_retries=0,
    )

    response: Final = await complete(
        core, cache={"ttl": 600, "no-store": False}, litellm_logging_obj=_RenderedPrompt(rendered_model)
    )

    (call,) = host.calls_to("acompletion")
    assert call.kwargs["cache"] == _OFF
    assert "allowed_cache_controls" not in call.kwargs
    assert headers(response)[_HEADER] == "ttl,no-store"
