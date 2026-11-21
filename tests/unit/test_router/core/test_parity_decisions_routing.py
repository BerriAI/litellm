"""
Parity of the Rust core with ``router_python_core.py`` on routing a Decisions call: a group's
deployments whose provider has no Decisions config are dropped before any later stage, unless that
leaves none (``_without_deployments_that_cannot_serve_route``), and a Decisions or SystemOne call
carrying audio, videos or an ``input_json`` part skips the deployments that refuse them, unless that
leaves none (``_without_decisions_deployments_refusing_media``).

The weights put the deployment that must be skipped first, so a core that does not narrow the
group picks it every time.
"""

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from typing import Final, cast

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.types.decisions import OpenAIDecisionInputMessage
from litellm.types.router import RouterRateLimitError
from tests.unit.test_router.core.endpoints_helpers import EndpointHost, patch_direct_endpoints
from tests.unit.test_router.core.scripted_host import auth_error, deployment, rate_limit

MakeCore = Callable[..., RouterCore]

QUESTION: Final[Mapping[str, object]] = {
    "input": "Is it?",
    "questions": [{"type": "predicate", "name": "is_it", "instructions": "Is it true?"}],
}
CLIP: Final = [{"url": "https://media.example/clip"}]


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> EndpointHost:
    """``litellm.decisions`` and ``litellm.systemone`` are bound directly, so they answer from this host."""
    endpoint_host: Final = EndpointHost()
    patch_direct_endpoints(monkeypatch, endpoint_host)
    return endpoint_host


def _sync(core: object, name: str) -> Callable[..., object]:
    """An endpoint both cores serve outside the ``RouterCore`` protocol."""
    return cast("Callable[..., object]", getattr(core, name))  # cast-ok: the member is a plain function


def _async(core: object, name: str) -> Callable[..., Coroutine[object, object, object]]:
    """``_sync`` for a coroutine endpoint."""
    return cast(
        "Callable[..., Coroutine[object, object, object]]", getattr(core, name)
    )  # cast-ok: the member is a coroutine function


def _call(core: RouterCore, op: str, **request: object) -> None:  # kwargs-ok: the endpoint's request
    """One call on ``op``: an ``a``-prefixed op is awaited, the other one called."""
    if op.startswith("a"):
        _ = asyncio.run(_async(core, op)(**request))
    else:
        _ = _sync(core, op)(**request)


# --- deployments whose provider has no Decisions config -------------------------------------------

_OPENAI: Final = deployment("e", "p7-openai", model="openai/gpt-4o", weight=0)
_ANTHROPIC: Final = deployment("e", "p7-anthropic", model="anthropic/claude-sonnet-4-5-20250929", weight=1)


def test_adecisions_skips_a_heavier_deployment_whose_provider_has_no_decisions_config(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=[_OPENAI, _ANTHROPIC], num_retries=0)

    for _ in range(5):
        _call(core, "adecisions", model="e", **QUESTION)

    assert host.deployments_called("adecisions") == ["p7-openai"] * 5


def test_sync_decisions_skips_a_heavier_deployment_whose_provider_has_no_decisions_config(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=[_OPENAI, _ANTHROPIC], num_retries=0)

    for _ in range(5):
        _call(core, "decisions", model="e", **QUESTION)

    assert host.deployments_called("decisions") == ["p7-openai"] * 5


@pytest.mark.parametrize("op", ["adecisions", "decisions"])
def test_a_group_no_provider_of_which_serves_decisions_still_picks(
    make_core: MakeCore, host: EndpointHost, op: str
) -> None:
    other: Final = deployment("e", "p7-anthropic-2", model="anthropic/claude-opus-4-1", weight=0)
    core: Final = make_core(model_list=[_ANTHROPIC, other], num_retries=0)

    _call(core, op, model="e", **QUESTION)

    assert host.deployments_called(op) == ["p7-anthropic"]


@pytest.mark.parametrize("op", ["adecisions", "decisions"])
def test_a_decisions_pick_by_id_is_not_narrowed(make_core: MakeCore, host: EndpointHost, op: str) -> None:
    core: Final = make_core(model_list=[_OPENAI, _ANTHROPIC], num_retries=0)

    _call(core, op, model="p7-anthropic", **QUESTION)

    assert host.deployments_called(op) == ["p7-anthropic"]


@pytest.mark.parametrize("op", ["adecisions", "decisions"])
def test_the_route_filter_runs_before_order(make_core: MakeCore, host: EndpointHost, op: str) -> None:
    first: Final = deployment("e", "p7-anthropic", model="anthropic/claude-sonnet-4-5-20250929", order=1)
    second: Final = deployment("e", "p7-openai", model="openai/gpt-4o", order=2)
    core: Final = make_core(model_list=[first, second], num_retries=0)

    _call(core, op, model="e", **QUESTION)

    assert host.deployments_called(op) == ["p7-openai"]


_T1_PUBLIC: Final = {"team_id": "T1", "team_public_model_name": "team-e"}
_T2_PUBLIC: Final = {"team_id": "T2", "team_public_model_name": "team-e"}


def test_adecisions_walks_the_order_tiers_of_the_litellm_metadata_team(
    make_core: MakeCore, host: EndpointHost
) -> None:
    host.answer("openai/t1-first", rate_limit())
    core: Final = make_core(
        model_list=[
            deployment("t1-first", "t1-first", order=1, model_info=_T1_PUBLIC),
            deployment("t2-first", "t2-first", order=2, model_info=_T2_PUBLIC),
            deployment("t1-second", "t1-second", order=3, model_info=_T1_PUBLIC),
        ],
        num_retries=0,
    )

    _call(core, "adecisions", model="team-e", litellm_metadata={"user_api_key_team_id": "T1"}, **QUESTION)

    assert host.deployments_called("adecisions") == ["t1-first", "t1-second"]


def test_the_route_filter_runs_before_cooldown(make_core: MakeCore, host: EndpointHost) -> None:
    """With the only Decisions deployment cooling down, the group has nothing to serve: the
    deployment the route filter dropped is not brought back."""
    host.answer("openai/gpt-4o", auth_error())
    core: Final = make_core(model_list=[_OPENAI, _ANTHROPIC], num_retries=0)

    with pytest.raises(litellm.AuthenticationError):
        _call(core, "adecisions", model="e", **QUESTION)
    with pytest.raises(RouterRateLimitError, match=r"(?i)no deployments available"):
        _call(core, "adecisions", model="e", **QUESTION)

    assert host.deployments_called("adecisions") == ["p7-openai"]


def test_the_retry_count_reads_the_group_before_the_route_filter(make_core: MakeCore, host: EndpointHost) -> None:
    """An authentication error is retried only when the group holds another deployment, counted
    without the route filter: both deployments count, so the Decisions deployment is retried."""
    host.answer("openai/gpt-4o", auth_error())
    core: Final = make_core(model_list=[_OPENAI, _ANTHROPIC], num_retries=2, retry_after=0, disable_cooldowns=True)

    with pytest.raises(litellm.AuthenticationError):
        _call(core, "adecisions", model="e", **QUESTION)

    assert host.deployments_called("adecisions") == ["p7-openai"] * 3


# --- deployments refusing a call's audio or videos ------------------------------------------------


def _media_group(**changes: object) -> list[dict[str, object]]:  # mutable-ok: a fresh model_list
    """``refuse`` declares no audio or video input and weighs more; ``take`` declares both."""
    flags: Final = ("supports_audio_input", "supports_video_input")
    return [
        deployment(
            "m",
            "refuse",
            model="hosted_vllm/refuse-m",
            api_base="https://vllm.example",
            weight=1,
            model_info=dict.fromkeys(flags, False),
            **changes,
        ),
        deployment(
            "m",
            "take",
            model="hosted_vllm/take-m",
            api_base="https://vllm.example",
            weight=0,
            model_info=dict.fromkeys(flags, True),
            **changes,
        ),
    ]


@pytest.mark.parametrize("media", ["audio", "videos"])
@pytest.mark.parametrize("op", ["adecisions", "decisions", "asystemone", "systemone"])
def test_a_call_carrying_media_skips_the_deployment_refusing_it(
    make_core: MakeCore, host: EndpointHost, op: str, media: str
) -> None:
    core: Final = make_core(model_list=_media_group(), num_retries=0)

    for _ in range(3):
        _call(core, op, model="m", **{media: CLIP}, **QUESTION)

    assert host.deployments_called(op) == ["take"] * 3


@pytest.mark.parametrize("op", ["adecisions", "decisions"])
def test_a_call_without_media_keeps_the_deployment_refusing_media(
    make_core: MakeCore, host: EndpointHost, op: str
) -> None:
    core: Final = make_core(model_list=_media_group(), num_retries=0)

    _call(core, op, model="m", **QUESTION)

    assert host.deployments_called(op) == ["refuse"]


@pytest.mark.parametrize(
    ("router_arguments", "request_extra"),
    [({"default_litellm_params": {"drop_params": True}}, {}), ({}, {"drop_params": True})],
    ids=["router-default", "request"],
)
def test_drop_params_keeps_the_deployment_refusing_media(
    make_core: MakeCore,
    host: EndpointHost,
    router_arguments: dict[str, object],
    request_extra: dict[str, object],
) -> None:
    """A deployment that drops the params it cannot send takes the call, media dropped."""
    core: Final = make_core(model_list=_media_group(), num_retries=0, **router_arguments)

    _call(core, "adecisions", model="m", audio=CLIP, **request_extra, **QUESTION)

    assert host.deployments_called("adecisions") == ["refuse"]


def test_the_media_filter_runs_before_order(make_core: MakeCore, host: EndpointHost) -> None:
    refuse, take = _media_group()
    cast("dict[str, object]", refuse["litellm_params"])["order"] = 1  # cast-ok: a model_list entry
    cast("dict[str, object]", take["litellm_params"])["order"] = 2  # cast-ok: a model_list entry
    core: Final = make_core(model_list=[refuse, take], num_retries=0)

    _call(core, "adecisions", model="m", audio=CLIP, **QUESTION)

    assert host.deployments_called("adecisions") == ["take"]


def _json_group() -> list[dict[str, object]]:  # mutable-ok: a fresh model_list
    """``no-json`` is a provider without JSON input and weighs more; ``json`` takes it."""
    return [
        deployment("j", "no-json", model="typesafe/jev-1.14.0", weight=1),
        deployment("j", "json", model="openai/gpt-4o", weight=0),
    ]


_EVIDENCE: Final = {"type": "input_json", "data": {"order_id": "A1", "charges": [49, 49]}}
_TICKET: Final = {"type": "input_text", "text": "Refund the duplicate charge."}
_RECEIPT: Final = {"type": "input_image", "image_url": "https://example.com/receipt.png"}
_POLICY: Final = {"role": "developer", "content": "Refund duplicate charges only."}


@pytest.mark.parametrize(
    "message",
    [{"role": "user", "content": [_TICKET, _EVIDENCE]}, OpenAIDecisionInputMessage(content=[_TICKET, _EVIDENCE])],
    ids=["as-sent", "validated"],
)
@pytest.mark.parametrize("op", ["adecisions", "decisions"])
def test_a_call_carrying_json_input_skips_the_deployment_refusing_it(
    make_core: MakeCore, host: EndpointHost, op: str, message: object
) -> None:
    core: Final = make_core(model_list=_json_group(), num_retries=0)

    for _ in range(3):
        _call(core, op, model="j", **{**QUESTION, "input": [message]})

    assert host.deployments_called(op) == ["json"] * 3


@pytest.mark.parametrize("op", ["adecisions", "decisions"])
def test_a_tuple_input_carrying_json_input_skips_the_deployment_refusing_it(
    make_core: MakeCore, host: EndpointHost, op: str
) -> None:
    core: Final = make_core(model_list=_json_group(), num_retries=0)
    message: Final = {"role": "user", "content": (_TICKET, _EVIDENCE)}

    for _ in range(3):
        _call(core, op, model="j", **{**QUESTION, "input": (message,)})

    assert host.deployments_called(op) == ["json"] * 3


@pytest.mark.parametrize(
    ("decision_input", "expected"),
    [
        ([{"role": "user", "content": [_EVIDENCE]}], ["json"] * 3),
        ([_POLICY, {"role": "user", "content": [_EVIDENCE]}], ["json"] * 3),
        ([{"role": "user", "content": [_RECEIPT, _EVIDENCE]}], ["json"] * 3),
        ([{"role": "user", "content": [_TICKET, _EVIDENCE]}], ["no-json"] * 3),
    ],
    ids=["only-json", "dropped-developer-message-and-json", "dropped-image-and-json", "json-and-text"],
)
def test_drop_params_skips_the_deployment_refusing_json_input_only_when_dropping_it_leaves_no_input(
    make_core: MakeCore, host: EndpointHost, decision_input: list[object], expected: list[str]
) -> None:
    core: Final = make_core(model_list=_json_group(), num_retries=0)

    for _ in range(3):
        _call(
            core,
            "adecisions",
            model="j",
            drop_params=True,
            **{**QUESTION, "input": decision_input},
        )

    assert host.deployments_called("adecisions") == expected


_GIF_RECEIPT: Final = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
_PNG_RECEIPT: Final = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.mark.parametrize(
    ("image_url", "expected"),
    [
        (_GIF_RECEIPT, ["json"] * 3),
        (_PNG_RECEIPT, ["no-json"] * 3),
        ("https://example.com/receipt.gif", ["json"] * 3),
        ("data:image/png;base64,", ["json"] * 3),
    ],
    ids=["gif-the-deployment-drops", "png-the-deployment-keeps", "remote-url-of-unknown-type", "malformed-data-url"],
)
def test_drop_params_counts_only_the_image_types_a_deployment_keeps(
    make_core: MakeCore, host: EndpointHost, image_url: str, expected: list[str]
) -> None:
    core: Final = make_core(model_list=_json_group(), num_retries=0)

    for _ in range(3):
        _call(
            core,
            "adecisions",
            model="j",
            drop_params=True,
            extra_headers={"typesafe-danger-zone-images": "true"},
            **{**QUESTION, "input": [{"role": "user", "content": [{"type": "input_image", "image_url": image_url}, _EVIDENCE]}]},
        )

    assert host.deployments_called("adecisions") == expected


@pytest.mark.parametrize(
    ("image_url", "expected"),
    [(_GIF_RECEIPT, ["json"] * 3), (_PNG_RECEIPT, ["openrouter-perplexity"] * 3)],
    ids=["gif-the-routed-model-drops", "png-the-routed-model-keeps"],
)
def test_drop_params_counts_only_the_image_types_the_openrouter_model_keeps(
    make_core: MakeCore, host: EndpointHost, image_url: str, expected: list[str]
) -> None:
    core: Final = make_core(
        model_list=[
            deployment("o", "openrouter-perplexity", model="openrouter/perplexity/sonar", weight=1),
            deployment("o", "json", model="openai/gpt-4o", weight=0),
        ],
        num_retries=0,
    )

    for _ in range(3):
        _call(
            core,
            "adecisions",
            model="o",
            drop_params=True,
            **{**QUESTION, "input": [{"role": "user", "content": [{"type": "input_image", "image_url": image_url}, _EVIDENCE]}]},
        )

    assert host.deployments_called("adecisions") == expected


@pytest.mark.parametrize("op", ["adecisions", "decisions"])
def test_a_call_with_only_text_parts_keeps_the_deployment_refusing_json_input(
    make_core: MakeCore, host: EndpointHost, op: str
) -> None:
    core: Final = make_core(model_list=_json_group(), num_retries=0)

    _call(core, op, model="j", **{**QUESTION, "input": [{"role": "user", "content": [_TICKET]}]})

    assert host.deployments_called(op) == ["no-json"]


def test_every_deployment_refusing_media_still_picks(make_core: MakeCore, host: EndpointHost) -> None:
    refuse, _ = _media_group()
    core: Final = make_core(model_list=[refuse, deployment("m", "refuse-2", model="openai/gpt-4o", weight=0)])

    _call(core, "adecisions", model="m", videos=CLIP, **QUESTION)

    assert host.deployments_called("adecisions") == ["refuse"]
