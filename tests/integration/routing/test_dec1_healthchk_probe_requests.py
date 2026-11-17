"""What a model health check sends upstream, per deployment mode and per health check knob.

A health check is a real provider call the proxy makes on its own: `GET /health` (or the background loop)
calls `litellm.ahealth_check` for every deployment with a fixed, tiny request. Operators pay for these
calls and providers rate limit them, so the exact shape is a contract: which endpoint is hit for each
`model_info.mode`, which prompt is sent, whether `max_tokens` is bounded, and how the per deployment
knobs `health_check_model`, `health_check_params`, `health_check_max_tokens`,
`health_check_supports_max_tokens` and `health_check_voice` change it.

One owned proxy runs without background health checks, so every probe in this file is caused by the
test's own `GET /health?model=<group>` call. Every deployment points at the scripted probe upstream of
`dec1_healthchk_support`, which records the path, headers and body of each request. The rules come from
`update_litellm_params_for_health_check` and `_resolve_health_check_max_tokens`
(litellm/proxy/health_check.py:786-983) and `HealthCheckHelpers.get_mode_handlers`
(litellm/litellm_core_utils/health_check_helpers.py:264-410). A Rust port of the health loop must send the
same requests, or providers start rejecting probes (a stray `max_tokens` is a 400 on embeddings) and every
deployment flips to unhealthy.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, object_value, string_value
from tests.integration.routing.dec1_healthchk_support import (
    PROBE_CHAT_MESSAGES,
    PROBE_PROMPT,
    PROVIDER_KEY,
    Rig,
    Seen,
    deployment,
    endpoint_for,
    endpoints,
    health,
    health_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

RUN: Final = uuid.uuid4().hex[:6]
UNKNOWN_MODE: Final = "dec1-not-a-mode"


def _m(label: str) -> str:
    """Upstream model name `ok-<label>-<run>`; the group name is the same string with `grp-` in front."""
    return f"ok-{label}-{RUN}"


def _g(label: str) -> str:
    return f"grp-{_m(label)}"


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    root: Final = api_base.removesuffix("/v1")
    return (
        deployment(_g("chat"), _m("chat"), api_base),
        deployment(_g("maxtok"), _m("maxtok"), api_base, model_info={"health_check_max_tokens": 3}),
        deployment(_g("nomaxtok"), _m("nomaxtok"), api_base, model_info={"health_check_supports_max_tokens": False}),
        deployment(
            _g("params"),
            _m("params"),
            api_base,
            model_info={"health_check_params": {"temperature": 0.25, "seed": 7, "max_tokens": 99}},
        ),
        deployment(
            _g("hcmodel"),
            _m("hcmodel"),
            api_base,
            model_info={"health_check_model": f"openai/{_m('hcmodel-alt')}"},
        ),
        deployment(_g("embed"), _m("embed"), api_base, model_info={"mode": "embedding"}),
        deployment(
            _g("text"), _m("text"), api_base, provider="text-completion-openai", model_info={"mode": "completion"}
        ),
        deployment(_g("image"), _m("image"), api_base, model_info={"mode": "image_generation"}),
        deployment(_g("speech"), _m("speech"), api_base, model_info={"mode": "audio_speech"}),
        deployment(
            _g("voice"), _m("voice"), api_base, model_info={"mode": "audio_speech", "health_check_voice": "nova"}
        ),
        deployment(_g("rerank"), _m("rerank"), root, provider="cohere", model_info={"mode": "rerank"}),
        deployment(_g("batch"), _m("batch"), api_base, model_info={"mode": "batch"}),
        deployment(_g("responses"), _m("responses"), api_base, model_info={"mode": "responses"}),
        deployment(
            _g("messages"), _m("messages"), root, provider="anthropic", model_info={"mode": "anthropic_messages"}
        ),
        deployment(_g("badmode"), _m("badmode"), api_base, model_info={"mode": UNKNOWN_MODE}),
        deployment(_g("costmap"), "text-embedding-3-small", api_base, model_info={"id": f"hc-costmap-{RUN}"}),
        deployment(
            _g("wildpinned"),
            "*",
            api_base,
            model_info={"id": f"hc-wildpinned-{RUN}", "health_check_model": f"openai/{_m('wild-probe')}"},
        ),
        deployment(_g("cache"), _m("cache"), api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(tmp_path_factory.mktemp("dec1-healthchk-probes"), _models) as value:
        yield value


def _probe_of(rig: Rig, label: str, *, upstream_model: str | None = None) -> Seen:
    """Run `GET /health?model=<group>` and return the one request the upstream saw for that deployment."""
    model: Final = upstream_model or _m(label)
    before: Final = rig.upstream.count(model)
    response: Final = health(rig.gateway, model=_g(label))
    assert response.status_code in (200, 503), response.text
    seen: Final = eventually(lambda: rig.upstream.calls(model)[before:], lambda calls: len(calls) >= 1, seconds=30)
    return seen[0]


def test_a_chat_probe_posts_one_fixed_user_message_to_chat_completions(rig: Rig) -> None:
    """A chat deployment with no health check settings is checked through `GET /health?model=<group>`.

    The upstream must receive exactly one `POST .../chat/completions` whose `messages` is a single user
    message with content `Hey how's it going?` or `What's 1 + 1?` (`_get_random_llm_message`), not the
    `DEFAULT_HEALTH_CHECK_PROMPT`. A rewrite that sends the prompt text instead still works, but changes
    what operators see in provider logs and what tokenizers bill.
    """
    probe: Final = _probe_of(rig, "chat")
    assert probe.method == "POST"
    assert probe.path.endswith("/chat/completions")
    messages: Final = probe.body["messages"]
    assert isinstance(messages, list) and len(messages) == 1
    first: Final = object_value(messages[0])
    assert first["role"] == "user"
    assert first["content"] in PROBE_CHAT_MESSAGES


def test_a_chat_probe_is_bounded_to_sixteen_output_tokens_by_default(rig: Rig) -> None:
    """The same chat deployment sets no `health_check_max_tokens` and the proxy has no
    `BACKGROUND_HEALTH_CHECK_MAX_TOKENS`.

    The probe body must carry `max_tokens: 16`, the non-wildcard default (health_check.py:830). Sixteen
    rather than one because GPT-5 class models reject `max_output_tokens` under 16.
    """
    probe: Final = _probe_of(rig, "chat")
    assert probe.body.get("max_tokens") == 16


def test_a_chat_probe_carries_the_deployments_own_provider_key(rig: Rig) -> None:
    """The probe is authenticated with the deployment's `litellm_params.api_key`.

    The upstream must see `Authorization: Bearer dec1-healthchk-provider-key`, never the master key or the
    caller's key: a health check is the proxy calling the provider as itself.
    """
    probe: Final = _probe_of(rig, "chat")
    assert probe.headers.get("authorization") == f"Bearer {PROVIDER_KEY}"


def test_a_chat_probe_leaks_no_proxy_bookkeeping_fields_to_the_provider(rig: Rig) -> None:
    """The proxy attaches `litellm_metadata`, a `user_api_key_auth` for the internal health account and
    `cache: {"no-cache": true}` to the probe's call params (`update_model_params_with_health_check_tracking_information`).

    None of that may reach the provider: the body must not contain `litellm_metadata`, `metadata`, `cache`
    or the text `litellm-internal-health-check`. UNVERIFIED: that OpenAI chat transformation drops all of them;
    look at `litellm/llms/openai/chat` if this fails.
    """
    probe: Final = _probe_of(rig, "chat")
    assert not {"litellm_metadata", "metadata", "cache", "model_info"} & set(probe.body)
    assert b"litellm-internal-health-check" not in probe.raw


def test_health_check_max_tokens_on_the_deployment_sets_the_probe_limit(rig: Rig) -> None:
    """The deployment sets `model_info.health_check_max_tokens: 3`.

    The probe body must carry `max_tokens: 3`: the explicit override wins over every default
    (`_resolve_health_check_max_tokens` step 1).
    """
    probe: Final = _probe_of(rig, "maxtok")
    assert probe.body.get("max_tokens") == 3


def test_health_check_supports_max_tokens_false_omits_max_tokens_from_a_chat_probe(rig: Rig) -> None:
    """The deployment sets `model_info.health_check_supports_max_tokens: false` and keeps chat mode.

    The probe body must have no `max_tokens` key at all. Operators use this for chat backends that reject
    the field, so sending `max_tokens: null` or 16 would break exactly the deployments that opted out.
    """
    probe: Final = _probe_of(rig, "nomaxtok")
    assert "max_tokens" not in probe.body


def test_health_check_params_are_merged_into_the_probe_body(rig: Rig) -> None:
    """The deployment sets `model_info.health_check_params: {temperature: 0.25, seed: 7, max_tokens: 99}`.

    The probe body must carry `temperature: 0.25` and `seed: 7`: the params are merged into the probe so a
    provider that requires extra payload fields can be checked.
    """
    probe: Final = _probe_of(rig, "params")
    assert probe.body.get("temperature") == 0.25
    assert probe.body.get("seed") == 7


def test_the_dedicated_max_tokens_knob_wins_over_max_tokens_in_health_check_params(rig: Rig) -> None:
    """The same deployment puts `max_tokens: 99` inside `health_check_params` but sets no dedicated knob.

    The probe must carry `max_tokens: 16`: `health_check_params` is applied first and the bounded default
    is written over it (health_check.py:885-903), so a stray params entry cannot make probes expensive.
    """
    probe: Final = _probe_of(rig, "params")
    assert probe.body.get("max_tokens") == 16


def test_health_check_model_replaces_the_upstream_model_of_the_probe(rig: Rig) -> None:
    """The deployment serves `openai/ok-hcmodel-<run>` and sets `health_check_model` to
    `openai/ok-hcmodel-alt-<run>`.

    `GET /health?model=<group>` must make the upstream receive a probe for `ok-hcmodel-alt-<run>` and none
    for `ok-hcmodel-<run>`: operators point probes at a cheaper model of the same endpoint.
    """
    before_real: Final = rig.upstream.count(_m("hcmodel"))
    probe: Final = _probe_of(rig, "hcmodel", upstream_model=_m("hcmodel-alt"))
    assert probe.path.endswith("/chat/completions")
    assert rig.upstream.count(_m("hcmodel")) == before_real


def test_the_health_report_names_the_health_check_model_it_probed(rig: Rig) -> None:
    """After the same check, the endpoint entry for deployment `hc-ok-hcmodel-<run>` in the `/health`
    answer is inspected.

    Its `model` field must be `openai/ok-hcmodel-alt-<run>`, the model that was actually called: the probe
    params are rewritten in place before the report is built (`update_litellm_params_for_health_check`
    mutates the deployment copy). UNVERIFIED that this is intended rather than incidental.
    """
    response: Final = health(rig.gateway, model=_g("hcmodel"))
    _, entry = endpoint_for(response, f"hc-{_m('hcmodel')}")
    assert entry["model"] == f"openai/{_m('hcmodel-alt')}"


def test_an_embedding_probe_posts_the_default_prompt_as_input_without_max_tokens(rig: Rig) -> None:
    """A deployment with `model_info.mode: embedding` is checked.

    The upstream must receive `POST .../embeddings` with `input: ["test from litellm"]` and no `max_tokens`
    and no `messages`: embeddings endpoints answer 400 `Unknown parameter: 'max_tokens'`.
    """
    probe: Final = _probe_of(rig, "embed")
    assert probe.path.endswith("/embeddings")
    assert probe.body.get("input") == [PROBE_PROMPT]
    assert "max_tokens" not in probe.body
    assert "messages" not in probe.body


def test_a_completion_probe_posts_the_default_prompt_to_completions_with_max_tokens(rig: Rig) -> None:
    """A `text-completion-openai/` deployment with `model_info.mode: completion` is checked.

    The upstream must receive `POST .../completions` (not chat) with `prompt: "test from litellm"` and
    `max_tokens: 16`: completion is one of the modes that accept the limit (`_MAX_TOKEN_SUPPORT_MODES`).
    UNVERIFIED that `prompt` arrives as a string rather than a one element list.
    """
    probe: Final = _probe_of(rig, "text")
    assert probe.path.endswith("/completions") and not probe.path.endswith("/chat/completions")
    assert probe.body.get("prompt") in (PROBE_PROMPT, [PROBE_PROMPT])
    assert probe.body.get("max_tokens") == 16


def test_an_image_generation_probe_posts_the_default_prompt_without_max_tokens(rig: Rig) -> None:
    """A deployment with `model_info.mode: image_generation` is checked.

    The upstream must receive `POST .../images/generations` with `prompt: "test from litellm"` and no
    `max_tokens`.
    """
    probe: Final = _probe_of(rig, "image")
    assert probe.path.endswith("/images/generations")
    assert probe.body.get("prompt") == PROBE_PROMPT
    assert "max_tokens" not in probe.body


def test_an_audio_speech_probe_speaks_the_default_prompt_in_the_alloy_voice(rig: Rig) -> None:
    """A deployment with `model_info.mode: audio_speech` and no `health_check_voice` is checked.

    The upstream must receive `POST .../audio/speech` with `input: "test from litellm"` and
    `voice: "alloy"`, the default voice (health_check.py:908).
    """
    probe: Final = _probe_of(rig, "speech")
    assert probe.path.endswith("/audio/speech")
    assert probe.body.get("input") == PROBE_PROMPT
    assert probe.body.get("voice") == "alloy"


def test_health_check_voice_picks_the_voice_of_an_audio_speech_probe(rig: Rig) -> None:
    """A speech deployment sets `model_info.health_check_voice: nova`.

    The probe body must carry `voice: "nova"`: providers whose voices do not include alloy need this to
    pass their health check at all.
    """
    probe: Final = _probe_of(rig, "voice")
    assert probe.body.get("voice") == "nova"


def test_a_rerank_probe_ranks_one_sample_document_against_the_default_prompt(rig: Rig) -> None:
    """A `cohere/` deployment with `model_info.mode: rerank` and an `api_base` on the scripted upstream is
    checked.

    The upstream must receive a POST to a path ending `/rerank` with `query: "test from litellm"` and
    `documents: ["my sample text"]`. UNVERIFIED that documents arrive as plain strings for the Cohere v1 or v2
    rerank transformation; see `litellm/llms/cohere/rerank*/transformation.py`.
    """
    probe: Final = _probe_of(rig, "rerank")
    assert probe.path.endswith("/rerank")
    assert probe.body.get("query") == PROBE_PROMPT
    assert probe.body.get("documents") in (["my sample text"], [{"text": "my sample text"}])


def test_a_batch_probe_lists_batches_instead_of_spending_tokens(rig: Rig) -> None:
    """An `openai/` deployment with `model_info.mode: batch` is checked.

    The upstream must receive a `GET` on a path ending `/batches` and no chat completion for that model:
    OpenAI is in `LIST_BATCHES_SUPPORTED_PROVIDERS`, so the probe only proves the key can list batches.
    """
    before_chat: Final = len(rig.upstream.on_path("/chat/completions"))
    before_batches: Final = len(rig.upstream.on_path("/batches"))
    response: Final = health(rig.gateway, model=_g("batch"))
    assert response.status_code == 200, response.text
    listed: Final = eventually(
        lambda: rig.upstream.on_path("/batches")[before_batches:], lambda calls: len(calls) >= 1, seconds=30
    )
    assert {seen.method for seen in listed} == {"GET"}
    assert all(seen.model != _m("batch") for seen in rig.upstream.on_path("/chat/completions")[before_chat:])


def test_a_responses_probe_posts_the_default_prompt_to_the_responses_api(rig: Rig) -> None:
    """A deployment with `model_info.mode: responses` is checked.

    The upstream must receive `POST .../responses` with `input: "test from litellm"`. The probe is bounded:
    the body must carry `max_output_tokens: 16` (`max_tokens` mapped for the Responses API). UNVERIFIED: the
    mapping of `max_tokens` to `max_output_tokens` for `aresponses`.
    """
    probe: Final = _probe_of(rig, "responses")
    assert probe.path.endswith("/responses")
    assert probe.body.get("input") == PROBE_PROMPT
    assert probe.body.get("max_output_tokens") == 16 or probe.body.get("max_tokens") == 16


def test_an_anthropic_messages_probe_posts_the_prompt_to_v1_messages_with_max_tokens(rig: Rig) -> None:
    """An `anthropic/` deployment with `model_info.mode: anthropic_messages` and `api_base` on the
    scripted upstream is checked.

    The upstream must receive `POST /v1/messages` with one user message whose content is
    `test from litellm` and `max_tokens: 16` (`ANTHROPIC_MESSAGES_HEALTH_CHECK_MAX_TOKENS`, also the
    chat-style default injected for this mode).
    """
    probe: Final = _probe_of(rig, "messages")
    assert probe.path.endswith("/v1/messages")
    assert probe.body.get("max_tokens") == 16
    messages: Final = probe.body["messages"]
    assert isinstance(messages, list) and object_value(messages[0]).get("content") in (
        PROBE_PROMPT,
        [{"type": "text", "text": PROBE_PROMPT}],
    )


def test_an_unknown_mode_is_reported_unhealthy_without_calling_the_provider(rig: Rig) -> None:
    """A deployment sets `model_info.mode: dec1-not-a-mode`.

    `GET /health?model=<group>` must answer 503 with the deployment in `unhealthy_endpoints`, its `error`
    starting `Mode dec1-not-a-mode not supported. See modes here: https://docs.litellm.ai/docs/proxy/health`,
    and the upstream must receive nothing for that model.
    """
    before: Final = rig.upstream.count(_m("badmode"))
    response: Final = health(rig.gateway, model=_g("badmode"))
    assert response.status_code == 503, response.text
    kind, entry = endpoint_for(response, f"hc-{_m('badmode')}")
    assert kind == "unhealthy"
    assert string_value(entry["error"]).startswith(
        f"Mode {UNKNOWN_MODE} not supported. See modes here: https://docs.litellm.ai/docs/proxy/health"
    )
    assert rig.upstream.count(_m("badmode")) == before


def test_a_deployment_without_a_mode_is_probed_in_the_mode_the_cost_map_gives_its_model(rig: Rig) -> None:
    """A deployment serves `openai/text-embedding-3-small` and sets no `model_info.mode`.

    The probe must be `POST .../embeddings` with `input: ["test from litellm"]` and no `max_tokens`: the mode
    is resolved from the cost map entry `embedding` (`resolve_health_check_mode`). Probing it as chat would
    send `max_tokens` and fail every time.
    """
    probe: Final = _probe_of(rig, "costmap", upstream_model="text-embedding-3-small")
    assert probe.path.endswith("/embeddings")
    assert "max_tokens" not in probe.body


def test_a_wildcard_deployment_with_a_health_check_model_probes_that_model_without_max_tokens(rig: Rig) -> None:
    """A wildcard deployment `openai/*` sets `health_check_model: openai/ok-wild-probe-<run>`.

    The upstream must receive one chat probe for `ok-wild-probe-<run>` with no `max_tokens`: the limit is
    resolved from the deployment model `openai/*`, which is a wildcard and gets none (step 6 of
    `_resolve_health_check_max_tokens`), before the model is replaced. Read from code; a port that resolves
    the limit after the swap would start sending 16.
    """
    probe: Final = _probe_of(rig, "wildpinned", upstream_model=_m("wild-probe"))
    assert probe.path.endswith("/chat/completions")
    assert "max_tokens" not in probe.body


def test_two_health_checks_reach_the_provider_twice_despite_the_response_cache(rig: Rig) -> None:
    """The proxy runs with the Redis response cache on (`litellm_settings.cache: true`). The same group is
    checked twice in a row.

    The upstream must receive two probes: health checks set `cache: {"no-cache": true}` (main.py:9603), so a
    cached answer can never make a dead deployment look healthy.
    """
    before: Final = rig.upstream.count(_m("cache"))
    for _ in range(2):
        assert health(rig.gateway, model=_g("cache")).status_code == 200
    eventually(lambda: rig.upstream.count(_m("cache")) - before, lambda count: count >= 2, seconds=30)


def test_a_healthy_endpoint_reports_the_providers_remaining_rate_limit_headers(rig: Rig) -> None:
    """The upstream answers the chat probe with `x-ratelimit-remaining-requests: 4999` and
    `x-ratelimit-remaining-tokens: 99999`.

    The healthy endpoint entry must carry those two fields with the values `4999` and `99999` (as the
    provider sent them, strings or integers): operators read remaining quota off `/health`.
    """
    response: Final = health(rig.gateway, model=_g("chat"))
    assert response.status_code == 200, response.text
    (entry,) = endpoints(response, "healthy")
    assert str(entry["x-ratelimit-remaining-requests"]) == "4999"
    assert str(entry["x-ratelimit-remaining-tokens"]) == "99999"
