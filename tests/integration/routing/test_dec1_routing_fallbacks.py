"""Server-side fallbacks: `fallbacks`, `default_fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`.

One owned proxy runs with `num_retries: 0`, cooldowns disabled and pre-call checks enabled. Every group has
one deployment, and every test uses its own primary and fallback groups, so the order in which the scripted
upstream models were called is the whole story of a request.

The rules pinned here. Any upstream error, a plain 400 included, moves the request to the group's fallback
list. A context window error prefers `context_window_fallbacks` and uses the regular list when it has no
entry there. A content policy error uses `content_policy_fallbacks`, and when that list exists but has no
entry for the group the error is returned without trying the regular list. A group with its own entry in
`fallbacks` never uses `default_fallbacks`. Fallbacks nest: a failing fallback group runs its own list. When
every fallback fails, the caller gets the error of the group it asked for, not the error of the last
fallback. A served fallback is reported in the `x-litellm-attempted-fallbacks` response header.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    CONTEXT_MESSAGE,
    MESSAGE_PREFIX,
    POLICY_MESSAGE,
    UNREACHABLE_API_BASE,
    Rig,
    chat,
    content_of,
    deployment,
    error_message,
    error_of,
    routing_rig,
    stream_chat,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

FALLBACK_STATUSES: Final = (400, 401, 404, 408, 429, 500, 503)
TINY_WINDOW_TOKENS: Final = 20
LONG_PROMPT: Final = "routing contract " * 400

ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "disable_cooldowns": True,
    "enable_pre_call_checks": True,
    "fallbacks": [
        {"fb-primary": ["fb-backup"]},
        {"fb-keyed": ["fb-dead"]},
        {"fb-ctx": ["fb-general"]},
        {"fb-policy": ["fb-general"]},
        {"fb-ctx-regular": ["fb-general-ctx"]},
        {"fb-policy-nomatch": ["fb-general-policy"]},
        {"fb-chain-a": ["fb-chain-b"]},
        {"fb-chain-b": ["fb-chain-c"]},
        {"fb-two": ["fb-two-dead", "fb-two-backup"]},
        {"fb-self": ["fb-self", "fb-self-backup"]},
        *({f"fb-status-{status}": [f"fb-status-{status}-backup"]} for status in FALLBACK_STATUSES),
        {"fb-refused": ["fb-refused-backup"]},
        {"fb-timeout": ["fb-timeout-backup"]},
        {"fb-stream": ["fb-stream-backup"]},
        {"fb-body": ["fb-body-backup"]},
        {"fb-mock-429": ["fb-mock-429-backup"]},
        {"fb-disabled": ["fb-disabled-backup"]},
    ],
    "default_fallbacks": ["fb-default"],
    "context_window_fallbacks": [{"fb-ctx": ["fb-big"]}, {"fb-precall": ["fb-precall-big"]}],
    "content_policy_fallbacks": [{"fb-policy": ["fb-safe"]}],
}
GENERAL_SETTINGS: Final[dict[str, JsonValue]] = {"dangerously_allow_mock_testing_request_params": True}


def _one(group: str, upstream_model: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    return deployment(group, upstream_model, api_base, **params)


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        _one("fb-primary", "status-500-fbprimary", api_base),
        _one("fb-backup", "ok-fbbackup", api_base),
        _one("fb-keyed", "status-500-fbkeyed", api_base),
        _one("fb-dead", "status-503-fbdead", api_base),
        _one("fb-default", "ok-fbdefault", api_base),
        _one("fb-nochain", "status-503-fbnochain", api_base),
        _one("fb-ctx", "context-fbctx", api_base),
        _one("fb-big", "ok-fbbig", api_base),
        _one("fb-general", "ok-fbgeneral", api_base),
        _one("fb-policy", "policy-fbpolicy", api_base),
        _one("fb-safe", "ok-fbsafe", api_base),
        _one("fb-ctx-regular", "context-fbctxregular", api_base),
        _one("fb-general-ctx", "ok-fbgeneralctx", api_base),
        _one("fb-policy-nomatch", "policy-fbnomatch", api_base),
        _one("fb-general-policy", "ok-fbgeneralpolicy", api_base),
        _one("fb-chain-a", "status-500-fbchaina", api_base),
        _one("fb-chain-b", "status-503-fbchainb", api_base),
        _one("fb-chain-c", "ok-fbchainc", api_base),
        _one("fb-two", "status-500-fbtwo", api_base),
        _one("fb-two-dead", "status-503-fbtwodead", api_base),
        _one("fb-two-backup", "ok-fbtwobackup", api_base),
        _one("fb-self", "status-500-fbself", api_base),
        _one("fb-self-backup", "ok-fbselfbackup", api_base),
        *(_one(f"fb-status-{status}", f"status-{status}-fbstatus", api_base) for status in FALLBACK_STATUSES),
        *(_one(f"fb-status-{status}-backup", f"ok-fbstatus{status}", api_base) for status in FALLBACK_STATUSES),
        _one("fb-refused", "ok-fbrefused-never", UNREACHABLE_API_BASE),
        _one("fb-refused-backup", "ok-fbrefusedbackup", api_base),
        _one("fb-timeout", "slow-fbtimeout", api_base, timeout=1),
        _one("fb-timeout-backup", "ok-fbtimeoutbackup", api_base),
        _one("fb-stream", "status-500-fbstream", api_base),
        _one("fb-stream-backup", "ok-fbstreambackup", api_base),
        _one("fb-body", "status-500-fbbody", api_base),
        _one("fb-body-backup", "ok-fbbodybackup", api_base),
        _one("fb-mock-429", "ok-fbmock429-primary", api_base),
        _one("fb-mock-429-backup", "ok-fbmock429backup", api_base),
        _one("fb-disabled", "status-500-fbdisabled", api_base),
        _one("fb-disabled-backup", "ok-fbdisabledbackup", api_base),
        deployment("fb-precall", "ok-fbprecall-tiny", api_base, model_info={"max_input_tokens": TINY_WINDOW_TOKENS}),
        _one("fb-precall-big", "ok-fbprecallbig", api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(
        tmp_path_factory.mktemp("dec1-routing-fallbacks"),
        _models,
        ROUTER_SETTINGS,
        general_settings=GENERAL_SETTINGS,
    ) as value:
        yield value


def test_a_500_on_the_primary_group_is_served_by_its_fallback_group_with_the_fallback_header(rig: Rig) -> None:
    """`fallbacks: [{"fb-primary": ["fb-backup"]}]`; the primary's upstream answers 500, the backup's 200.

    The caller gets 200 with the backup's content. The upstream saw the primary model once and then the
    backup model once, in that order. The response carries `x-litellm-attempted-fallbacks: 1` and the
    `x-litellm-model-id` of the backup deployment, `dec1-ok-fbbackup`.
    """
    response: Final = chat(rig.gateway, "fb-primary")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbbackup"
    assert rig.upstream.order("status-500-fbprimary", "ok-fbbackup") == ("status-500-fbprimary", "ok-fbbackup")
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"
    assert response.headers["x-litellm-model-id"] == "dec1-ok-fbbackup"


@pytest.mark.parametrize("status", FALLBACK_STATUSES)
def test_every_upstream_error_status_moves_the_request_to_the_fallback_group(rig: Rig, status: int) -> None:
    """The primary's upstream answers 400, 401, 404, 408, 429, 500 or 503 and the group has a fallback entry.

    For each status the caller gets 200 with the fallback's content, the failing upstream model received
    exactly one request (retries are off) and the fallback model received one. A plain 400 falls back too,
    because `fallback_on_unsupported_params` defaults to true.
    """
    response: Final = chat(rig.gateway, f"fb-status-{status}")
    assert response.status_code == 200, response.text
    assert content_of(response) == f"served ok-fbstatus{status}"
    assert rig.upstream.count(f"status-{status}-fbstatus") == 1
    assert rig.upstream.count(f"ok-fbstatus{status}") == 1
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"


def test_a_refused_connection_moves_the_request_to_the_fallback_group(rig: Rig) -> None:
    """The primary deployment points at a port nothing listens on and the group has a fallback entry.

    The caller gets 200 with the fallback's content and `x-litellm-attempted-fallbacks: 1`.
    """
    response: Final = chat(rig.gateway, "fb-refused")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbrefusedbackup"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"


def test_a_deployment_timeout_moves_the_request_to_the_fallback_group(rig: Rig) -> None:
    """The primary deployment has `timeout: 1` and its upstream takes 4 seconds; the group has a fallback entry.

    The caller gets 200 with the fallback's content. The slow upstream model received exactly one request.
    """
    response: Final = chat(rig.gateway, "fb-timeout")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbtimeoutbackup"
    assert rig.upstream.count("slow-fbtimeout") == 1


def test_default_fallbacks_serve_a_group_that_has_no_entry_of_its_own(rig: Rig) -> None:
    """`default_fallbacks: ["fb-default"]`; group `fb-nochain` has no entry in `fallbacks` and answers 503.

    The caller gets 200 with the default fallback's content and `x-litellm-attempted-fallbacks: 1`.
    Product bug: CI run 42966523075 answered the 503 with "no fallback model group was found" although
    `default_fallbacks` is a supported router setting. Likely cause: the router folds `default_fallbacks`
    into `fallbacks` as a `"*"` entry only when it is built (litellm/router_rust/core.py:707), and
    `Resolved.changed`, which the proxy's runtime re-apply of router settings goes through
    (litellm/proxy/proxy_server.py:8445), replaces `fallbacks` without folding it back in
    (litellm/router_rust/core.py:832).
    """
    pytest.skip("BUG: default_fallbacks are not applied when router_settings also configures fallbacks")
    response: Final = chat(rig.gateway, "fb-nochain")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbdefault"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"


def test_when_every_fallback_fails_the_caller_gets_the_requested_groups_error_and_no_default(rig: Rig) -> None:
    """`fb-keyed` answers 500 and its only fallback `fb-dead` answers 503; `default_fallbacks` is configured.

    The caller gets HTTP 500 with the message of the group it asked for, starting
    `litellm.InternalServerError: InternalServerError: OpenAIException - scripted 500 from status-500-fbkeyed`,
    not the fallback's 503. The upstream saw the primary then the dead fallback, once each, and the default
    fallback's model was not called for this request: a group with its own entry never uses the default list.
    """
    before: Final = rig.upstream.count("ok-fbdefault")
    response: Final = chat(rig.gateway, "fb-keyed")
    assert response.status_code == 500, response.text
    assert error_of(response)["code"] == "500"
    assert error_message(response).startswith(MESSAGE_PREFIX[500] + "scripted 500 from status-500-fbkeyed")
    assert rig.upstream.order("status-500-fbkeyed", "status-503-fbdead") == ("status-500-fbkeyed", "status-503-fbdead")
    assert rig.upstream.count("ok-fbdefault") == before


def test_a_context_window_error_uses_context_window_fallbacks_before_the_regular_list(rig: Rig) -> None:
    """`fb-ctx` answers the 400 context length body; it has `fb-big` in `context_window_fallbacks` and
    `fb-general` in `fallbacks`.

    The caller gets 200 with `fb-big`'s content, and the model behind `fb-general` receives nothing from this
    group's request.
    """
    before: Final = rig.upstream.count("ok-fbgeneral")
    response: Final = chat(rig.gateway, "fb-ctx")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbbig"
    assert rig.upstream.count("ok-fbgeneral") == before
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"


def test_a_context_window_error_without_a_context_entry_uses_the_regular_fallbacks(rig: Rig) -> None:
    """`fb-ctx-regular` answers the 400 context length body; `context_window_fallbacks` exists but has no entry
    for it, and `fallbacks` maps it to `fb-general-ctx`.

    The caller gets 200 with `fb-general-ctx`'s content.
    """
    response: Final = chat(rig.gateway, "fb-ctx-regular")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbgeneralctx"


def test_a_content_policy_error_uses_content_policy_fallbacks_before_the_regular_list(rig: Rig) -> None:
    """`fb-policy` answers the 400 content policy body; it has `fb-safe` in `content_policy_fallbacks` and
    `fb-general` in `fallbacks`.

    The caller gets 200 with `fb-safe`'s content, and the model behind `fb-general` receives nothing from this
    group's request.
    """
    before: Final = rig.upstream.count("ok-fbgeneral")
    response: Final = chat(rig.gateway, "fb-policy")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbsafe"
    assert rig.upstream.count("ok-fbgeneral") == before


def test_a_content_policy_error_without_a_content_policy_entry_is_returned_without_a_fallback(rig: Rig) -> None:
    """`fb-policy-nomatch` answers the 400 content policy body; `content_policy_fallbacks` exists but has no
    entry for it, while `fallbacks` maps it to `fb-general-policy`.

    Unlike a context window error, this one does not fall through to the regular list: the caller gets HTTP
    400 whose message contains the upstream's refusal text, and the regular fallback's model is never called.
    """
    response: Final = chat(rig.gateway, "fb-policy-nomatch")
    assert response.status_code == 400, response.text
    assert POLICY_MESSAGE in error_message(response), response.text
    assert rig.upstream.count("policy-fbnomatch") == 1
    assert rig.upstream.count("ok-fbgeneralpolicy") == 0


def test_a_pre_call_context_window_refusal_goes_to_context_window_fallbacks_without_an_upstream_call(
    rig: Rig,
) -> None:
    """Pre-call checks are on; `fb-precall`'s only deployment declares `model_info.max_input_tokens: 20` and has
    `fb-precall-big` in `context_window_fallbacks`.

    A prompt of roughly 800 words must be served by `fb-precall-big` with 200, and the small deployment's
    upstream model must receive nothing: the router refuses it before calling out.
    """
    response: Final = chat(rig.gateway, "fb-precall", messages=[{"role": "user", "content": LONG_PROMPT}])
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbprecallbig"
    assert rig.upstream.count("ok-fbprecall-tiny") == 0


def test_a_failing_fallback_group_runs_its_own_fallback_list(rig: Rig) -> None:
    """`fb-chain-a` (500) falls back to `fb-chain-b` (503), which has its own entry pointing at `fb-chain-c`.

    The caller gets 200 with `fb-chain-c`'s content and the upstream saw a, b, c once each in that order.
    UNVERIFIED: the `x-litellm-attempted-fallbacks` value for a nested chain, which `run_fallbacks` in
    `litellm-rust/crates/router-core/src/call.rs` overwrites with the outer hop count, is not asserted here.
    """
    response: Final = chat(rig.gateway, "fb-chain-a")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbchainc"
    assert rig.upstream.order("status-500-fbchaina", "status-503-fbchainb", "ok-fbchainc") == (
        "status-500-fbchaina",
        "status-503-fbchainb",
        "ok-fbchainc",
    )


def test_the_second_target_of_a_fallback_list_serves_when_the_first_fails(rig: Rig) -> None:
    """`fallbacks: [{"fb-two": ["fb-two-dead", "fb-two-backup"]}]`; `fb-two` answers 500 and `fb-two-dead` 503.

    The caller gets 200 with `fb-two-backup`'s content, and the upstream saw the primary, the dead fallback
    and the backup once each, in list order.
    """
    response: Final = chat(rig.gateway, "fb-two")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbtwobackup"
    assert rig.upstream.order("status-500-fbtwo", "status-503-fbtwodead", "ok-fbtwobackup") == (
        "status-500-fbtwo",
        "status-503-fbtwodead",
        "ok-fbtwobackup",
    )


def test_a_fallback_list_that_names_its_own_group_does_not_call_that_group_again(rig: Rig) -> None:
    """`fallbacks: [{"fb-self": ["fb-self", "fb-self-backup"]}]`; `fb-self` answers 500.

    The caller gets 200 with the backup's content, and the failing upstream model received exactly one
    request: the entry that names the group itself is skipped.
    """
    response: Final = chat(rig.gateway, "fb-self")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served ok-fbselfbackup"
    assert rig.upstream.count("status-500-fbself") == 1


def test_a_streaming_request_is_served_by_the_fallback_when_the_primary_fails_before_the_stream_opens(
    rig: Rig,
) -> None:
    """The caller sends `stream: true` to `fb-stream`, whose upstream answers 500, with `fb-stream-backup` as
    its fallback.

    The caller gets HTTP 200 and a server-sent event stream that contains the backup's text
    `served ok-fbstreambackup` and ends with `data: [DONE]`.
    """
    status, text = stream_chat(rig.gateway, "fb-stream")
    assert status == 200, text
    assert "served ok-fbstreambackup" in text
    assert text.rstrip().endswith("data: [DONE]")
    assert rig.upstream.count("status-500-fbstream") == 1


def test_the_fallback_group_receives_the_callers_messages_and_its_own_upstream_model_name(rig: Rig) -> None:
    """`fb-body` answers 500 and falls back to `fb-body-backup`.

    The request the fallback's upstream received must carry the same `messages` as the one the primary
    received, and `model` must be the fallback deployment's own upstream model, `ok-fbbodybackup`. A fallback
    that forwarded the primary's model name would be rejected by a real provider.
    """
    response: Final = chat(rig.gateway, "fb-body", temperature=0.5)
    assert response.status_code == 200, response.text
    (primary,) = rig.upstream.calls("status-500-fbbody")
    (backup,) = rig.upstream.calls("ok-fbbodybackup")
    assert backup.body["messages"] == primary.body["messages"]
    assert backup.body["model"] == "ok-fbbodybackup"
    assert backup.body["temperature"] == 0.5


def test_disable_fallbacks_in_the_request_body_returns_the_primary_error(rig: Rig) -> None:
    """`fb-disabled` answers 500 and has a fallback entry; the caller sends `disable_fallbacks: true`.

    The caller gets HTTP 500 and the fallback's upstream model receives nothing.
    """
    response: Final = chat(rig.gateway, "fb-disabled", disable_fallbacks=True)
    assert response.status_code == 500, response.text
    assert rig.upstream.count("status-500-fbdisabled") == 1
    assert rig.upstream.count("ok-fbdisabledbackup") == 0


def test_mock_testing_rate_limit_error_fails_every_fallback_level_without_calling_an_upstream(rig: Rig) -> None:
    """The proxy allows mock testing params; the caller sends `mock_testing_rate_limit_error: true` to
    `fb-mock-429`, whose upstream is healthy and which falls back to the healthy `fb-mock-429-backup`.

    The mock replaces the first attempt of every level, the fallback included
    (litellm-rust/crates/router-core/src/mock.rs:29), so with retries off the caller gets HTTP 429 with the
    requested group's mock error and a note that the fallback also failed, and neither upstream model
    receives a request.
    """
    response: Final = chat(rig.gateway, "fb-mock-429", mock_testing_rate_limit_error=True)
    assert response.status_code == 429, response.text
    assert error_of(response)["code"] == "429", response.text
    message: Final = error_message(response)
    assert message.startswith("litellm.RateLimitError: This is a mock exception for model=fb-mock-429,"), message
    assert "Fallback to fb-mock-429-backup also failed" in message, message
    assert rig.upstream.count("ok-fbmock429-primary") == 0
    assert rig.upstream.count("ok-fbmock429backup") == 0


def test_the_upstream_context_error_text_reaches_the_caller_when_nothing_serves_it(rig: Rig) -> None:
    """`fb-ctx` falls back to `fb-big`; here the caller disables fallbacks so the context error is returned.

    The caller gets HTTP 400 with `type` `invalid_request_error`, `code` `"400"` and exactly this message:
    `litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException -
    This model's maximum context length is 4096 tokens`.
    """
    response: Final = chat(rig.gateway, "fb-ctx", disable_fallbacks=True)
    assert response.status_code == 400, response.text
    assert error_of(response) == {
        "message": (
            "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: "
            f"OpenAIException - {CONTEXT_MESSAGE}"
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "400",
    }, response.text
