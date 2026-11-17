"""Contracts for when each guardrail mode runs and what it can change, driven through a real proxy.

A guardrail named in a request runs at the point its `mode` says: `pre_call` before the provider is called (it can
refuse the request or rewrite the prompt the provider sees), `during_call` alongside the provider call (it can only
refuse), and `post_call` after the provider answered (it can refuse or rewrite the answer). A list mode runs at each
point it names. A guardrail with `default_on: false` runs only when the request names it, either in the body
`guardrails` field, in `metadata.guardrails`, or as a `{name: {"extra_body": ...}}` entry that passes per-request
parameters on to the guardrail service. Every guardrail that ran is named in the `x-litellm-applied-guardrails`
response header.

All guardrails here are Generic Guardrail API guardrails pointed at a scripted policy peer (see
`dec1_guardrails_support`). These pins matter for the Rust port of the request path: the guardrail hooks are called
from Python around a Rust router call and a Rust `/v1/messages` path, and the order and number of calls, the payload
the guardrail service receives and the request the provider finally receives are all user-visible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import object_value
from integration.observability.dec1_guardrails_support import (
    APPLIED_HEADER,
    CALL_ID_HEADER,
    MASKED_SECRET_ANSWER,
    PLAIN_ANSWER,
    POLICY_KEY,
    SECRET,
    Rig,
    boot_rig,
    chat,
    chat_text,
    error_message,
    generic,
    marker,
    prompts_seen_by_provider,
    rails_called,
    refused_input,
    refused_output,
)
from pydantic import JsonValue

PRE: Final = "dec1-gr-pre"
PRE_TWO: Final = "dec1-gr-pre-two"
DURING: Final = "dec1-gr-during"
POST: Final = "dec1-gr-post"
BOTH: Final = "dec1-gr-both"
NEVER_CONFIGURED: Final = "dec1-gr-never-configured"


def _guardrails(policy_url: str) -> list[dict[str, JsonValue]]:
    return [
        generic(PRE, "pre_call", policy_url),
        generic(PRE_TWO, "pre_call", policy_url),
        generic(DURING, "during_call", policy_url),
        generic(POST, "post_call", policy_url),
        generic(BOTH, ["pre_call", "post_call"], policy_url),
    ]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with boot_rig(tmp_path_factory.mktemp("dec1-guardrails-modes"), _guardrails) as value:
        yield value


@pytest.fixture(autouse=True)
def _drained(rig: Rig) -> None:
    rig.drain()


def _requested(*names: str) -> dict[str, JsonValue]:
    return {"guardrails": list(names)}


def test_a_requested_pre_call_guardrail_sees_the_prompt_before_the_provider_is_called(rig: Rig) -> None:
    """A request names the pre_call guardrail and the peer answers NONE. The peer gets exactly one call with
    `input_type` "request" whose `texts` contain the prompt, the provider then gets the prompt unchanged, and the
    caller gets the provider's answer. A port that moves the hook after dispatch would show the provider call first."""
    prompt: Final = f"hello {marker()}"
    response: Final = chat(rig, prompt, extra=_requested(PRE))
    assert chat_text(response) == PLAIN_ANSWER
    calls: Final = rig.policy_calls()
    assert rails_called(calls) == ((PRE, "request"),)
    assert any(prompt in str(text) for text in calls[0]["texts"] or []), calls
    assert prompts_seen_by_provider(rig.provider_calls()) == (prompt,)


def test_the_guardrail_service_gets_the_api_key_header_call_id_and_model(rig: Rig) -> None:
    """The guardrail is configured with `api_key`. The peer's request must carry it as the `x-api-key` header,
    its body `litellm_call_id` must equal the caller's `x-litellm-call-id` response header, and `model` must be the
    model group the caller asked for (read from code: generic_guardrail_api.py `GenericGuardrailAPIRequest`)."""
    response: Final = chat(rig, f"identify {marker()}", extra=_requested(PRE))
    assert response.status_code == 200, response.text
    requests: Final = rig.policy.drain()
    assert len(requests) == 1, requests
    assert requests[0].headers.get("x-api-key") == POLICY_KEY, requests[0].headers
    body: Final = json.loads(requests[0].body)
    assert body["litellm_call_id"] == response.headers[CALL_ID_HEADER], body
    assert body["input_type"] == "request", body
    assert body["model"] == rig.chat_model, body


def test_the_guardrail_service_gets_the_callers_key_and_team_identity(rig: Rig) -> None:
    """A key with an alias inside a team calls with the guardrail requested. The peer body's `request_data` must
    carry `user_api_key_alias`, `user_api_key_team_id` and `user_api_key_hash` (the hashed token), which guardrail
    services use for per-tenant policy; the Rust auth port must keep filling these metadata fields."""
    team: Final = rig.scenario.team()
    alias: Final = f"dec1-alias-{marker()}"
    key: Final = rig.scenario.key(team_id=team, key_alias=alias)
    response: Final = chat(rig, f"identity {marker()}", key=key, extra=_requested(PRE))
    assert response.status_code == 200, response.text
    calls: Final = rig.policy_calls()
    assert len(calls) == 1, calls
    identity: Final = object_value(calls[0]["request_data"])
    assert identity.get("user_api_key_alias") == alias, identity
    assert identity.get("user_api_key_team_id") == team, identity
    assert isinstance(identity.get("user_api_key_hash"), str) and len(str(identity["user_api_key_hash"])) == 64


def test_the_guardrail_service_gets_allowlisted_inbound_headers_and_placeholders_for_the_rest(rig: Rig) -> None:
    """The caller sends `x-litellm-dec1-tag: visible` and `x-dec1-private: hidden`. The peer body's
    `request_headers` must carry the first value as sent (the `x-litellm-*` glob is allowlisted) and the second as
    the literal "[present]", and must never carry the caller's Authorization value (read from code:
    generic_guardrail_api.py `_HEADER_VALUE_ALLOWLIST`, `_HEADER_PRESENT_PLACEHOLDER`)."""
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": f"hdr {marker()}"}], "guardrails": [PRE]},
        headers={"x-litellm-dec1-tag": "visible", "x-dec1-private": "hidden"},
    )
    assert response.status_code == 200, response.text
    calls: Final = rig.policy_calls()
    headers: Final = {str(k).lower(): v for k, v in object_value(calls[0]["request_headers"]).items()}
    assert headers.get("x-litellm-dec1-tag") == "visible", headers
    assert headers.get("x-dec1-private") == "[present]", headers
    assert headers.get("authorization") == "[present]", headers


def test_a_pre_call_block_refuses_with_400_and_never_calls_the_provider(rig: Rig) -> None:
    """The prompt carries BLOCK-IN so the peer answers BLOCKED. The caller gets 400 whose `error.message` is the
    peer's `blocked_reason` verbatim ("dec1 refused input by dec1-gr-pre", not wrapped in a default message), and the
    provider receives nothing."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra=_requested(PRE))
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_input(PRE)
    assert rig.provider_calls() == ()


def test_when_two_guardrails_are_requested_the_first_to_block_answers(rig: Rig) -> None:
    """Two pre_call guardrails are requested and the prompt carries BLOCK-IN, which both would refuse. The caller
    gets 400 naming the rail configured first ("dec1 refused input by dec1-gr-pre") and the other rail is never
    called, because pre_call guardrails run one after the other in callback (config) order and stop at the first
    refusal."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra=_requested(PRE, PRE_TWO))
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_input(PRE)
    assert rails_called(rig.policy_calls()) == ((PRE, "request"),)


def test_pre_call_masking_rewrites_the_prompt_the_provider_receives(rig: Rig) -> None:
    """The prompt carries SECRET-1234 and the peer answers GUARDRAIL_INTERVENED with that token replaced by
    [MASKED]. The provider must receive the masked prompt and never the raw token, while the caller still gets 200
    with the provider's answer."""
    tag: Final = marker()
    response: Final = chat(rig, f"my token is {SECRET} {tag}", extra=_requested(PRE))
    assert chat_text(response) == PLAIN_ANSWER
    seen: Final = prompts_seen_by_provider(rig.provider_calls())
    assert seen == (f"my token is [MASKED] {tag}",), seen


def test_two_masking_pre_call_guardrails_apply_in_turn(rig: Rig) -> None:
    """Two pre_call guardrails are requested and the prompt carries the secret twice. The first one masks, so the
    second must be shown the already masked text (no SECRET-1234 in its texts), and the provider must receive the
    masked prompt once. A port that fans pre_call guardrails out in parallel on the original body would break this."""
    tag: Final = marker()
    response: Final = chat(rig, f"{SECRET} and {SECRET} {tag}", extra=_requested(PRE, PRE_TWO))
    assert response.status_code == 200, response.text
    calls: Final = rig.policy_calls()
    assert [rail for rail, _ in rails_called(calls)] == [PRE, PRE_TWO], calls
    assert SECRET not in str(calls[1]["texts"]), calls
    assert prompts_seen_by_provider(rig.provider_calls()) == (f"[MASKED] and [MASKED] {tag}",)


def test_a_during_call_guardrail_is_shown_the_request_and_lets_it_through(rig: Rig) -> None:
    """A request names the during_call guardrail and the peer answers NONE. The peer gets one call with
    `input_type` "request" and the caller gets the provider's answer."""
    response: Final = chat(rig, f"during {marker()}", extra=_requested(DURING))
    assert chat_text(response) == PLAIN_ANSWER
    assert rails_called(rig.policy_calls()) == ((DURING, "request"),)


def test_a_during_call_block_refuses_with_400_even_though_the_provider_was_called(rig: Rig) -> None:
    """The during_call guardrail blocks. It runs in parallel with the provider call, so the provider may already
    have been called (at most once), but the caller must get 400 with the peer's reason and never the answer."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra=_requested(DURING))
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_input(DURING)
    assert len(rig.provider_calls()) <= 1


def test_during_call_masking_does_not_change_what_the_caller_receives(rig: Rig) -> None:
    """The during_call guardrail answers GUARDRAIL_INTERVENED for a prompt carrying the secret. A during_call
    guardrail cannot modify the request, so the caller gets 200 with the provider's ordinary answer.
    UNVERIFIED: whether the provider saw the raw or the masked prompt depends on when the body was serialized;
    only the caller-visible result is pinned (read from code: proxy/utils.py during_call_hook returns `data`)."""
    response: Final = chat(rig, f"{SECRET} during {marker()}", extra=_requested(DURING))
    assert chat_text(response) == PLAIN_ANSWER


def test_a_post_call_guardrail_is_shown_the_answer_after_the_provider(rig: Rig) -> None:
    """A request names the post_call guardrail. The provider is called once, then the peer gets one call with
    `input_type` "response" whose texts contain the provider's answer; the request itself is never shown to it."""
    response: Final = chat(rig, f"post {marker()}", extra=_requested(POST))
    assert chat_text(response) == PLAIN_ANSWER
    calls: Final = rig.policy_calls()
    assert rails_called(calls) == ((POST, "response"),)
    assert PLAIN_ANSWER in str(calls[0]["texts"]), calls
    assert len(rig.provider_calls()) == 1


def test_a_post_call_block_replaces_the_answer_with_a_400(rig: Rig) -> None:
    """The provider answers with BLOCK-OUT inside and the post_call guardrail blocks it. The caller gets 400 whose
    message is "dec1 refused output by dec1-gr-post" and the blocked answer text appears nowhere in the body, while
    the provider was called exactly once."""
    response: Final = chat(rig, f"REPLY-BLOCK-OUT {marker()}", extra=_requested(POST))
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_output(POST)
    assert "BLOCK-OUT inside" not in response.text
    assert len(rig.provider_calls()) == 1


def test_post_call_masking_rewrites_the_answer_the_caller_receives(rig: Rig) -> None:
    """The provider answers "the code is SECRET-1234" and the post_call guardrail masks it. The caller must get
    "the code is [MASKED]" as the message content."""
    response: Final = chat(rig, f"REPLY-SECRET {marker()}", extra=_requested(POST))
    assert chat_text(response) == MASKED_SECRET_ANSWER


def test_a_list_mode_guardrail_runs_on_the_request_and_on_the_answer(rig: Rig) -> None:
    """A guardrail configured with mode [pre_call, post_call] is requested once. The peer must get exactly two calls
    from it, first with input_type "request" and then with input_type "response"."""
    response: Final = chat(rig, f"both {marker()}", extra=_requested(BOTH))
    assert response.status_code == 200, response.text
    assert rails_called(rig.policy_calls()) == ((BOTH, "request"), (BOTH, "response"))


def test_a_list_mode_guardrail_can_refuse_the_answer_after_letting_the_request_through(rig: Rig) -> None:
    """The list-mode guardrail passes the request and then blocks the BLOCK-OUT answer. The caller gets 400 with the
    output reason and the peer saw both directions."""
    response: Final = chat(rig, f"REPLY-BLOCK-OUT {marker()}", extra=_requested(BOTH))
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_output(BOTH)
    assert rails_called(rig.policy_calls()) == ((BOTH, "request"), (BOTH, "response"))


def test_a_guardrail_that_is_not_requested_and_not_default_on_never_runs(rig: Rig) -> None:
    """A plain request names no guardrail. Even with BLOCK-IN in the prompt the caller gets 200, the peer receives
    nothing and the response carries no x-litellm-applied-guardrails header."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}")
    assert chat_text(response) == PLAIN_ANSWER
    assert APPLIED_HEADER not in response.headers, dict(response.headers)
    assert rig.policy_calls() == ()


def test_a_guardrail_named_in_metadata_guardrails_runs_like_one_named_in_the_body(rig: Rig) -> None:
    """The request carries `metadata: {"guardrails": ["dec1-gr-pre"]}` instead of a top-level field. The guardrail
    must run and refuse a BLOCK-IN prompt with 400 (read from code: custom_guardrail.py get_guardrail_from_metadata)."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra={"metadata": {"guardrails": [PRE]}})
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_input(PRE)


def test_body_and_metadata_guardrail_lists_are_both_honoured(rig: Rig) -> None:
    """The request names one guardrail in the body and another in `metadata.guardrails`. Both must run (the body
    list is appended to the metadata list in move_guardrails_to_metadata), so the peer is called by both rails."""
    response: Final = chat(
        rig, f"merged {marker()}", extra={"guardrails": [PRE], "metadata": {"guardrails": [PRE_TWO]}}
    )
    assert response.status_code == 200, response.text
    assert sorted(rail for rail, _ in rails_called(rig.policy_calls())) == sorted([PRE, PRE_TWO])


def test_a_dict_entry_passes_its_extra_body_to_the_guardrail_service(rig: Rig) -> None:
    """The request names the guardrail as `{"dec1-gr-pre": {"extra_body": {"threshold": 0.7}}}`. The peer body's
    `additional_provider_specific_params` must hold both the configured `rail` and the per-request `threshold`
    (the suite proxy is licensed, so dynamic extra_body is honoured; read from code: custom_guardrail.py
    get_guardrail_dynamic_request_body_params)."""
    response: Final = chat(rig, f"extra {marker()}", extra={"guardrails": [{PRE: {"extra_body": {"threshold": 0.7}}}]})
    assert response.status_code == 200, response.text
    calls: Final = rig.policy_calls()
    assert len(calls) == 1, calls
    assert object_value(calls[0]["additional_provider_specific_params"]) == {"rail": PRE, "threshold": 0.7}


def test_the_applied_header_names_the_guardrail_that_ran(rig: Rig) -> None:
    """A request names one pre_call guardrail and is served. The response header x-litellm-applied-guardrails must
    be exactly "dec1-gr-pre"."""
    response: Final = chat(rig, f"header {marker()}", extra=_requested(PRE))
    assert response.status_code == 200, response.text
    assert response.headers[APPLIED_HEADER] == PRE


def test_the_applied_header_names_every_requested_guardrail_comma_separated(rig: Rig) -> None:
    """A request names two pre_call guardrails. The header must name both, joined by a bare comma and no space."""
    response: Final = chat(rig, f"header two {marker()}", extra=_requested(PRE, PRE_TWO))
    assert response.status_code == 200, response.text
    assert sorted(response.headers[APPLIED_HEADER].split(",")) == sorted([PRE, PRE_TWO])
    assert " " not in response.headers[APPLIED_HEADER]


def test_the_applied_header_names_a_post_call_guardrail_that_ran(rig: Rig) -> None:
    """A request names only the post_call guardrail. On success the header must still name it."""
    response: Final = chat(rig, f"header post {marker()}", extra=_requested(POST))
    assert response.status_code == 200, response.text
    assert POST in response.headers[APPLIED_HEADER].split(",")


def test_a_refusal_still_carries_the_applied_header_and_the_call_id(rig: Rig) -> None:
    """A pre_call block answers 400. That 400 must still carry x-litellm-applied-guardrails naming the guardrail
    that refused and a non-empty x-litellm-call-id, so callers can correlate refusals (read from code:
    proxy/utils.py `_record_raising_guardrail`, common_request_processing.py get_custom_headers)."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra=_requested(PRE))
    assert response.status_code == 400, response.text
    assert PRE in response.headers[APPLIED_HEADER].split(","), dict(response.headers)
    assert response.headers.get(CALL_ID_HEADER), dict(response.headers)


def test_a_request_naming_an_unconfigured_guardrail_is_served_without_naming_it(rig: Rig) -> None:
    """A request names a guardrail that does not exist on the proxy. It must be served with 200 (requests are not
    validated against the registry), and the applied header must not claim a guardrail ran that does not exist.
    Today `_process_guardrail_metadata` (proxy/utils.py) adds every requested string name to applied_guardrails."""
    pytest.skip("BUG: x-litellm-applied-guardrails names a requested guardrail that is not configured and never ran")
    response: Final = chat(rig, f"ghost {marker()}", extra=_requested(NEVER_CONFIGURED))
    assert chat_text(response) == PLAIN_ANSWER
    assert NEVER_CONFIGURED not in response.headers.get(APPLIED_HEADER, "").split(",")
