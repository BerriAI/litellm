"""The `LiteLLM_SpendLogs` row written for a `POST /v1/messages` call.

Every call writes one row, asynchronously, keyed by the provider's message id (`request_id ==
response.id`). For a native Anthropic call the row records `call_type = "anthropic_messages"`,
`custom_llm_provider = "anthropic"`, the proxy model name in `model_group`, the deployment id in
`model_id`, the token counts from the provider's `usage`, the priced `spend`, the SHA-256 of the
calling key in `api_key`, and the caller's attribution: tags from `litellm_metadata.tags` or the
`x-litellm-tags` header, an end user from `litellm_metadata.user` or Anthropic's
`metadata.user_id`, and a session id from an `x-litellm-session-id` header or a Claude Code style
`metadata.user_id` ending in `_session_<id>`. A failed call writes a row with `status = "failure"`
and the error code and class under `metadata.error_information`.

Spend logging and DB access are both being moved to Rust, and billing is read from these rows, so
each column is pinned by its own test. Rows land asynchronously: every read polls with
`eventually`. Tags and basic usage for one happy call are also pinned by
`providers/anthropic/test_passthrough_migration_wire.py`; this file covers the remaining columns.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, object_value, string_value
from integration.messages_endpoint.dec1_messages_support import (
    BACKEND,
    CACHE_CREATION_COST,
    CACHE_READ_COST,
    INPUT_COST,
    NO_CACHE,
    OUTPUT_COST,
    PRICES,
    Rig,
    always,
    caller_tags,
    deployment,
    json_column,
    marker,
    message_id_of,
    message_reply,
    spend_row,
    spend_rows_for_model,
    stream_reply,
    upstream_error,
)
from pydantic import JsonValue


def _deployment_id(rig: Rig) -> str:
    entries: Final = rig.gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    matches: Final = [
        object_value(object_value(entry)["model_info"])["id"]
        for entry in entries
        if object_value(entry)["model_name"] == rig.model
    ]
    assert len(matches) == 1, matches
    return string_value(matches[0])


def _error_information(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(object_value(json_column(row["metadata"]))["error_information"])


def test_a_successful_call_writes_one_row_keyed_by_the_message_id(gateway: Gateway) -> None:
    """After one successful call the table must hold exactly one row whose `request_id` is the
    `msg_...` id the caller received, with `status = "success"` and `call_type = "anthropic_messages"`."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        assert message_id_of(rig.post(rig.body())) == message_id
        row: Final = spend_row(message_id)
        assert (row["status"], row["call_type"]) == ("success", "anthropic_messages")
        assert len(spend_rows_for_model(rig.model, 1)) == 1


def test_the_row_records_the_provider_token_counts(gateway: Gateway) -> None:
    """The provider reports 11 input and 7 output tokens. The row must record `prompt_tokens = 11`,
    `completion_tokens = 7` and `total_tokens = 18`."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        rig.post(rig.body())
        row: Final = spend_row(message_id)
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (11, 7, 18)


def test_the_row_spend_is_the_priced_cost_and_matches_the_response_header(gateway: Gateway) -> None:
    """At 0.001 per input token and 0.002 per output token, 11 input and 7 output tokens cost 0.025.
    The row's `spend` must be 0.025 and equal the `x-litellm-response-cost` header the caller got."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        response: Final = rig.post(rig.body())
        row: Final = spend_row(message_id)
        assert float(str(row["spend"])) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)
        assert float(str(row["spend"])) == pytest.approx(float(response.headers["x-litellm-response-cost"]))


def test_the_row_spend_prices_cache_read_and_cache_creation_tokens(gateway: Gateway) -> None:
    """With 10 uncached input, 4 output, 5 cache-read and 9 cache-creation tokens at the four
    configured rates the row's `spend` must be
    10 * 0.001 + 4 * 0.002 + 5 * 0.0001 + 9 * 0.003 = 0.0455 and `completion_tokens` must be 4."""
    message_id: Final = f"msg_{marker()}"
    usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 9,
    }
    with deployment(gateway, always(message_reply(message_id, usage=usage)), **PRICES) as rig:
        rig.post(rig.body())
        row: Final = spend_row(message_id)
        assert float(str(row["spend"])) == pytest.approx(
            10 * INPUT_COST + 4 * OUTPUT_COST + 5 * CACHE_READ_COST + 9 * CACHE_CREATION_COST
        )
        assert row["completion_tokens"] == 4


def test_the_row_names_the_model_group_deployment_and_provider(gateway: Gateway) -> None:
    """The row must record the proxy model name the caller used in `model_group`, the deployment's
    `model_info.id` in `model_id`, and `custom_llm_provider = "anthropic"`. UNVERIFIED: `model` is
    expected to contain the backend name `claude-sonnet-4-5-20250929` (with or without the
    `anthropic/` prefix), asserted as a substring."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        rig.post(rig.body())
        row: Final = spend_row(message_id)
        assert row["model_group"] == rig.model
        assert row["model_id"] == _deployment_id(rig)
        assert row["custom_llm_provider"] == "anthropic"
        assert BACKEND in str(row["model"])


def test_the_row_records_the_deployment_api_base(gateway: Gateway) -> None:
    """The row's `api_base` must point at the deployment's provider host. UNVERIFIED: it is expected
    to start with the deployment's configured `api_base` (it may carry the `/v1/messages` path)."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        rig.post(rig.body())
        assert str(spend_row(message_id)["api_base"]).startswith(rig.wire.url)


def test_the_row_api_key_is_the_sha256_of_the_calling_key_whichever_header_carried_it(gateway: Gateway) -> None:
    """Two calls are made with the same virtual key, one as `Authorization: Bearer`, one as
    `x-api-key`. Both rows must store the SHA-256 hex digest of the key in `api_key`, never the key
    itself, so spend is attributed to the same key for both header forms."""
    first: Final = f"msg_{marker()}"
    second: Final = f"msg_{marker()}"
    replies: Final = iter((message_reply(first), message_reply(second)))
    with deployment(gateway, lambda _request: next(replies), **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        hashed: Final = sha256(key.encode()).hexdigest()
        assert rig.post(rig.body(), key=key).status_code == 200
        assert rig.gateway.client.post("/v1/messages", json=rig.body(), headers={"x-api-key": key}).status_code == 200
        assert spend_row(first)["api_key"] == hashed
        assert spend_row(second)["api_key"] == hashed


def test_the_row_records_the_key_owner_and_team(gateway: Gateway) -> None:
    """A virtual key is created for an internal user inside a team. A call with that key must write
    a row whose `user` is the key's `user_id` and whose `team_id` is the key's team."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        team_id: Final = rig.scenario.team()
        user_id: Final = rig.scenario.user()
        key: Final = rig.scenario.key(models=[rig.model], team_id=team_id, user_id=user_id)
        assert rig.post(rig.body(), key=key).status_code == 200
        row: Final = spend_row(message_id)
        assert (row["user"], row["team_id"]) == (user_id, team_id)


def test_anthropic_metadata_user_id_becomes_the_row_end_user(gateway: Gateway) -> None:
    """Anthropic clients identify their end user with `metadata.user_id`. With no `litellm_metadata`
    and no customer header, the row's `end_user` must be that value
    (`auth_utils.py` `get_end_user_id_from_request_body`, check 5)."""
    message_id: Final = f"msg_{marker()}"
    end_user: Final = f"synthetic-end-user-{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        assert rig.post(rig.body(metadata={"user_id": end_user})).status_code == 200
        assert spend_row(message_id)["end_user"] == end_user


def test_litellm_metadata_user_wins_over_metadata_user_id_for_the_end_user(gateway: Gateway) -> None:
    """When both `litellm_metadata.user` and `metadata.user_id` are sent, the row's `end_user` must be
    the `litellm_metadata.user` value: it is checked first. The provider still receives the Anthropic
    `metadata.user_id`."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        body: Final = rig.body(
            metadata={"user_id": "anthropic-side-user"}, litellm_metadata={"user": "proxy-side-user"}
        )
        assert rig.post(body).status_code == 200
        assert spend_row(message_id)["end_user"] == "proxy-side-user"
        assert rig.one_body()["metadata"] == {"user_id": "anthropic-side-user"}


def test_the_x_litellm_tags_header_becomes_request_tags(gateway: Gateway) -> None:
    """A caller that cannot change the body sends `x-litellm-tags: <a>,<b>`. The row's `request_tags`
    must contain both tags (the proxy may add its own `User-Agent` tags, which are ignored here)."""
    message_id: Final = f"msg_{marker()}"
    first, second = marker("tag"), marker("tag")
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        assert rig.post(rig.body(), headers={"x-litellm-tags": f"{first},{second}"}).status_code == 200
        assert sorted(str(tag) for tag in caller_tags(spend_row(message_id))) == sorted([first, second])


def test_a_claude_code_style_user_id_sets_the_row_session_id(gateway: Gateway) -> None:
    """Claude Code sends `metadata.user_id` shaped `user_<hash>_account_<id>_session_<uuid>`. With no
    session header, the row's `session_id` must be the part after the last `_session_`
    (`litellm_pre_call_utils.py` `_get_anthropic_session_id_from_metadata`)."""
    message_id: Final = f"msg_{marker()}"
    session_id: Final = "0f9c1b1e-7a55-4c1d-9f36-3d2a6f7b8c90"
    user_id: Final = f"user_synthetic_account_synthetic_session_{session_id}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        assert rig.post(rig.body(metadata={"user_id": user_id})).status_code == 200
        assert spend_row(message_id)["session_id"] == session_id


def test_the_x_litellm_session_id_header_sets_the_row_session_id(gateway: Gateway) -> None:
    """An explicit `x-litellm-session-id` header must be stored as the row's `session_id`, and it
    wins over a session embedded in `metadata.user_id`."""
    message_id: Final = f"msg_{marker()}"
    session_id: Final = marker("session")
    embedded: Final = "user_synthetic_account_synthetic_session_0f9c1b1e-7a55-4c1d-9f36-3d2a6f7b8c90"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        response: Final = rig.post(
            rig.body(metadata={"user_id": embedded}), headers={"x-litellm-session-id": session_id}
        )
        assert response.status_code == 200, response.text
        assert spend_row(message_id)["session_id"] == session_id


def test_the_row_litellm_call_id_matches_the_response_header(gateway: Gateway) -> None:
    """The row's `litellm_call_id` column must equal the `x-litellm-call-id` response header of the
    same call, which is how support joins a caller's report to its spend row."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert spend_row(message_id)["litellm_call_id"] == response.headers["x-litellm-call-id"]


def test_the_row_times_are_ordered_and_it_is_not_a_cache_hit(gateway: Gateway) -> None:
    """A fresh provider call must write a row whose `startTime` is not after `endTime` and whose
    `cache_hit` is not `True`."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        rig.post(rig.body())
        row: Final = spend_row(message_id)
        assert row["times_ordered"] is True
        assert str(row["cache_hit"]).lower() != "true"


def test_a_streamed_call_writes_the_same_row_from_the_stream_usage(gateway: Gateway) -> None:
    """A `stream: true` call whose stream reports 23 input tokens in `message_start` and 17 output
    tokens in `message_delta` must write one success row keyed by the streamed message id with
    `prompt_tokens = 23`, `completion_tokens = 17`, `total_tokens = 40`, spend
    23 * 0.001 + 17 * 0.002 = 0.057 and `call_type = "anthropic_messages"`."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(stream_reply(message_id, input_tokens=23, output_tokens=17)), **PRICES) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        row: Final = spend_row(message_id)
        assert (row["status"], row["call_type"]) == ("success", "anthropic_messages")
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (23, 17, 40)
        assert float(str(row["spend"])) == pytest.approx(23 * INPUT_COST + 17 * OUTPUT_COST)


def test_two_calls_write_two_rows(gateway: Gateway) -> None:
    """Two calls with different prompts on one deployment must write exactly two rows with the two
    distinct message ids. Neither call may be dropped or written twice."""
    first: Final = f"msg_{marker()}"
    second: Final = f"msg_{marker()}"
    replies: Final = iter((message_reply(first), message_reply(second)))
    with deployment(gateway, lambda _request: next(replies), **PRICES) as rig:
        rig.post(rig.body())
        rig.post(rig.body())
        spend_row(first)
        spend_row(second)
        assert sorted(str(row["request_id"]) for row in spend_rows_for_model(rig.model, 2)) == sorted([first, second])


@pytest.mark.parametrize(
    ("status", "error_type", "error_class"),
    (
        (400, "invalid_request_error", "BadRequestError"),
        (429, "rate_limit_error", "RateLimitError"),
        (500, "api_error", "InternalServerError"),
    ),
)
def test_a_provider_failure_writes_a_failure_row_with_the_error_code_and_class(
    gateway: Gateway, status: int, error_type: str, error_class: str
) -> None:
    """The provider rejects the call with 400, 429 or 500. One row must be written for the model
    group with `status = "failure"`, `metadata.error_information.error_code` equal to the status as
    a string, `error_class` equal to litellm's exception class for it, and zero spend."""
    with deployment(gateway, always(upstream_error(status, error_type, "synthetic refusal")), **PRICES) as rig:
        assert rig.post(rig.body()).status_code == status
        rows: Final = spend_rows_for_model(rig.model, 1)
        assert len(rows) == 1, rows
        assert rows[0]["status"] == "failure"
        information: Final = _error_information(rows[0])
        assert (information["error_code"], information["error_class"]) == (str(status), error_class)
        assert float(str(rows[0]["spend"])) == 0.0


def test_a_failure_row_keeps_the_call_type_and_caller_key(gateway: Gateway) -> None:
    """A failed call made with a virtual key must still be attributed: the failure row must have
    `call_type = "anthropic_messages"`, the SHA-256 of the key in `api_key`, and the proxy model name
    in `model_group`."""
    with deployment(gateway, always(upstream_error(400, "invalid_request_error", "synthetic refusal"))) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        assert rig.post(rig.body(), key=key).status_code == 400
        row: Final = spend_rows_for_model(rig.model, 1)[0]
        assert row["call_type"] == "anthropic_messages"
        assert row["api_key"] == sha256(key.encode()).hexdigest()
        assert row["model_group"] == rig.model


@pytest.mark.parametrize("uncached", (False, True), ids=("cache_default", "no_cache"))
def test_the_row_is_the_same_with_and_without_the_response_cache(gateway: Gateway, uncached: bool) -> None:
    """A call on the default path and one sent with `cache: {"no-cache": true}` (the shape eligible
    for the Rust route) must write the same row: success, `anthropic_messages`, provider `anthropic`,
    11/7/18 tokens, spend 0.025, the proxy model name as `model_group`, and the caller's tag."""
    message_id: Final = f"msg_{marker()}"
    tag: Final = marker("tag")
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        body: Final = rig.body(litellm_metadata={"tags": [tag]}, **(NO_CACHE if uncached else {}))
        assert rig.post(body).status_code == 200
        row: Final = spend_row(message_id)
        assert (row["status"], row["call_type"], row["custom_llm_provider"]) == (
            "success",
            "anthropic_messages",
            "anthropic",
        )
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (11, 7, 18)
        assert float(str(row["spend"])) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)
        assert row["model_group"] == rig.model
        assert caller_tags(row) == [tag]
