"""StandardLoggingPayload cache fields: what a logger sees for a cache miss, a cache hit and a cache bypass.

The rig turns on the Redis response cache (`litellm_settings.cache: true`) next to `callbacks: ["generic_api",
<custom recorder>]`. A caller sends the same chat request twice with two different `x-litellm-call-id` values.
The second one is served from the cache, so the provider sees one request, and its payload must say so: `cache_hit`
true, a `cache_key`, zero `response_cost`, and `saved_cache_cost` equal to what the original call cost (0.019 with
the rig's custom price). Spend reports and "money saved by caching" dashboards read exactly these fields.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import JsonValue, list_value, object_value
from integration.observability.dec1_callbacks_support import (
    CHAT_COST,
    CHAT_USAGE,
    EMBEDDING,
    EMBEDDING_COST,
    RECORDER,
    REPLY_TEXT,
    CallbackRig,
    callback_rig,
    chat,
    marker,
    new_call_id,
    post,
    provider_bodies,
)

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-payload-cache"),
        {
            "callbacks": ["generic_api", RECORDER],
            "cache": True,
            "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
        },
    ) as created:
        yield created


def _miss_then_hit(rig: CallbackRig, **extra: JsonValue) -> tuple[dict[str, JsonValue], dict[str, JsonValue], str]:
    """Send one unique chat request twice and return (first payload, second payload, the unique text)."""
    text: Final = marker()
    first, second = new_call_id(), new_call_id()
    assert chat(rig, first, text, **extra).status_code == 200
    miss: Final = rig.event(first)
    assert chat(rig, second, text, **extra).status_code == 200
    return miss, rig.event(second), text


def test_second_identical_request_is_logged_as_a_cache_hit(rig: CallbackRig) -> None:
    """The same chat request is sent twice. The provider receives it once. The first payload has `cache_hit` not
    true and the second has `cache_hit` true, both with status "success" and call_type "acompletion"."""
    miss, hit, text = _miss_then_hit(rig)
    assert len(tuple(body for body in provider_bodies(rig) if text in json.dumps(body))) == 1
    assert miss["cache_hit"] is not True
    assert (hit["cache_hit"], hit["status"], hit["call_type"]) == (True, "success", "acompletion")


def test_cache_hit_costs_nothing_and_reports_the_saved_cost(rig: CallbackRig) -> None:
    """The original call cost 0.019. The cache-hit payload must report `response_cost` 0 and `saved_cache_cost`
    0.019, while the first payload reports `response_cost` 0.019 and `saved_cache_cost` 0."""
    miss, hit, _ = _miss_then_hit(rig)
    assert (miss["response_cost"], miss["saved_cache_cost"]) == (pytest.approx(CHAT_COST), 0)
    assert (hit["response_cost"], hit["saved_cache_cost"]) == (0, pytest.approx(CHAT_COST))


def test_cache_hit_id_is_the_original_id_with_a_cache_hit_suffix(rig: CallbackRig) -> None:
    """A cache hit returns the stored response, so its provider id repeats. The cache-hit payload's `id` must be
    the original id followed by "_cache_hit" and a timestamp, so it never collides with the original request's
    row in LiteLLM_SpendLogs."""
    miss, hit, _ = _miss_then_hit(rig)
    original: Final = str(miss["id"])
    assert str(hit["id"]).startswith(original + "_cache_hit")
    assert float(str(hit["id"]).removeprefix(original + "_cache_hit")) > 1_600_000_000


def test_cache_key_is_set_on_the_hit_and_matches_hidden_params(rig: CallbackRig) -> None:
    """On the cache-hit payload `cache_key` must be a non-empty string and equal `hidden_params.cache_key`."""
    _, hit, _ = _miss_then_hit(rig)
    assert isinstance(hit["cache_key"], str) and hit["cache_key"]
    assert object_value(hit["hidden_params"])["cache_key"] == hit["cache_key"]


def test_cache_hit_payload_still_reports_the_cached_tokens_and_reply(rig: CallbackRig) -> None:
    """The cache-hit payload reports the token counts of the stored response (11, 4, 15) and logs the stored
    assistant text as its `response`, so usage dashboards keep counting tokens served from cache."""
    _, hit, _ = _miss_then_hit(rig)
    assert (hit["prompt_tokens"], hit["completion_tokens"], hit["total_tokens"]) == (*CHAT_USAGE, 15)
    choice: Final = object_value(list_value(object_value(hit["response"])["choices"])[0])
    assert object_value(choice["message"])["content"] == REPLY_TEXT


def test_each_cache_hit_gets_its_own_call_id_and_trace_id(rig: CallbackRig) -> None:
    """The miss and the hit are two caller requests, so their payloads must carry their own `litellm_call_id`
    (the two header values the caller sent) and two different `trace_id` values, even though the response body is
    the same."""
    miss, hit, _ = _miss_then_hit(rig)
    assert miss["litellm_call_id"] != hit["litellm_call_id"]
    assert miss["trace_id"] != hit["trace_id"]


def test_no_cache_request_is_never_logged_as_a_hit(rig: CallbackRig) -> None:
    """The same request is sent twice with `cache: {"no-cache": true}` in the body. The provider receives both, and
    neither payload has `cache_hit` true; both report the full `response_cost` 0.019."""
    miss, second, text = _miss_then_hit(rig, cache={"no-cache": True})
    assert len(tuple(body for body in provider_bodies(rig) if text in json.dumps(body))) == 2
    assert miss["cache_hit"] is not True and second["cache_hit"] is not True
    assert second["response_cost"] == pytest.approx(CHAT_COST)


def test_embedding_cache_hit_is_logged_with_zero_cost_and_saved_cost(rig: CallbackRig) -> None:
    """The same embedding request is sent twice. The second payload has call_type "aembedding", `cache_hit` true,
    `response_cost` 0 and `saved_cache_cost` 0.005 (5 tokens at 0.001). UNVERIFIED: saved_cache_cost for embeddings
    was inferred from the chat path in get_standard_logging_object_payload."""
    text: Final = marker()
    first, second = new_call_id(), new_call_id()
    body: Final[dict[str, JsonValue]] = {"model": EMBEDDING, "input": text}
    assert post(rig, "/v1/embeddings", body, first).status_code == 200
    rig.event(first)
    assert post(rig, "/v1/embeddings", body, second).status_code == 200
    hit: Final = rig.event(second)
    assert (hit["call_type"], hit["cache_hit"], hit["response_cost"]) == ("aembedding", True, 0)
    assert hit["saved_cache_cost"] == pytest.approx(EMBEDDING_COST)


def test_custom_logger_sees_the_cache_hit_as_a_success_event_only(rig: CallbackRig) -> None:
    """For the cached second request the custom callback must get exactly one `async_log_success_event` and no
    failure event, and its standard_logging_object must have `cache_hit` true. The provider-facing hooks
    (`log_pre_api_call`, `async_pre_call_deployment_hook`) are not asserted here, only the log event."""
    text: Final = marker()
    first, second = new_call_id(), new_call_id()
    assert chat(rig, first, text).status_code == 200
    rig.event(first)
    assert chat(rig, second, text).status_code == 200
    record: Final = rig.hook(second, "async_log_success_event")
    assert object_value(record["standard"])["cache_hit"] is True
    names: Final = rig.hook_names(second)
    assert names.count("async_log_success_event") == 1 and "async_log_failure_event" not in names
