"""Hop merging for Interactions API long decode."""

from typing import Final

import httpx
import pytest

from litellm.interactions.long_decode import (
    collect_hops_async,
    collect_hops_sync,
    continuation_token_of,
    hop_request,
    merge_hops,
    sum_usage,
)

_REQUEST: Final = {
    "model": "gemini-4-argon",
    "input": "Prove the prime number theorem.",
    "generation_config": {"max_output_tokens": 120000, "thinking_level": "high"},
    "tools": [{"type": "google_search"}],
}


def _hop(hop_id: str, status: str, text: str, token: str | None = None, **usage: int) -> dict[str, object]:
    return {
        "id": hop_id,
        "object": "interaction",
        "model": "gemini-4-argon",
        "status": status,
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": {
            "total_input_tokens": usage.get("input", 0),
            "input_tokens_by_modality": [{"modality": "text", "tokens": usage.get("input", 0)}],
            "total_output_tokens": usage.get("output", 0),
            "output_tokens_by_modality": [{"modality": "text", "tokens": usage.get("output", 0)}],
            "total_thought_tokens": usage.get("thought", 0),
            "total_tokens": usage.get("input", 0) + usage.get("output", 0) + usage.get("thought", 0),
            "initial_input_tokens": usage.get("input", 0),
        },
        "continuation_token": token,
    }


def test_only_a_requires_continuation_hop_has_a_token_to_continue() -> None:
    assert continuation_token_of(_hop("h1", "requires_continuation", "a", token="tok-1")) == "tok-1"
    assert continuation_token_of(_hop("h1", "completed", "a", token="tok-1")) is None
    assert continuation_token_of(_hop("h1", "incomplete", "a", token="tok-1")) is None
    assert continuation_token_of(_hop("h1", "requires_continuation", "a", token=None)) is None


def test_a_stored_hop_is_chained_by_id_instead_of_resending_the_input() -> None:
    assert hop_request(_REQUEST, _hop("h1", "requires_continuation", "a", token="tok-1"), "tok-1") == {
        "model": "gemini-4-argon",
        "generation_config": {"max_output_tokens": 120000, "thinking_level": "high"},
        "tools": [{"type": "google_search"}],
        "previous_interaction_id": "h1",
        "continuation_token": "tok-1",
    }


def test_an_unstored_hop_resends_the_input_with_the_token() -> None:
    assert hop_request({**_REQUEST, "store": False}, _hop("h1", "requires_continuation", "a", "tok-1"), "tok-1") == {
        **_REQUEST,
        "store": False,
        "continuation_token": "tok-1",
    }


def test_merged_usage_sums_every_hop_and_keeps_the_first_prompt_size() -> None:
    merged: Final = sum_usage(
        (
            _hop("h1", "requires_continuation", "a", "tok-1", input=79, output=26076, thought=6692)["usage"],
            _hop("h2", "completed", "b", input=32847, output=4000, thought=500)["usage"],
        )
    )
    assert merged == {
        "total_input_tokens": 32926,
        "input_tokens_by_modality": [{"modality": "text", "tokens": 32926}],
        "total_output_tokens": 30076,
        "output_tokens_by_modality": [{"modality": "text", "tokens": 30076}],
        "total_thought_tokens": 7192,
        "total_tokens": 70194,
        "initial_input_tokens": 79,
    }


def test_merged_interaction_has_the_last_id_and_status_and_every_hops_steps() -> None:
    merged: Final = merge_hops(
        (
            _hop("h1", "requires_continuation", "Lemma 1.", "tok-1", input=79, output=100),
            _hop("h2", "requires_continuation", " Lemma 2.", "tok-2", input=179, output=100),
            _hop("h3", "completed", " QED.", input=279, output=10),
        )
    )
    assert merged["id"] == "h3"
    assert merged["status"] == "completed"
    assert merged["continuation_token"] is None
    assert [content["text"] for step in merged["steps"] for content in step["content"]] == [
        "Lemma 1.",
        " Lemma 2.",
        " QED.",
    ]
    assert merged["usage"]["total_output_tokens"] == 210
    assert merged["usage"]["initial_input_tokens"] == 79


def test_a_single_hop_is_returned_as_is() -> None:
    hop: Final = _hop("h1", "completed", "done", input=5, output=3)
    assert merge_hops((hop,)) == hop


def _response(body: dict[str, object], status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=body, request=httpx.Request("POST", "https://example.test/interactions"))


def test_sync_hops_follow_the_token_until_the_decode_finishes() -> None:
    sent: list[dict[str, object]] = []  # mutable-ok: records the hop bodies the loop posts
    replies: Final = iter(
        (_hop("h2", "requires_continuation", "b", "tok-2", input=10, output=5), _hop("h3", "completed", "c", output=1))
    )

    def post(body: dict[str, object]) -> httpx.Response:
        sent.append(body)
        return _response(next(replies))

    hops: Final = collect_hops_sync(
        _hop("h1", "requires_continuation", "a", "tok-1", input=1, output=5), _REQUEST, post
    )

    assert [hop["id"] for hop in hops] == ["h1", "h2", "h3"]
    assert [(body["previous_interaction_id"], body["continuation_token"]) for body in sent] == [
        ("h1", "tok-1"),
        ("h2", "tok-2"),
    ]


def test_a_failed_hop_returns_the_answer_so_far_with_its_token() -> None:
    def post(body: dict[str, object]) -> httpx.Response:
        raise httpx.HTTPStatusError("boom", request=httpx.Request("POST", "https://x"), response=httpx.Response(500))

    hops: Final = collect_hops_sync(_hop("h1", "requires_continuation", "a", "tok-1"), _REQUEST, post)

    assert len(hops) == 1
    assert merge_hops(hops)["continuation_token"] == "tok-1"
    assert merge_hops(hops)["status"] == "requires_continuation"


def test_a_hop_that_repeats_its_token_stops_the_loop() -> None:
    calls: list[int] = []  # mutable-ok: counts the hop posts

    def post(body: dict[str, object]) -> httpx.Response:
        calls.append(1)
        return _response(_hop("h2", "requires_continuation", "b", "tok-1"))

    hops: Final = collect_hops_sync(_hop("h1", "requires_continuation", "a", "tok-1"), _REQUEST, post)

    assert len(calls) == 1
    assert len(hops) == 2


@pytest.mark.asyncio
async def test_async_hops_follow_the_token_until_the_decode_finishes() -> None:
    sent: list[dict[str, object]] = []  # mutable-ok: records the hop bodies the loop posts

    async def post(body: dict[str, object]) -> httpx.Response:
        sent.append(body)
        return _response(_hop("h2", "completed", "b", output=1))

    hops: Final = await collect_hops_async(_hop("h1", "requires_continuation", "a", "tok-1"), _REQUEST, post)

    assert [hop["id"] for hop in hops] == ["h1", "h2"]
    assert sent == [hop_request(_REQUEST, hops[0], "tok-1")]
