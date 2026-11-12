import json

import httpx
import pytest
from openai import AsyncOpenAI

from litellm.task_insights.demo_traffic import (
    DISCOVERY_PROMPTS,
    HELDOUT_PROMPTS,
    agreement,
    batch_prompts,
    export_heldout,
    read_labels,
    score_report,
    send,
)


def test_agreement_counts_unknown_labels_and_ignores_unlabeled_prompts():
    labels = {"a": "Debug", "b": "Docs", "c": "unknown", "d": ""}
    assigned = {"a": "Debug", "b": "Debug", "c": None, "d": "Debug"}

    assert agreement(labels, assigned) == (3, 2, 2)


def test_batches_have_stable_distinct_ids_and_no_overlap_between_discovery_and_heldout():
    discovery = batch_prompts("discovery")
    heldout = batch_prompts("heldout")

    assert batch_prompts("discovery") == discovery
    assert len({request_id for request_id, _ in discovery + heldout}) == len(discovery) + len(heldout)
    assert not {prompt for _, prompt in HELDOUT_PROMPTS} & set(DISCOVERY_PROMPTS)


@pytest.mark.asyncio
async def test_send_gives_every_request_its_own_session_and_the_configured_model():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "claude-haiku",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    client = AsyncOpenAI(
        base_url="http://gateway/v1",
        api_key="sk-test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )

    sent = await send(client, "heldout", model="claude-haiku", max_tokens=16)

    assert sent == len(HELDOUT_PROMPTS) == len(seen)
    bodies = [json.loads(request.content) for request in seen]
    assert {(body["model"], body["max_tokens"]) for body in bodies} == {("claude-haiku", 16)}
    assert [body["messages"][0]["content"] for body in bodies] == [prompt for _, prompt in HELDOUT_PROMPTS]
    assert len({request.headers["x-litellm-session-id"] for request in seen}) == len(seen)


def test_the_exported_sheet_has_no_answer_column_and_reads_back_as_labels(tmp_path):
    path = tmp_path / "heldout.csv"

    export_heldout(str(path))

    assert path.read_text().splitlines()[0] == "prompt,label"
    labels = read_labels(str(path))
    assert set(labels) == {prompt for _, prompt in HELDOUT_PROMPTS}
    assert set(labels.values()) == {""}


def test_the_score_report_states_coverage_and_agreement_over_labeled_prompts():
    labels = {"a": "Debug", "b": "unknown", "c": ""}

    assert score_report(labels, {"a": "Debug", "b": "Docs"}) == "labeled=2 assigned=2 coverage=1.00 agreement=0.50"
    assert score_report({}, {}) == "labeled=0 assigned=0 coverage=0.00 agreement=0.00"
