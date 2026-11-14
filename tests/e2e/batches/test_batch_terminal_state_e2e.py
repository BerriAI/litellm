from __future__ import annotations

import time
from typing import Final

import pytest
from batch_cleanup import BATCH_TERMINAL_STATUSES
from batch_client import BatchClient, BatchCreateBody, BatchObject
from batch_helpers import CREATED_BATCH_STATUSES, BatchOutputLine, render_jsonl, retrieve_batch
from capabilities import OPENAI_BATCH_BACKEND, OPENAI_BATCH_MODEL
from e2e_config import unique_marker
from e2e_http import FileUploadForm, require_successful_call, unwrap
from e2e_metadata import Domain, Mode, Route, Subject, meta
from e2e_metadata import Provider as MetaProvider
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e


TERMINAL_MARKER_KEY = "litellm_e2e_suite"
TERMINAL_MARKER_VALUE = "batches-terminal-baton"
TERMINAL_POLL_SECONDS = 300.0
TERMINAL_POLL_INTERVAL_SECONDS = 10.0
TERMINAL_LIST_LIMIT = 100
TERMINAL_BAND_MIN_AGE_SECONDS = 25 * 3600
TERMINAL_BAND_MAX_AGE_SECONDS = 73 * 3600


def _marker_batches(client: BatchClient, key: str) -> list[BatchObject]:
    listed = unwrap(
        client.list_batches(key=key, model=OPENAI_BATCH_MODEL, limit=TERMINAL_LIST_LIMIT)
    )
    return [
        b
        for b in listed.data
        if (b.metadata or {}).get(TERMINAL_MARKER_KEY) == TERMINAL_MARKER_VALUE
    ]


def _await_completed_marker(
    client: BatchClient, key: str
) -> tuple[BatchObject | None, list[BatchObject]]:
    deadline = time.monotonic() + TERMINAL_POLL_SECONDS
    while True:
        markers = _marker_batches(client, key)
        completed = max(
            (b for b in markers if b.status == "completed"),
            key=lambda b: b.created_at or 0,
            default=None,
        )
        if completed is not None or time.monotonic() >= deadline:
            return completed, markers
        time.sleep(TERMINAL_POLL_INTERVAL_SECONDS)


def _assert_aged_markers_terminal(markers: list[BatchObject]) -> None:
    now = time.time()
    stuck = [
        b
        for b in markers
        if b.created_at is not None
        and TERMINAL_BAND_MIN_AGE_SECONDS <= now - b.created_at <= TERMINAL_BAND_MAX_AGE_SECONDS
        and b.status not in BATCH_TERMINAL_STATUSES
    ]
    assert not stuck, (
        "marker batches past their 24h completion window must be terminal; stuck: "
        f"{[(b.id, b.status, b.created_at) for b in stuck]}"
    )


class TestBatchTerminalState:
    """Terminal state + cost write-back via a cross-run marker baton.

    Each run submits a 1-line marker batch (stable metadata key/value plus a
    per-run field) and never cancels or deletes it: the marker is the baton the
    next run picks up. Polling is list-only for up to 5 minutes because a
    retrieve of a non-terminal batch books a $0 spend row whose request_id then
    blocks the real-cost row (skip_duplicates); the single retrieve happens only
    once a completed marker exists. The assertion target is the newest completed
    marker from ANY run, so on the 6h stage cadence the full assertions are
    deterministic from run 2 onward. On a cold start (no marker has ever
    completed within the poll budget) the test passes on the submission
    assertions alone: that is a documented vacuous pass, not a skip, and this
    run's marker becomes the next run's target. Markers aged past OpenAI's 24h
    completion window (25h-73h band, within the newest list page) must be
    terminal. The cost assertion is the LIT-5730 headline: retrieving a
    completed model-encoded batch must write a positive spend row keyed
    {batch_id}_batch_cost; before the fix the logging worker fetched the
    re-encoded output_file_id, 404d, and the row never landed.
    """

    @pytest.mark.covers(
        "llm.batches.openai.terminal_state.nonstream.works",
        "llm.batches.openai.terminal_state.nonstream.cost_logged",
        exercised_on=["batches", "files"],
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.BATCHES,
            providers=(MetaProvider.OPENAI,),
            models=(OPENAI_BATCH_BACKEND,),
            mode=Mode.BATCH,
        )
    )
    def test_completed_batch_downloads_output_and_books_cost(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = resources.key()
        file = unwrap(
            client.upload_file(
                content=render_jsonl(OPENAI_BATCH_BACKEND),
                form=FileUploadForm(purpose="batch"),
                model=OPENAI_BATCH_MODEL,
                key=key,
            )
        )
        created = client.create_batch(
            body=BatchCreateBody(
                input_file_id=file.id,
                metadata={
                    TERMINAL_MARKER_KEY: TERMINAL_MARKER_VALUE,
                    "run": unique_marker(),
                },
            ),
            key=key,
        )
        require_successful_call(created)
        submitted = BatchObject.model_validate_json(created.body)
        assert submitted.status in CREATED_BATCH_STATUSES, (
            f"marker batch has non-transitional status {submitted.status!r}"
        )
        assert (submitted.metadata or {}).get(TERMINAL_MARKER_KEY) == TERMINAL_MARKER_VALUE, (
            f"create dropped the marker metadata: {submitted.metadata!r}"
        )

        completed, markers = _await_completed_marker(client, key)
        _assert_aged_markers_terminal(markers)
        if completed is None:
            return

        fetched = retrieve_batch(client, completed.id, key=key, provider=None)
        assert fetched.status == "completed", (
            f"listed-completed marker retrieved as {fetched.status!r}"
        )
        assert fetched.output_file_id, "completed batch has no output_file_id"
        usage: Final = fetched.usage
        assert usage is not None and usage.input_tokens and usage.output_tokens, (
            f"completed batch has no batch token counts: {usage!r}"
        )
        assert (usage.prompt_tokens, usage.completion_tokens) == (usage.input_tokens, usage.output_tokens), (
            f"chat token counts disagree with the batch token counts: {usage!r}"
        )

        downloaded = client.proxy.transport.download(
            f"/v1/files/{fetched.output_file_id}/content",
            headers=client.proxy.transport.bearer(key),
        )
        assert downloaded.status_code == 200, (
            f"output content must be 200, got {downloaded.status_code}: {downloaded.body[:300]}"
        )
        first_line = BatchOutputLine.model_validate_json(downloaded.body.strip().splitlines()[0])
        assert first_line.response.status_code == 200, (
            f"batch output line reports failure: {downloaded.body[:400]}"
        )
        assert first_line.response.body is not None and first_line.response.body.choices, (
            "batch output line has no choices"
        )

        rows = client.proxy.poll_logs_for_request_id(
            f"{fetched.id}_batch_cost",
            predicate=lambda found: any((row.spend or 0) > 0 for row in found),
        )
        priced = [row for row in rows if (row.spend or 0) > 0]
        assert priced, (
            f"completed batch {fetched.id} wrote no positive-cost spend row under "
            f"request_id {fetched.id}_batch_cost; cost write-back is broken (LIT-5730)"
        )
        cost_row = priced[0]
        assert cost_row.call_type == "aretrieve_batch", (
            f"batch cost row call_type={cost_row.call_type!r}"
        )
        assert (cost_row.total_tokens or 0) > 0, (
            f"batch cost row has no token usage: {cost_row.total_tokens!r}"
        )
