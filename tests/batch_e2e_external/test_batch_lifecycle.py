"""
End-to-end batch lifecycle against a LIVE, published LiteLLM image.

Real OpenAI SDK -> real proxy -> real providers. No mocks, no VCR.
Opt-in only:  pytest -m batch_e2e tests/batch_e2e_external

For each configured case (routing strategy x provider) the suite walks the full
lifecycle: create file -> create batch -> retrieve -> list -> cancel -> delete.

Hard-fail contract:
  * A supported op MUST succeed. Any failure -- including a missing DB for the
    managed-files (target_model_names) scenario -- fails the test. We catch
    regressions here, at the client boundary, rather than letting them ship.
  * An op declared unsupported for a provider (config.expected_unsupported) MUST
    raise the declared error. Success there, or a different error, is a failure.
"""

import os

import pytest

from .config import (
    OP_CANCEL,
    OP_CREATE,
    OP_DELETE,
    OP_FILE,
    OP_LIST,
    OP_RETRIEVE,
    Case,
    Settings,
)
from .helpers import assert_expected_error, build_input_jsonl, poll_until_terminal
from .strategies import build_strategy

pytestmark = pytest.mark.batch_e2e


def _run(case: Case, op: str, supported_call, unsupported_call=None):
    """Run an op, honouring the declared support contract for this provider.

    Returns the call's result when the op is supported, else asserts the
    declared error was raised and returns None.
    """
    expected = case.expected_unsupported.get(op)
    if expected is None:
        return supported_call()
    assert_expected_error(unsupported_call or supported_call, expected)
    return None


def test_batch_lifecycle(case: Case, client, settings: Settings):
    strat = build_strategy(case, settings.endpoint)
    input_path = build_input_jsonl(case.request_model, settings.endpoint)

    file_id = None
    batch_id = None
    try:
        # 1. create file (purpose="batch") --------------------------------
        file_obj = _run(case, OP_FILE, lambda: strat.create_file(client, input_path))
        if file_obj is None:
            return  # file creation is the declared contract gap; nothing downstream
        assert file_obj.id, "file create returned no id"
        file_id = file_obj.id

        # 2. create batch -------------------------------------------------
        batch = _run(case, OP_CREATE, lambda: strat.create_batch(client, file_id))
        if batch is None:
            return
        assert batch.id, "batch create returned no id"
        batch_id = batch.id

        # 3. retrieve batch -----------------------------------------------
        retrieved = _run(
            case, OP_RETRIEVE, lambda: strat.retrieve_batch(client, batch_id)
        )
        if retrieved is not None:
            assert retrieved.id == batch_id

        # 4. list batches -------------------------------------------------
        listing = _run(case, OP_LIST, lambda: strat.list_batches(client))
        if listing is not None:
            assert any(
                b.id == batch_id for b in listing.data
            ), f"created batch {batch_id} not present in list response"

        # 5. (optional) await completion + fetch output -------------------
        if settings.await_completion and OP_RETRIEVE not in case.expected_unsupported:
            done = poll_until_terminal(
                lambda: strat.retrieve_batch(client, batch_id),
                settings.completion_timeout_s,
            )
            if done.status == "completed" and getattr(done, "output_file_id", None):
                content = client.files.content(done.output_file_id)
                assert content.content, "completed batch returned empty output"

        # 6. cancel batch -------------------------------------------------
        cancelled = _run(case, OP_CANCEL, lambda: strat.cancel_batch(client, batch_id))
        if cancelled is not None:
            assert cancelled.status in (
                "cancelling",
                "cancelled",
            ), f"unexpected status after cancel: {cancelled.status}"

    finally:
        # 7. delete file (best-effort cleanup, but still contract-checked) -
        if file_id is not None:
            deleted = _run(case, OP_DELETE, lambda: strat.delete_file(client, file_id))
            if deleted is not None:
                assert deleted.deleted is True, "file delete did not report deleted"
        try:
            os.remove(input_path)
        except OSError:
            pass
