"""Live e2e: a LangSmith batch whose metadata holds non JSON-native Python values
(datetime, Decimal) must reach the real LangSmith API instead of dying in
json.dumps and dropping the whole batch. Only the SDK path can put such values
into the batch (the proxy JSON-decodes request metadata), so this test drives
litellm.acompletion in-process against the real OpenAI API with a LangsmithLogger
injected per request, flushes the batch, and reads the run back by id through
LangSmith's own API. Nothing is mocked.
"""

from __future__ import annotations

import asyncio
import datetime
import decimal
import os
import time
import uuid
from dataclasses import dataclass
from typing import Final

import pytest
from e2e_config import CHEAP_OPENAI_MODEL, POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import Headers, Success, get_external
from pydantic import BaseModel, ConfigDict, Field, JsonValue

import litellm
from litellm.integrations.langsmith import LangsmithLogger

pytestmark = pytest.mark.e2e


class LangsmithHeaders(Headers):
    x_api_key: str = Field(serialization_alias="x-api-key")


class LangsmithRunExtra(BaseModel):
    model_config = ConfigDict(extra="allow")
    requester_metadata: dict[str, JsonValue] | None = None


class LangsmithRun(BaseModel):
    id: str
    session_name: str | None = None
    extra: LangsmithRunExtra


@dataclass(frozen=True, slots=True)
class LangsmithCreds:
    api_key: str
    base_url: str
    project: str


def load_langsmith_creds() -> LangsmithCreds:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        pytest.fail("LangSmith e2e requires LANGSMITH_API_KEY; missing credentials is a hard failure, not a skip")
    if os.getenv("LANGSMITH_MOCK"):
        pytest.fail("LANGSMITH_MOCK is set; this e2e must hit the real LangSmith API")
    return LangsmithCreds(
        api_key=api_key,
        base_url=(os.getenv("LANGSMITH_BASE_URL") or "https://api.smith.langchain.com").rstrip("/"),
        project=os.getenv("LANGSMITH_PROJECT") or "litellm-e2e",
    )


def _fetch_run(creds: LangsmithCreds, run_id: uuid.UUID) -> LangsmithRun | None:
    result = get_external(
        f"{creds.base_url}/runs/{run_id}",
        response_type=LangsmithRun,
        headers=LangsmithHeaders(x_api_key=creds.api_key),
    )
    match result:
        case Success(data=run):
            return run
        case _:
            return None


def _poll_run(creds: LangsmithCreds, run_id: uuid.UUID) -> LangsmithRun:
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        run = _fetch_run(creds, run_id)
        if run is not None:
            return run
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"LangSmith run {run_id} never appeared within {POLL_TIMEOUT}s; the batch flush dropped it")


class TestLangsmithBatchSerialization:
    @pytest.mark.asyncio
    @pytest.mark.covers("logging.langsmith.success.serializes_non_native_metadata")
    async def test_non_json_native_metadata_reaches_langsmith(self) -> None:
        creds: Final = load_langsmith_creds()
        logger: Final = LangsmithLogger(
            langsmith_api_key=creds.api_key, langsmith_project=creds.project, langsmith_base_url=creds.base_url
        )
        assert not logger.is_mock_mode, "LangsmithLogger initialised in mock mode; this e2e needs the real API"
        marker: Final = unique_marker()
        run_id: Final = uuid.uuid4()
        created_at: Final = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
        spend: Final = decimal.Decimal("0.0042")
        response: Final = await litellm.acompletion(
            model=f"openai/{CHEAP_OPENAI_MODEL}",
            messages=[{"role": "user", "content": f"Reply with the single word ok ({marker})"}],
            max_completion_tokens=5,
            callbacks=[logger],
            metadata={"run_id": str(run_id), "metadata": {"marker": marker, "created_at": created_at, "spend": spend}},
        )
        assert isinstance(response, litellm.ModelResponse) and response.id, (
            "a non-streaming completion must return a ModelResponse before the batch flush is meaningful"
        )
        enqueue_deadline: Final = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < enqueue_deadline and len(logger.log_queue) == 0:
            await asyncio.sleep(0.5)
        assert len(logger.log_queue) == 1, (
            f"the completion must be queued for the LangSmith batch, got {len(logger.log_queue)} queued entries"
        )
        await logger.async_send_batch()
        run: Final = _poll_run(creds, run_id)
        requester_metadata: Final = run.extra.requester_metadata
        assert requester_metadata is not None, "the run must carry the caller metadata under extra.requester_metadata"
        assert requester_metadata["marker"] == marker
        assert requester_metadata["created_at"] == str(created_at)
        assert requester_metadata["spend"] == str(spend)
