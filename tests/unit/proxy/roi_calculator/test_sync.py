import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, cast

import httpx
import pytest
from pydantic import TypeAdapter

from litellm.proxy.roi_calculator.estimator import CompletionCaller
from litellm.proxy.roi_calculator.github import GitHubPullListItem
from litellm.proxy.roi_calculator.sync import SpendReader, SyncManager, read_spend
from litellm.types.roi_calculator import (
    ROICompletionRequest,
    ROIReport,
    ROISettings,
    ROISpendRecord,
    ROISyncStatus,
)

_PULL_LIST_JSON: Final = """[
  {
    "number": 42,
    "title": "Fix timezone conversion",
    "body": "Preserve UTC behavior.",
    "merged_at": "2026-09-12T12:00:00Z",
    "updated_at": "2026-09-12T12:00:00Z",
    "head": {"sha": "abcdef"},
    "user": {"login": "alice"}
  }
]"""
_PULL_DETAIL_JSON: Final = """{
  "number": 42,
  "title": "Fix timezone conversion",
  "body": "Preserve UTC behavior.",
  "html_url": "https://github.com/org/repo/pull/42",
  "user": {"login": "alice"},
  "merged_at": "2026-09-12T12:00:00Z",
  "head": {"sha": "abcdef"},
  "additions": 1,
  "deletions": 1,
  "changed_files": 1,
  "commits": 1
}"""
_PULL_FILES_JSON: Final = """[
  {"filename": "time.py", "status": "modified", "additions": 1, "deletions": 1}
]"""
_USER_JSON: Final = """{"email": "alice@example.com"}"""
_COMMITS_JSON: Final = """[
  {
    "sha": "abcdef",
    "author": {"login": "alice"},
    "commit": {
      "message": "Fix timezone conversion",
      "author": {"email": "alice@example.com"}
    }
  }
]"""


def _assert_json_round_trip(value: object) -> None:
    serialized: Final = json.dumps(value)
    decoded: Final[object] = cast(object, json.loads(serialized))
    assert decoded == value


class _Parameter:
    def __init__(self, param_value: object) -> None:
        self.param_value: Final = param_value


class _ReportRepository:
    def __init__(self) -> None:
        self.values: Mapping[str, object] = MappingProxyType({})

    async def get_param(self, param_name: str) -> _Parameter | None:
        value: Final = self.values.get(param_name)
        return _Parameter(value) if value is not None else None

    async def set_param(self, param_name: str, param_value: object) -> object:
        _assert_json_round_trip(param_value)
        self.values = MappingProxyType({**self.values, param_name: param_value})
        return self.values[param_name]


class _DailySpendTable:
    async def group_by(
        self,
        *,
        by: Sequence[Literal["user_id", "date"]],
        sum: Mapping[str, object],
        where: Mapping[str, object],
        order: Mapping[str, object],
    ) -> Sequence[Mapping[str, object]]:
        _assert_json_round_trip({"by": by, "sum": sum, "where": where, "order": order})
        assert by == ["user_id", "date"]
        assert sum == {"spend": True, "api_requests": True}
        assert where == {"date": {"gte": "2026-09-01", "lte": "2026-09-30"}}
        assert order == {"date": "asc"}
        return (
            {
                "user_id": "u1",
                "date": "2026-09-12",
                "_sum": {"spend": 12.5, "api_requests": 2},
            },
            {
                "user_id": "team@example.com",
                "date": "2026-09-13",
                "_sum": {"spend": 3.0, "api_requests": 1},
            },
            {
                "user_id": "missing",
                "date": "2026-09-14",
                "_sum": {"spend": 1.0, "api_requests": 1},
            },
        )


class _UserTable:
    async def find_many(
        self,
        *,
        where: Mapping[str, object],
    ) -> Sequence[Mapping[str, str | None]]:
        _assert_json_round_trip({"where": where})
        assert where == {"user_id": {"in": ["missing", "team@example.com", "u1"]}}
        return (MappingProxyType({"user_id": "u1", "user_email": " Alice@Example.com "}),)


class _SpendDatabase:
    def __init__(self) -> None:
        self.litellm_dailyuserspend: Final = _DailySpendTable()
        self.litellm_usertable: Final = _UserTable()


class _SpendPrismaClient:
    def __init__(self) -> None:
        self.db: Final = _SpendDatabase()


def _settings(estimator_prompt: str = "Estimate effort.") -> ROISettings:
    return ROISettings(
        github_api_url="https://api.github.com",
        repos=("org/repo",),
        estimator_model="test-estimator",
        estimator_prompt=estimator_prompt,
        backfill_days=30,
    )


def _transport(
    pull_detail_status: int = 200,
    unexpected_details: bool = False,
    profile_email: str = "alice@example.com",
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/org/repo/pulls":
            return httpx.Response(200, content=_PULL_LIST_JSON)
        if path == "/repos/org/repo/pulls/42":
            if unexpected_details:
                raise AssertionError("A reused estimate must not fetch pull request details.")
            return httpx.Response(pull_detail_status, content=_PULL_DETAIL_JSON)
        if path == "/repos/org/repo/pulls/42/files":
            return httpx.Response(
                200,
                content=_PULL_FILES_JSON,
            )
        if path == "/users/alice":
            return httpx.Response(200, json={"email": profile_email})
        if path == "/repos/org/repo/pulls/42/commits":
            return httpx.Response(200, content=_COMMITS_JSON)
        raise AssertionError(f"Unexpected GitHub request: {request.method} {path}")

    return httpx.MockTransport(respond)


def _spend_reader() -> SpendReader:
    async def read(start: date, end: date) -> tuple[ROISpendRecord, ...]:
        record: Final[ROISpendRecord] = {
            "date": "2026-09-12",
            "user_id": "alice-id",
            "email": "alice@example.com",
            "spend": 12.0,
            "requests": 2,
        }
        return (record,)

    return read


def _completion() -> CompletionCaller:
    async def complete(request: ROICompletionRequest) -> object:
        assert request.model == "test-estimator"
        message: Final = MappingProxyType(
            {"content": '{"hours": 4, "reasoning": "Timezone conversion and regression verification."}'}
        )
        choice: Final = MappingProxyType({"finish_reason": "stop", "message": message})
        response: Final = MappingProxyType({"choices": (choice,)})
        return response

    return complete


def _fixed_now() -> datetime:
    return datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


async def _wait_until_finished(manager: SyncManager) -> None:
    while manager.status.running:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_unchanged_estimated_pull_refreshes_identity_without_model_call() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    complete: Final = _completion()

    assert await manager.start(_settings(), repository, _spend_reader(), complete, _transport())
    await _wait_until_finished(manager)

    async def unexpected_completion(request: ROICompletionRequest) -> object:
        raise AssertionError("A reused estimate must not call the estimator.")

    assert await manager.start(
        _settings(),
        repository,
        _spend_reader(),
        unexpected_completion,
        _transport(unexpected_details=True, profile_email="new@example.com"),
    )
    await _wait_until_finished(manager)

    assert manager.status.phase == "complete"
    assert manager.status.reused == 1
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert report["pulls"][0]["estimate"].get("cached") is True
    assert report["pulls"][0]["profile_email"] == "new@example.com"
    assert report["pulls"][0]["emails"] == ("alice@example.com", "new@example.com")


@pytest.mark.asyncio
async def test_read_spend_joins_user_emails_and_preserves_unmatched_identities() -> None:
    spend: Final = await read_spend(
        _SpendPrismaClient(),
        date(2026, 9, 1),
        date(2026, 9, 30),
    )

    expected_first: Final[ROISpendRecord] = {
        "date": "2026-09-12",
        "user_id": "u1",
        "email": "alice@example.com",
        "spend": 12.5,
        "requests": 2,
    }
    expected_second: Final[ROISpendRecord] = {
        "date": "2026-09-13",
        "user_id": "team@example.com",
        "email": "team@example.com",
        "spend": 3.0,
        "requests": 1,
    }
    expected_third: Final[ROISpendRecord] = {
        "date": "2026-09-14",
        "user_id": "missing",
        "email": "",
        "spend": 1.0,
        "requests": 1,
    }
    assert spend == (expected_first, expected_second, expected_third)


@pytest.mark.asyncio
async def test_metadata_outage_keeps_previous_report_and_retries_on_next_run() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)

    assert await manager.start(_settings(), repository, _spend_reader(), _completion(), _transport())
    await _wait_until_finished(manager)
    previous: Final = repository.values["roi_calculator_report"]
    assert await manager.start(
        _settings(estimator_prompt="New prompt invalidates saved estimates"),
        repository,
        _spend_reader(),
        _completion(),
        _transport(pull_detail_status=500),
    )
    await _wait_until_finished(manager)

    assert manager.status.phase == "error"
    assert manager.status.needs_attention == 1
    assert manager.status.error is not None and "No new report was published" in manager.status.error
    assert repository.values["roi_calculator_report"] == previous
    assert await manager.start(
        _settings(estimator_prompt="New prompt invalidates saved estimates"),
        repository,
        _spend_reader(),
        _completion(),
        _transport(),
    )
    await _wait_until_finished(manager)
    recovered: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert recovered["pulls"][0]["estimate"]["status"] == "estimated"
    assert recovered["pulls"][0]["estimate"]["hours"] == 4
    assert manager.status.reused == 0


@pytest.mark.asyncio
async def test_cancelling_estimation_leaves_the_previous_report_unchanged() -> None:
    entered_estimator: Final = asyncio.Event()
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)

    assert await manager.start(_settings(), repository, _spend_reader(), _completion(), _transport())
    await _wait_until_finished(manager)
    previous_report: Final = repository.values["roi_calculator_report"]

    async def blocked_completion(request: ROICompletionRequest) -> object:
        assert request.model == "test-estimator"
        entered_estimator.set()
        await asyncio.Event().wait()

    assert await manager.start(
        _settings(estimator_prompt="Different estimator instructions."),
        repository,
        _spend_reader(),
        blocked_completion,
        _transport(),
    )
    await entered_estimator.wait()

    assert await manager.cancel()
    assert manager.status.phase == "cancelled"
    assert repository.values["roi_calculator_report"] is previous_report


@pytest.mark.asyncio
async def test_immediate_cancel_allows_another_run() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    assert await manager.start(_settings(), repository, _spend_reader(), _completion(), _transport())
    assert await manager.cancel()
    assert manager.status.phase == "cancelled"
    assert manager.status.finished_at is not None
    assert await manager.start(_settings(), repository, _spend_reader(), _completion(), _transport())
    await _wait_until_finished(manager)
    assert manager.status.phase == "complete"


@pytest.mark.asyncio
async def test_saved_estimates_survive_report_reset() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    assert await manager.start(_settings(), repository, _spend_reader(), _completion(), _transport())
    await _wait_until_finished(manager)
    repository.values = MappingProxyType(
        {key: value for key, value in repository.values.items() if key != "roi_calculator_report"}
    )

    async def unexpected_completion(request: ROICompletionRequest) -> object:
        raise AssertionError("Saved estimates should survive report reset")

    restarted: Final = SyncManager(clock=_fixed_now)
    assert await restarted.start(
        _settings(), repository, _spend_reader(), unexpected_completion, _transport(unexpected_details=True)
    )
    await _wait_until_finished(restarted)
    assert restarted.status.phase == "complete"
    assert restarted.status.reused == 1


class _LeaseCoordinator:
    def __init__(self) -> None:
        self.current: ROISyncStatus | None = None
        self.owner: str | None = None

    async def status(self) -> ROISyncStatus | None:
        return self.current

    async def acquire(self, owner: str, status: ROISyncStatus, scheduled_interval: float = 0) -> bool:
        if self.current is not None and self.current.running:
            return False
        self.owner = owner
        self.current = status
        return True

    async def heartbeat(self, owner: str, status: ROISyncStatus) -> bool:
        return self.owner == owner and self.current is not None and self.current.running

    async def finish(self, owner: str, status: ROISyncStatus, report: ROIReport | None = None) -> bool:
        if self.owner != owner:
            return False
        self.current = status
        return True


@pytest.mark.asyncio
async def test_expired_lease_can_restart_without_restarting_the_gateway() -> None:
    coordinator: Final = _LeaseCoordinator()
    entered: Final = asyncio.Event()
    cancelled: Final = asyncio.Event()
    manager: Final = SyncManager(clock=_fixed_now)
    repository: Final = _ReportRepository()

    async def blocked_completion(request: ROICompletionRequest) -> object:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    assert await manager.start(
        _settings(), repository, _spend_reader(), blocked_completion, _transport(), coordinator=coordinator
    )
    await entered.wait()
    assert not await manager.start(
        _settings(), repository, _spend_reader(), _completion(), _transport(), coordinator=coordinator
    )
    assert coordinator.current is not None
    coordinator.current = coordinator.current.model_copy(update={"running": False, "phase": "error"})
    assert await manager.start(
        _settings(), repository, _spend_reader(), _completion(), _transport(), coordinator=coordinator
    )
    await _wait_until_finished(manager)
    assert cancelled.is_set()
    assert manager.status.phase == "complete"
    assert manager.status.estimated == 1


@pytest.mark.asyncio
async def test_one_unreadable_pr_preserves_other_estimates_in_report() -> None:
    baseline: Final = _transport()
    listed: Final = TypeAdapter(tuple[GitHubPullListItem, ...]).validate_json(_PULL_LIST_JSON)[0]
    second: Final = listed.model_copy(update=MappingProxyType({"number": 43}))
    listing: Final = TypeAdapter(tuple[GitHubPullListItem, ...]).dump_json((listed, second))

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/org/repo/pulls":
            return httpx.Response(200, content=listing)
        if request.url.path == "/repos/org/repo/pulls/43":
            return httpx.Response(404)
        return baseline.handle_request(request)

    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    assert await manager.start(_settings(), repository, _spend_reader(), _completion(), httpx.MockTransport(respond))
    await _wait_until_finished(manager)
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert tuple((pull["number"], pull["estimate"]["status"]) for pull in report["pulls"]) == (
        (42, "estimated"),
        (43, "needs_review"),
    )
    assert manager.status.phase == "complete"
    assert manager.status.estimated == 1
    assert manager.status.needs_attention == 1
