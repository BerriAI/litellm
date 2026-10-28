import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, cast

import httpx
import pytest
from pydantic import TypeAdapter

from litellm.proxy.roi_calculator.analytics import summarize
from litellm.proxy.roi_calculator.estimator import CompletionCaller
from litellm.proxy.roi_calculator.github import GitHubPullListItem
from litellm.proxy.roi_calculator.sync import SpendReader, SyncManager, read_gateway_user_emails, read_spend
from litellm.types.roi_calculator import (
    ROIBranchSpend,
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
    "head": {"sha": "abcdef", "ref": "feature", "repo": {"full_name": "org/repo"}},
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
  "head": {"sha": "abcdef", "ref": "feature", "repo": {"full_name": "org/repo"}},
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
        self.pull_writes: int = 0

    async def get_param(self, param_name: str) -> _Parameter | None:
        value: Final = self.values.get(param_name)
        return _Parameter(value) if value is not None else None

    async def set_param(self, param_name: str, param_value: object) -> object:
        if param_name.startswith("roi_calculator_pull_"):
            self.pull_writes += 1
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
        if where == {"user_email": {"not": None}}:
            return (
                {"user_id": "u1", "user_email": " Alice@Example.com "},
                {"user_id": "inactive", "user_email": "inactive@example.com"},
                {"user_id": "invalid", "user_email": "not-an-email"},
                {"user_id": "private", "user_email": "123@users.noreply.github.com"},
            )
        assert where == {"user_id": {"in": ["missing", "team@example.com", "u1"]}}
        return (MappingProxyType({"user_id": "u1", "user_email": " Alice@Example.com "}),)


class _SpendDatabase:
    def __init__(self, directory: tuple[Mapping[str, str], ...] = ()) -> None:
        self.litellm_dailyuserspend: Final = _DailySpendTable()
        self.litellm_usertable: Final = _UserTable()
        self.directory: Final = directory or (
            {"user_id": "inactive", "user_email": "inactive@example.com"},
            {"user_id": "invalid", "user_email": "not-an-email"},
            {"user_id": "private", "user_email": "123@users.noreply.github.com"},
            {"user_id": "u1", "user_email": " Alice@Example.com "},
        )
        self.pages_read = 0

    async def query_raw(self, query: str, *args: object) -> object:
        cursor, size = args
        assert cursor is None or isinstance(cursor, str)
        assert isinstance(size, int) and 0 < size <= 1000
        self.pages_read += 1
        return tuple(row for row in self.directory if cursor is None or row["user_id"] > cursor)[:size]


class _SpendPrismaClient:
    def __init__(self, directory: tuple[Mapping[str, str], ...] = ()) -> None:
        self.db: Final = _SpendDatabase(directory)


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


async def _gateway_users() -> frozenset[str]:
    return frozenset({"alice@example.com"})


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

    assert await manager.start(
        _settings(), repository, _spend_reader(), complete, _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)

    async def unexpected_completion(request: ROICompletionRequest) -> object:
        raise AssertionError("A reused estimate must not call the estimator.")

    assert await manager.start(
        _settings(),
        repository,
        _spend_reader(),
        unexpected_completion,
        _transport(unexpected_details=True, profile_email="new@example.com"),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)

    assert manager.status.phase == "complete"
    assert manager.status.reused == 1
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert report["pulls"][0]["estimate"].get("cached") is True
    assert report["pulls"][0]["source_branch"] == "feature"
    assert report["pulls"][0]["source_repo"] == "github.com/org/repo"
    assert report["pulls"][0]["profile_email"] == "new@example.com"
    assert report["pulls"][0]["emails"] == ("alice@example.com", "new@example.com")


def _gitlab_transport(source_path: str | None, *, details_fail: bool = False) -> httpx.MockTransport:
    detail: Final = {
        "iid": 42,
        "title": "Fix timezone conversion",
        "description": "Preserve UTC behavior.",
        "web_url": "https://gitlab.com/org/repo/-/merge_requests/42",
        "author": {"username": "alice"},
        "merged_at": "2026-09-12T12:00:00Z",
        "updated_at": "2026-09-12T12:00:00Z",
        "sha": "abcdef",
        "source_branch": "feature",
        "source_project_id": 2,
        "changes_count": "1",
    }

    def respond(request: httpx.Request) -> httpx.Response:
        path: Final = request.url.path
        if path.endswith("/projects/org/repo"):
            return httpx.Response(200, json={"id": 1, "path_with_namespace": "org/repo"})
        if path.endswith("/projects/2"):
            return (
                httpx.Response(200, json={"id": 2, "path_with_namespace": source_path})
                if source_path
                else httpx.Response(404)
            )
        if path.endswith("/merge_requests"):
            return httpx.Response(
                200, json=[detail, {**detail, "iid": 43, "source_branch": "other"}] if details_fail else [detail]
            )
        if path.endswith("/merge_requests/43"):
            return httpx.Response(200, json={**detail, "iid": 43, "source_branch": "other"})
        if path.endswith("/merge_requests/42"):
            return httpx.Response(404) if details_fail else httpx.Response(200, json=detail)
        if path.endswith("/diffs"):
            return httpx.Response(200, json=[{"new_path": "time.py", "old_path": "time.py", "diff": "+fixed"}])
        if path.endswith("/commits"):
            return httpx.Response(200, json=[{"id": "abcdef", "message": "Fix timezone conversion"}])
        if path.endswith("/users"):
            return httpx.Response(200, json=[{"username": "alice", "public_email": "alice@example.com"}])
        raise AssertionError(path)

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.parametrize("before,after", [(None, "dev/fork"), ("dev/fork", None), ("dev/fork", "dev/renamed")])
async def test_gitlab_cache_refreshes_branch_attribution_when_source_access_changes(
    before: str | None, after: str | None
) -> None:
    settings: Final = _settings().model_copy(update={"source_provider": "gitlab"})
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)

    async def branch_spend(start: date, end: date, repos: tuple[str, ...]) -> tuple[ROIBranchSpend, ...]:
        return (ROIBranchSpend(repo="gitlab.com/" + (after or "dev/fork"), branch="feature", spend=2.5, requests=3),)

    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        _completion(),
        _gitlab_transport(before),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "complete"
    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        _completion(),
        _gitlab_transport(after),
        branch_spend_reader=branch_spend,
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "complete"
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert report["pulls"][0]["source_repo"] == ("gitlab.com/" + after if after else "")
    result: Final = summarize(report, {})
    assert result["pulls"][0]["branch_cost"].status == ("matched" if after else "unattributed")

    async def unexpected_completion(request: ROICompletionRequest) -> object:
        raise AssertionError("Unchanged source metadata must reuse the estimate")

    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        unexpected_completion,
        _gitlab_transport(after),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "complete"
    assert manager.status.reused == 1


@pytest.mark.asyncio
async def test_unreadable_gitlab_details_keep_known_branch_costs() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    settings: Final = _settings().model_copy(update={"source_provider": "gitlab"})

    async def branch_spend(start: date, end: date, repos: tuple[str, ...]) -> tuple[ROIBranchSpend, ...]:
        return (ROIBranchSpend(repo="gitlab.com/dev/fork", branch="feature", spend=2.5, requests=3),)

    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        _completion(),
        _gitlab_transport("dev/fork", details_fail=True),
        branch_spend_reader=branch_spend,
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    result: Final = summarize(report, {})
    assert result["pulls"][0]["branch_cost"].spend == 2.5
    assert result["pulls"][0]["estimate"]["status"] == "needs_review"
    assert result["branch_metrics"].matched_pulls == 1
    assert result["branch_metrics"].cost_per_hour is None


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

    assert await manager.start(
        _settings(), repository, _spend_reader(), _completion(), _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)
    previous: Final = repository.values["roi_calculator_report"]
    assert await manager.start(
        _settings(estimator_prompt="New prompt invalidates saved estimates"),
        repository,
        _spend_reader(),
        _completion(),
        _transport(pull_detail_status=500),
        gateway_user_reader=_gateway_users,
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
        gateway_user_reader=_gateway_users,
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

    assert await manager.start(
        _settings(), repository, _spend_reader(), _completion(), _transport(), gateway_user_reader=_gateway_users
    )
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
        gateway_user_reader=_gateway_users,
    )
    await entered_estimator.wait()

    assert await manager.cancel()
    assert manager.status.phase == "cancelled"
    assert repository.values["roi_calculator_report"] is previous_report


@pytest.mark.asyncio
async def test_immediate_cancel_allows_another_run() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    assert await manager.start(
        _settings(), repository, _spend_reader(), _completion(), _transport(), gateway_user_reader=_gateway_users
    )
    assert await manager.cancel()
    assert manager.status.phase == "cancelled"
    assert manager.status.finished_at is not None
    assert await manager.start(
        _settings(), repository, _spend_reader(), _completion(), _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "complete"


@pytest.mark.asyncio
async def test_saved_estimates_survive_report_reset() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    assert await manager.start(
        _settings(), repository, _spend_reader(), _completion(), _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)
    repository.values = MappingProxyType(
        {key: value for key, value in repository.values.items() if key != "roi_calculator_report"}
    )

    async def unexpected_completion(request: ROICompletionRequest) -> object:
        raise AssertionError("Saved estimates should survive report reset")

    restarted: Final = SyncManager(clock=_fixed_now)
    assert await restarted.start(
        _settings(),
        repository,
        _spend_reader(),
        unexpected_completion,
        _transport(unexpected_details=True),
        gateway_user_reader=_gateway_users,
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
        _settings(),
        repository,
        _spend_reader(),
        blocked_completion,
        _transport(),
        coordinator=coordinator,
        gateway_user_reader=_gateway_users,
    )
    await entered.wait()
    assert not await manager.start(
        _settings(),
        repository,
        _spend_reader(),
        _completion(),
        _transport(),
        coordinator=coordinator,
        gateway_user_reader=_gateway_users,
    )
    assert coordinator.current is not None
    coordinator.current = coordinator.current.model_copy(update={"running": False, "phase": "error"})
    assert await manager.start(
        _settings(),
        repository,
        _spend_reader(),
        _completion(),
        _transport(),
        coordinator=coordinator,
        gateway_user_reader=_gateway_users,
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
    assert await manager.start(
        _settings(),
        repository,
        _spend_reader(),
        _completion(),
        httpx.MockTransport(respond),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert tuple((pull["number"], pull["estimate"]["status"]) for pull in report["pulls"]) == (
        (42, "estimated"),
        (43, "needs_review"),
    )
    assert manager.status.phase == "complete"
    assert manager.status.estimated == 1
    assert manager.status.needs_attention == 1
    assert report["pulls"][1]["source_repo"] == "github.com/org/repo"
    assert report["pulls"][1]["source_branch"] == "feature"


def _repository_outage_transport(
    status: int, *, all_unavailable: bool = False, healthy_empty: bool = False
) -> httpx.MockTransport:
    baseline: Final = _transport()

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/org/unavailable/pulls":
            return httpx.Response(status, json=[] if status == 200 else {"message": "Repository unavailable"})
        if all_unavailable and request.url.path.endswith("/pulls"):
            return httpx.Response(status)
        if healthy_empty and request.url.path == "/repos/org/repo/pulls":
            return httpx.Response(200, json=[])
        return baseline.handle_request(request)

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (403, 404, 429))
async def test_unavailable_repository_publishes_flagged_partial_report_and_recovers(status: int) -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    settings: Final = _settings().model_copy(update=MappingProxyType({"repos": ("org/repo", "org/unavailable")}))

    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        _completion(),
        _repository_outage_transport(status),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)

    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    summary: Final = summarize(report, MappingProxyType({}))
    assert manager.status.phase == "complete"
    assert manager.status.estimated == 1
    assert report["unavailable_repos"] == ("org/unavailable",)
    assert "Incomplete report" in report["warnings"][0] and "org/unavailable" in report["warnings"][0]
    assert report["pulls"][0]["estimate"]["status"] == "estimated"
    assert summary["metrics"]["total_output_hours"] == 4
    assert summary["metrics"]["cost_per_hour"] is None
    assert summary["metrics"]["hours_per_dollar"] is None
    assert all(person["cost_per_hour"] is None for person in summary["people"])

    async def unexpected_completion(request: ROICompletionRequest) -> object:
        raise AssertionError("The healthy repository's estimate must be reused after recovery")

    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        unexpected_completion,
        _repository_outage_transport(200),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    recovered: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert recovered["unavailable_repos"] == ()
    assert recovered["warnings"] == ()
    assert manager.status.reused == 1
    assert summarize(recovered, MappingProxyType({}))["metrics"]["cost_per_hour"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("all_unavailable", (True, False))
async def test_repository_outage_without_usable_pulls_preserves_previous_report(all_unavailable: bool) -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    settings: Final = _settings().model_copy(update=MappingProxyType({"repos": ("org/repo", "org/unavailable")}))
    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        _completion(),
        _repository_outage_transport(200),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    previous: Final = repository.values["roi_calculator_report"]

    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        _completion(),
        _repository_outage_transport(403, all_unavailable=all_unavailable, healthy_empty=not all_unavailable),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "error"
    assert manager.status.error is not None and "No new report was published" in manager.status.error
    assert repository.values["roi_calculator_report"] == previous


@pytest.mark.asyncio
@pytest.mark.parametrize("profile_status", (200, 403, 429, 503))
async def test_reused_profile_preserves_email_only_when_lookup_fails(profile_status: int) -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    baseline: Final = _transport()

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/commits"):
            return httpx.Response(200, content=_COMMITS_JSON.replace("alice@example.com", ""))
        return baseline.handle_request(request)

    assert await manager.start(
        _settings(),
        repository,
        _spend_reader(),
        _completion(),
        httpx.MockTransport(respond),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)

    def refreshed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/alice":
            return httpx.Response(profile_status, json={"email": None})
        return baseline.handle_request(request)

    async def unexpected_completion(request: ROICompletionRequest) -> object:
        raise AssertionError("A reused estimate must not call the estimator")

    assert await manager.start(
        _settings(),
        repository,
        _spend_reader(),
        unexpected_completion,
        httpx.MockTransport(refreshed),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    expected: Final = "" if profile_status == 200 else "alice@example.com"
    assert manager.status.phase == "complete"
    assert manager.status.reused == (0 if profile_status == 200 else 1)
    assert report["pulls"][0]["profile_email"] == expected
    assert report["pulls"][0]["emails"] == ((expected,) if expected else ())
    assert summarize(report, MappingProxyType({}))["metrics"]["cost_per_hour"] == (None if profile_status == 200 else 3)
    repository.values = MappingProxyType(
        {key: value for key, value in repository.values.items() if key != "roi_calculator_report"}
    )

    def unavailable_profile(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/alice":
            return httpx.Response(503)
        return baseline.handle_request(request)

    restarted: Final = SyncManager(clock=_fixed_now)
    assert await restarted.start(
        _settings(),
        repository,
        _spend_reader(),
        unexpected_completion,
        httpx.MockTransport(unavailable_profile),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(restarted)
    subsequent: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert subsequent["pulls"][0]["profile_email"] == expected
    assert subsequent["pulls"][0]["emails"] == ((expected,) if expected else ())
    assert repository.pull_writes == (2 if profile_status == 200 else 1)


@pytest.mark.asyncio
async def test_complete_estimator_outage_preserves_report_and_recovers() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    assert await manager.start(
        _settings(), repository, _spend_reader(), _completion(), _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)
    previous: Final = repository.values["roi_calculator_report"]
    changed: Final = _settings(estimator_prompt="Updated estimation instructions")

    async def failed_completion(request: ROICompletionRequest) -> object:
        raise httpx.ConnectError("Estimator unavailable")

    assert await manager.start(
        changed, repository, _spend_reader(), failed_completion, _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "error"
    assert manager.status.error is not None and "No new report was published" in manager.status.error
    assert repository.values["roi_calculator_report"] == previous
    assert await manager.start(
        changed, repository, _spend_reader(), _completion(), _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "complete"
    recovered: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert recovered["pulls"][0]["estimate"]["hours"] == 4


class _CompletionRecorder:
    def __init__(self) -> None:
        self.requests: tuple[ROICompletionRequest, ...] = ()

    async def __call__(self, request: ROICompletionRequest) -> object:
        self.requests = (*self.requests, request)
        return await _completion()(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("registered", "mapping", "expected_calls"),
    (
        (frozenset(), MappingProxyType({}), 0),
        (frozenset({"alice@example.com"}), MappingProxyType({}), 1),
        (frozenset({"other@example.com"}), MappingProxyType({}), 0),
        (frozenset({"other@example.com"}), MappingProxyType({"alice": "other@example.com"}), 1),
        (frozenset({"alice@example.com"}), MappingProxyType({"alice": "outside@example.com"}), 0),
        (frozenset({"alice@example.com", "profile@example.com"}), MappingProxyType({}), 0),
    ),
)
async def test_only_authors_linked_to_registered_gateway_users_trigger_estimation(
    registered: frozenset[str], mapping: Mapping[str, str], expected_calls: int
) -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    recorder: Final = _CompletionRecorder()
    settings: Final = _settings().model_copy(update={"identity_map": mapping})

    async def users() -> frozenset[str]:
        return registered

    assert await manager.start(
        settings,
        repository,
        _spend_reader(),
        recorder,
        _transport(profile_email="profile@example.com"),
        gateway_user_reader=users,
    )
    await _wait_until_finished(manager)

    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    estimate: Final = report["pulls"][0]["estimate"]
    assert manager.status.phase == "complete"
    assert len(recorder.requests) == expected_calls
    assert repository.pull_writes == expected_calls
    assert estimate["status"] == ("estimated" if expected_calls else "needs_review")
    assert estimate["hours"] == (4 if expected_calls else None)


@pytest.mark.asyncio
async def test_registered_author_without_spend_is_estimated() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    recorder: Final = _CompletionRecorder()

    async def no_spend(start: date, end: date) -> tuple[ROISpendRecord, ...]:
        return ()

    assert await manager.start(
        _settings(), repository, no_spend, recorder, _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert len(recorder.requests) == 1
    assert report["pulls"][0]["estimate"]["hours"] == 4
    assert report["spend"] == ()


@pytest.mark.asyncio
async def test_unlinked_author_is_estimated_after_linking_and_cached_estimate_is_hidden_after_unlinking() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    recorder: Final = _CompletionRecorder()

    async def users() -> frozenset[str]:
        return frozenset({"member@example.com"})

    async def run(settings: ROISettings) -> ROIReport:
        assert await manager.start(
            settings, repository, _spend_reader(), recorder, _transport(), gateway_user_reader=users
        )
        await _wait_until_finished(manager)
        assert manager.status.phase == "complete"
        return TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])

    unlinked: Final = await run(_settings())
    assert unlinked["pulls"][0]["estimate"]["hours"] is None
    assert len(recorder.requests) == 0
    linked_settings: Final = _settings().model_copy(update={"identity_map": {"alice": "member@example.com"}})
    linked: Final = await run(linked_settings)
    assert linked["pulls"][0]["estimate"]["hours"] == 4
    assert len(recorder.requests) == 1
    unlinked_again: Final = await run(_settings())
    assert unlinked_again["pulls"][0]["estimate"]["hours"] is None
    assert manager.status.reused == 0
    assert len(recorder.requests) == 1
    relinked: Final = await run(linked_settings)
    assert relinked["pulls"][0]["estimate"]["hours"] == 4
    assert manager.status.reused == 1
    assert len(recorder.requests) == 1


@pytest.mark.asyncio
async def test_unavailable_gateway_directory_stops_estimation_and_preserves_report() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    recorder: Final = _CompletionRecorder()
    assert await manager.start(
        _settings(), repository, _spend_reader(), recorder, _transport(), gateway_user_reader=_gateway_users
    )
    await _wait_until_finished(manager)
    previous: Final = repository.values["roi_calculator_report"]

    async def unavailable_users() -> frozenset[str]:
        raise ConnectionError("Gateway directory unavailable")

    assert await manager.start(
        _settings("Changed prompt"),
        repository,
        _spend_reader(),
        recorder,
        _transport(),
        gateway_user_reader=unavailable_users,
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "error"
    assert len(recorder.requests) == 1
    assert repository.values["roi_calculator_report"] == previous


@pytest.mark.asyncio
async def test_gateway_directory_includes_users_without_spend_and_normalizes_emails() -> None:
    assert await read_gateway_user_emails(_SpendPrismaClient()) == frozenset(
        {"alice@example.com", "inactive@example.com"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", (1000, 2501))
async def test_gateway_directory_reads_every_page(size: int) -> None:
    directory: Final = tuple(
        {"user_id": f"user-{index:04d}", "user_email": f" Member-{index}@Example.com "} for index in range(size)
    )
    client: Final = _SpendPrismaClient(directory)
    assert await read_gateway_user_emails(client) == frozenset(f"member-{index}@example.com" for index in range(size))
    assert client.db.pages_read == size // 1000 + 1


@pytest.mark.asyncio
async def test_unlinked_results_survive_when_the_only_linked_estimate_fails() -> None:
    repository: Final = _ReportRepository()
    manager: Final = SyncManager(clock=_fixed_now)
    baseline: Final = _transport()

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/org/repo/pulls":
            return httpx.Response(
                200,
                content=_PULL_LIST_JSON[:-1]
                + ","
                + _PULL_LIST_JSON[1:].replace("42", "43").replace("alice", "outsider"),
            )
        if request.url.path.startswith("/repos/org/repo/pulls/43"):
            original: Final = baseline.handle_request(httpx.Request("GET", str(request.url).replace("/43", "/42")))
            return httpx.Response(
                original.status_code, content=original.text.replace("42", "43").replace("alice", "outsider")
            )
        if request.url.path == "/users/outsider":
            return httpx.Response(200, json={"email": "outsider@example.com"})
        return baseline.handle_request(request)

    async def failed_completion(request: ROICompletionRequest) -> object:
        raise httpx.ConnectError("Estimator unavailable")

    assert await manager.start(
        _settings(),
        repository,
        _spend_reader(),
        failed_completion,
        httpx.MockTransport(respond),
        gateway_user_reader=_gateway_users,
    )
    await _wait_until_finished(manager)
    assert manager.status.phase == "complete", manager.status.error
    report: Final = TypeAdapter(ROIReport).validate_python(repository.values["roi_calculator_report"])
    assert tuple(
        (pull["login"], pull["estimate"]["status"], pull["estimate"]["hours"]) for pull in report["pulls"]
    ) == (
        ("alice", "error", None),
        ("outsider", "needs_review", None),
    )
    assert "not linked" in report["pulls"][1]["estimate"]["reasoning"]
