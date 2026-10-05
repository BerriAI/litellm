from collections.abc import Mapping
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.roi_calculator import ROIBranchAttribution, ROIBranchSpend, ROISpendRecord


class ObservedModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ObservedWindow(ObservedModel):
    start: date
    end: date


class ObservedSource(ObservedModel):
    id: str
    source_provider: Literal["github", "gitlab"]
    api_url: str
    repos: tuple[str, ...]


class ObservedAccount(ObservedModel):
    connection_id: str
    login: str


class ObservedPull(ObservedModel):
    connection_id: str = ""
    repo: str
    number: int
    title: str
    url: str
    author: str
    agent: bool = False
    requester: str = ""
    profile_email: str = ""
    created_at: datetime | None = None
    merged_at: datetime
    source_repo: str = ""
    source_branch: str = ""


class ObservedIssue(ObservedModel):
    repo: str
    number: int
    created_at: datetime
    labels: tuple[str, ...] = ()


class ObservedPeriodData(ObservedModel):
    window: ObservedWindow
    pulls: tuple[ObservedPull, ...]
    issues: tuple[ObservedIssue, ...] | None
    spend: tuple[ROISpendRecord, ...]
    branch_spend: tuple[ROIBranchSpend, ...] | None = None


class ObservedData(ObservedModel):
    source_provider: Literal["github", "gitlab", "mixed"]
    connections: tuple[ObservedSource, ...] = ()
    source_api_url: str
    repos: tuple[str, ...]
    captured_at: datetime
    gateway_emails: tuple[str, ...]
    current: ObservedPeriodData
    previous: ObservedPeriodData
    last_year: ObservedPeriodData


class ObservedPersonPeriod(ObservedModel):
    merged_prs: int
    prs_per_week: float
    median_merge_hours: float | None
    direct_authored: int
    declared_agent_owned: int
    gateway_recorded_spend: float
    recorded_spend_per_attributed_pr: float | None
    spend_observation: Literal["records_present", "no_records"]
    pr_urls: tuple[str, ...]


class ObservedPersonPeriods(ObservedModel):
    current: ObservedPersonPeriod
    previous: ObservedPersonPeriod
    last_year: ObservedPersonPeriod


class ObservedPerson(ObservedModel):
    name: str
    email: str
    logins: tuple[str, ...]
    accounts: tuple[ObservedAccount, ...] = ()
    periods: ObservedPersonPeriods


class ObservedHumanSummary(ObservedModel):
    median_merge_hours: float | None


class ObservedPeriod(ObservedModel):
    window: ObservedWindow
    merged_prs: int
    median_merge_hours: float | None
    human_authored: int
    agent_authored: int
    missing_author: int
    agents_without_requester: int
    matched_internal_prs: int
    new_bug_labeled_issues: int | None
    new_regression_labeled_issues: int | None
    explicitly_titled_revert_prs: int
    matched_users_recorded_spend: float
    spend_observation: Literal["records_present", "no_records"]
    human_summary: ObservedHumanSummary


class ObservedPeriods(ObservedModel):
    current: ObservedPeriod
    previous: ObservedPeriod
    last_year: ObservedPeriod


class ObservedPullResponse(ObservedPull):
    merge_hours: float | None
    branch_cost: ROIBranchAttribution


class ObservedPullPeriods(ObservedModel):
    current: tuple[ObservedPullResponse, ...]
    previous: tuple[ObservedPullResponse, ...]
    last_year: tuple[ObservedPullResponse, ...]


class ObservedReport(ObservedModel):
    source_provider: Literal["github", "gitlab", "mixed"]
    connections: tuple[ObservedSource, ...] = ()
    repos: tuple[str, ...]
    captured_at: datetime
    periods: ObservedPeriods
    people: tuple[ObservedPerson, ...]
    pulls: ObservedPullPeriods
    unlinked_branches: tuple[ROIBranchSpend, ...]
    unmatched_logins: tuple[str, ...]


class ObservedReportResponse(ObservedModel):
    report: ObservedReport | None


class ObservedIdentityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    logins: tuple[str, ...] = Field(default=(), max_length=100)
    accounts: tuple[ObservedAccount, ...] | None = Field(default=None, max_length=500)


class ObservedConnectionIdentities(ObservedSource):
    identity_map: Mapping[str, str]
    unmatched_logins: tuple[str, ...]


class ObservedIdentities(ObservedModel):
    gateway_emails: tuple[str, ...]
    identity_map: Mapping[str, str]
    unmatched_logins: tuple[str, ...]
    connections: tuple[ObservedConnectionIdentities, ...] = ()


class ObservedConnection(ObservedModel):
    id: str = ""
    source_provider: Literal["github", "gitlab"]
    api_url: str
    repos: tuple[str, ...]
    has_token: bool
    update_interval_minutes: float
    ready: bool
    connection_type: Literal["token", "app"]


class ObservedSettings(ObservedConnection):
    connections: tuple[ObservedConnection, ...] = ()


class ObservedSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str | None = Field(default=None, max_length=100)
    source_provider: Literal["github", "gitlab"]
    api_url: str
    token: str | None = None
    repos: tuple[str, ...]
    update_interval_minutes: float | None = Field(default=None, ge=0, le=43200, allow_inf_nan=False)


class ObservedApp(ObservedModel):
    configured: bool
    can_install: bool = False
    api_url: str | None = None
    callback_url: str | None = None


class ObservedApps(ObservedModel):
    github: ObservedApp
    gitlab: ObservedApp


class ObservedAuthorization(ObservedModel):
    url: str
