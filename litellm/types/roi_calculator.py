from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictFloat, StrictInt, field_validator
from typing_extensions import NotRequired, ReadOnly, TypedDict

DEFAULT_PROMPT: Final = (
    "Estimate how many hours it would take an engineer to complete the work in this pull request without AI assistance. "
    "Explain your estimate briefly."
)


class ROISettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    github_api_url: str = "https://api.github.com"
    github_token: SecretStr = SecretStr("")
    estimator_key: SecretStr = SecretStr("")
    repos: tuple[str, ...] = ()
    estimator_model: str = ""
    estimator_prompt: str = DEFAULT_PROMPT
    backfill_days: int = Field(default=7, ge=1, le=3650)
    update_interval_minutes: float = Field(default=1440, ge=0, le=43200, allow_inf_nan=False)
    identity_map: Mapping[str, str] = Field(default_factory=lambda: MappingProxyType({}))

    @field_validator("update_interval_minutes")
    @classmethod
    def validate_update_interval(cls, value: float) -> float:
        if 0 < value < 5:
            raise ValueError("Choose manual updates (0), or an interval of at least 5 minutes.")
        return value

    @field_validator("github_api_url")
    @classmethod
    def normalize_github_api_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        normalized: Final[str] = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("A GitHub API URL is required.")
        parsed: Final = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use an HTTPS GitHub API URL without credentials, query, or fragment.")
        return normalized

    @field_validator("repos")
    @classmethod
    def validate_repositories(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        import re

        normalized_values: Final = tuple(repo.strip().rstrip("/").removesuffix(".git") for repo in values)
        normalized: Final = tuple(
            repo for index, repo in enumerate(normalized_values) if repo not in normalized_values[:index]
        )
        invalid_repositories: Final = tuple(
            repo
            for repo in normalized
            if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None
            or any(part in (".", "..") for part in repo.split("/"))
        )
        if invalid_repositories:
            raise ValueError("Repositories must use owner/repo format.")
        return normalized

    @field_validator("estimator_prompt")
    @classmethod
    def validate_estimator_prompt(cls, value: str) -> str:
        normalized: Final[str] = value.strip()
        if not normalized or len(normalized) > 20000:
            raise ValueError("The estimator prompt must contain between 1 and 20,000 characters.")
        return normalized

    @field_validator("identity_map")
    @classmethod
    def normalize_identity_map(cls, values: Mapping[str, str]) -> Mapping[str, str]:
        import re

        from litellm.proxy.roi_calculator.analytics import normalize_email

        normalized: Final[Mapping[str, str]] = MappingProxyType(
            {
                login.strip().casefold(): normalize_email(address)
                for login, address in values.items()
                if re.fullmatch(r"[A-Za-z0-9_\[\]-]+", login.strip()) is not None and normalize_email(address)
            }
        )
        if len(normalized) != len(values):
            raise ValueError("Each identity needs a GitHub username and a valid gateway email.")
        return normalized


class ROISettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    github_api_url: str | None = None
    github_token: str | None = None
    estimator_key: str | None = None
    repos: tuple[str, ...] | None = None
    estimator_model: str | None = None
    estimator_prompt: str | None = None
    backfill_days: int | None = Field(default=None, ge=1, le=3650)
    update_interval_minutes: float | None = Field(default=None, ge=0, le=43200, allow_inf_nan=False)


class ROISettingsResponse(BaseModel):
    github_api_url: str
    repos: tuple[str, ...]
    estimator_model: str
    estimator_prompt: str
    backfill_days: int
    update_interval_minutes: float
    has_estimator_key: bool
    identity_map: Mapping[str, str]
    has_github_token: bool
    default_prompt: str
    available_models: tuple[str, ...]
    ready: bool


class ROIRepository(BaseModel):
    name: str
    visibility: str
    archived: bool


class ROIRepositoriesResponse(BaseModel):
    repositories: tuple[ROIRepository, ...]
    page: int
    has_more: bool


class ROISyncStatus(BaseModel):
    running: bool
    phase: Literal["idle", "spend", "repositories", "estimates", "complete", "cancelled", "error"]
    stage: str
    done: int
    total: int
    estimated: int
    reused: int
    needs_attention: int
    error: str | None
    started_at: str | None = None
    finished_at: str | None = None
    next_update: str | None = None
    elapsed_seconds: int = 0
    remaining_seconds: int | None = None


class ROISpendRecord(TypedDict):
    date: ReadOnly[str]
    user_id: ReadOnly[str]
    email: ReadOnly[str]
    spend: ReadOnly[float]
    requests: ReadOnly[int]


class ROIEstimate(TypedDict):
    status: ReadOnly[Literal["estimated", "needs_review", "error"]]
    hours: ReadOnly[float | None]
    reasoning: ReadOnly[str]
    model: NotRequired[ReadOnly[str]]
    evidence_source: NotRequired[ReadOnly[str]]
    effort_basis: NotRequired[ReadOnly[str]]
    cached: NotRequired[ReadOnly[bool]]


class ROIPullRecord(TypedDict):
    repo: ReadOnly[str]
    number: ReadOnly[int]
    title: ReadOnly[str]
    url: ReadOnly[str]
    login: ReadOnly[str]
    emails: ReadOnly[tuple[str, ...]]
    profile_email: ReadOnly[str]
    commit_emails: NotRequired[ReadOnly[tuple[str, ...]]]
    merged_at: ReadOnly[str]
    head_sha: ReadOnly[str]
    additions: ReadOnly[int]
    deletions: ReadOnly[int]
    changed_files: ReadOnly[int]
    commit_count: ReadOnly[int]
    incomplete_metadata: ReadOnly[bool]
    estimate: ReadOnly[ROIEstimate]
    cache_key: ReadOnly[str | None]


class ROIReport(TypedDict):
    mode: ReadOnly[str]
    start: ReadOnly[str]
    end: ReadOnly[str]
    synced_at: ReadOnly[str]
    repos: ReadOnly[tuple[str, ...]]
    estimator_model: ReadOnly[str]
    estimator_prompt: ReadOnly[str]
    effort_basis: ReadOnly[str]
    spend: ReadOnly[tuple[ROISpendRecord, ...]]
    pulls: ReadOnly[tuple[ROIPullRecord, ...]]
    settings_fingerprint: ReadOnly[str]
    warnings: NotRequired[ReadOnly[tuple[str, ...]]]
    id: NotRequired[ReadOnly[str]]


class ROIPullFile(TypedDict):
    filename: ReadOnly[str | None]
    status: ReadOnly[str | None]
    additions: ReadOnly[int | None]
    deletions: ReadOnly[int | None]


class ROIPullCommit(TypedDict):
    sha: ReadOnly[str]
    message: ReadOnly[str]
    additions: NotRequired[ReadOnly[int]]
    deletions: NotRequired[ReadOnly[int]]
    changed_files: NotRequired[ReadOnly[int | None]]


class ROIPullEvidence(TypedDict):
    repo: ReadOnly[str]
    number: ReadOnly[int]
    title: ReadOnly[str]
    body: ReadOnly[str]
    url: ReadOnly[str]
    login: ReadOnly[str]
    emails: ReadOnly[tuple[str, ...]]
    profile_email: ReadOnly[str]
    commit_emails: NotRequired[ReadOnly[tuple[str, ...]]]
    merged_at: ReadOnly[str]
    head_sha: ReadOnly[str]
    additions: ReadOnly[int]
    deletions: ReadOnly[int]
    changed_files: ReadOnly[int]
    files: ReadOnly[tuple[ROIPullFile, ...]]
    commits: ReadOnly[tuple[ROIPullCommit, ...]]
    commit_count: ReadOnly[int]
    incomplete_metadata: ReadOnly[bool]


class ROIIdentityMatch(TypedDict):
    email: ReadOnly[str]
    match_method: ReadOnly[str]
    matched: ReadOnly[bool]


class ROIPersonSummary(TypedDict):
    id: ReadOnly[str]
    email: ReadOnly[str]
    logins: ReadOnly[tuple[str, ...]]
    spend: ReadOnly[float | None]
    hours: ReadOnly[float]
    prs: ReadOnly[int]
    estimated_prs: ReadOnly[int]
    pending_prs: ReadOnly[int]
    match_methods: ReadOnly[tuple[str, ...]]
    eligible: ReadOnly[bool]
    cost_per_hour: ReadOnly[float | None]


class ROIPullSummary(TypedDict):
    repo: ReadOnly[str]
    number: ReadOnly[int]
    title: ReadOnly[str]
    url: ReadOnly[str]
    login: ReadOnly[str]
    emails: ReadOnly[tuple[str, ...]]
    profile_email: ReadOnly[str]
    merged_at: ReadOnly[str]
    head_sha: ReadOnly[str]
    additions: ReadOnly[int]
    deletions: ReadOnly[int]
    changed_files: ReadOnly[int]
    commit_count: ReadOnly[int]
    incomplete_metadata: ReadOnly[bool]
    estimate: ReadOnly[ROIEstimate]
    cache_key: ReadOnly[str | None]
    email: ReadOnly[str]
    match_method: ReadOnly[str]
    matched: ReadOnly[bool]


class ROISummaryMetrics(TypedDict):
    matched_spend: ReadOnly[float]
    output_hours: ReadOnly[float]
    total_spend: ReadOnly[float]
    total_output_hours: ReadOnly[float]
    excluded_spend: ReadOnly[float]
    cost_per_hour: ReadOnly[float | None]
    hours_per_dollar: ReadOnly[float | None]
    merged_prs: ReadOnly[int]
    estimated_prs: ReadOnly[int]
    matched_prs: ReadOnly[int]
    cohort_people: ReadOnly[int]
    people_with_prs: ReadOnly[int]
    pending_prs: ReadOnly[int]


class ROITrendDay(TypedDict):
    date: ReadOnly[str]
    spend: ReadOnly[float]
    hours: ReadOnly[float]
    prs: ReadOnly[int]


class ROISummary(TypedDict):
    id: ReadOnly[str | None]
    mode: ReadOnly[str]
    start: ReadOnly[str]
    end: ReadOnly[str]
    synced_at: ReadOnly[str]
    repos: ReadOnly[tuple[str, ...]]
    estimator_model: ReadOnly[str]
    estimator_prompt: ReadOnly[str]
    warnings: ReadOnly[tuple[str, ...]]
    effort_basis: ReadOnly[str | None]
    metrics: ReadOnly[ROISummaryMetrics]
    people: ReadOnly[tuple[ROIPersonSummary, ...]]
    pulls: ReadOnly[tuple[ROIPullSummary, ...]]
    trend: ReadOnly[tuple[ROITrendDay, ...]]


class ROIMetricsResponse(BaseModel):
    matched_spend: float
    output_hours: float
    total_spend: float
    total_output_hours: float
    excluded_spend: float
    cost_per_hour: float | None
    hours_per_dollar: float | None
    merged_prs: int
    estimated_prs: int
    matched_prs: int
    cohort_people: int
    people_with_prs: int
    pending_prs: int


class ROIPersonResponse(BaseModel):
    id: str
    email: str
    logins: tuple[str, ...]
    spend: float | None
    hours: float
    prs: int
    estimated_prs: int
    pending_prs: int
    match_methods: tuple[str, ...]
    eligible: bool
    cost_per_hour: float | None


class ROIEstimateResponse(BaseModel):
    status: Literal["estimated", "needs_review", "error"]
    hours: float | None
    reasoning: str
    model: str | None = None
    evidence_source: str | None = None
    effort_basis: str | None = None
    cached: bool = False


class ROIPullResponse(BaseModel):
    repo: str
    number: int
    title: str
    url: str
    login: str
    emails: tuple[str, ...]
    profile_email: str
    merged_at: str
    head_sha: str
    additions: int
    deletions: int
    changed_files: int
    commit_count: int
    incomplete_metadata: bool
    estimate: ROIEstimateResponse
    cache_key: str | None = None
    email: str
    match_method: str
    matched: bool


class ROITrendResponse(BaseModel):
    date: str
    spend: float
    hours: float
    prs: int


class ROISummaryResponse(BaseModel):
    id: str | None
    mode: str
    start: str
    end: str
    synced_at: str
    repos: tuple[str, ...]
    estimator_model: str
    estimator_prompt: str
    warnings: tuple[str, ...]
    effort_basis: str | None
    metrics: ROIMetricsResponse
    people: tuple[ROIPersonResponse, ...]
    pulls: tuple[ROIPullResponse, ...]
    trend: tuple[ROITrendResponse, ...]


class ROIReportResponse(BaseModel):
    report: ROISummaryResponse | None


class ROIIdentityMapUpdate(BaseModel):
    github_login: str
    email: str | None


class ROIIdentityMapResponse(BaseModel):
    report: ROISummaryResponse | None
    identity_map: Mapping[str, str]


class ROIEstimatorChanges(BaseModel):
    additions: int
    deletions: int
    files: int
    commits: int


class ROIEstimatorFile(BaseModel):
    filename: str | None
    status: str | None
    additions: int | None
    deletions: int | None


class ROIEstimatorCommit(BaseModel):
    sha: str
    message: str
    additions: int | None = None
    deletions: int | None = None
    changed_files: int | None = None


class ROIEstimatorEvidence(BaseModel):
    repo: str
    number: int
    title: str
    body: str
    changes: ROIEstimatorChanges
    files: tuple[ROIEstimatorFile, ...]
    commits: tuple[ROIEstimatorCommit, ...]


class ROICompletionMessage(TypedDict):
    role: ReadOnly[Literal["system", "user"]]
    content: ReadOnly[str]


class ROICompletionMetadata(TypedDict):
    tags: ReadOnly[tuple[str, ...]]
    litellm_roi_estimator: ReadOnly[bool]


class ROIResponseFormat(TypedDict):
    type: ReadOnly[Literal["json_object"]]


class ROICompletionRequest(BaseModel):
    model: str
    temperature: Literal[0]
    messages: tuple[ROICompletionMessage, ...]
    response_format: ROIResponseFormat
    max_tokens: Literal[1200]
    metadata: ROICompletionMetadata
    reasoning_effort: Literal["none"] | None = None


class _ROICompletionMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    content: str | None = None


class _ROICompletionChoice(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    finish_reason: str | None = None
    message: _ROICompletionMessageResponse


class ROICompletionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    choices: tuple[_ROICompletionChoice, ...]


class ROIEstimatorResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    hours: StrictInt | StrictFloat
    reasoning: str

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, value: StrictInt | StrictFloat) -> StrictInt | StrictFloat:
        import math

        if not math.isfinite(value) or value < 0:
            raise ValueError("Hours must be finite and nonnegative.")
        return value

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reasoning must not be empty.")
        return value
