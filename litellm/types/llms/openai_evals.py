"""
Type definitions for OpenAI Evals API
"""

from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import Required, TypedDict


# Evals API Request Types
class DataSourceConfigCustom(TypedDict, total=False):
    """Data source configuration for custom data sources"""

    type: Required[Literal["custom"]]
    """Data source type - custom"""

    item_schema: Required[dict[str, Any]]
    """JSON schema describing the structure of each row in the dataset"""

    include_sample_schema: bool | None
    """Whether eval expects sample schema population"""


class DataSourceConfigLogs(TypedDict, total=False):
    """Data source configuration for logs-based evals"""

    type: Required[Literal["logs"]]
    """Data source type - logs"""

    metadata: dict[str, Any] | None
    """Optional metadata for filtering logs"""


class DataSourceConfigStoredCompletions(TypedDict, total=False):
    """Data source configuration for stored completions (deprecated)"""

    type: Required[Literal["stored_completions"]]
    """Data source type - stored_completions (deprecated)"""

    metadata: dict[str, Any] | None
    """Optional metadata for filtering stored completions"""


DataSourceConfig = DataSourceConfigCustom | DataSourceConfigLogs | DataSourceConfigStoredCompletions


class LLMAsJudgeGraderConfig(TypedDict, total=False):
    """Configuration for LLM as judge grading"""

    type: Required[Literal["llm_as_judge"]]
    """Grader type - LLM as judge"""

    model: str | None
    """Model to use as judge (e.g., 'gpt-4')"""

    prompt: str | None
    """Custom prompt for the judge model"""


class GroundTruthGraderConfig(TypedDict, total=False):
    """Configuration for ground truth grading"""

    type: Required[Literal["ground_truth"]]
    """Grader type - ground truth comparison"""

    metric: Literal["exact_match", "f1_score", "bleu"] | None
    """Metric to use for comparison"""


class CustomGraderConfig(TypedDict, total=False):
    """Configuration for custom grading function"""

    type: Required[Literal["custom"]]
    """Grader type - custom"""

    function_id: Required[str]
    """ID of the custom grading function"""


GraderConfig = LLMAsJudgeGraderConfig | GroundTruthGraderConfig | CustomGraderConfig


class CreateEvalRequest(TypedDict, total=False):
    """Request parameters for creating an evaluation"""

    name: str | None
    """The name of the evaluation"""

    data_source_config: Required[DataSourceConfig]
    """Configuration for the data source"""

    testing_criteria: Required[list[GraderConfig]]
    """List of graders for all eval runs"""

    metadata: dict[str, Any] | None
    """Set of 16 key-value pairs that can be attached to an object (max 64 char keys, 512 char values)"""


class UpdateEvalRequest(TypedDict, total=False):
    """Request parameters for updating an evaluation"""

    name: str | None
    """Updated name"""

    metadata: dict[str, Any] | None
    """Updated metadata"""


class ListEvalsParams(TypedDict, total=False):
    """Query parameters for listing evaluations"""

    limit: int | None
    """Number of results to return per page. Maximum value is 100. Defaults to 20."""

    after: str | None
    """Cursor for pagination - returns evals after this ID"""

    before: str | None
    """Cursor for pagination - returns evals before this ID"""

    order: Literal["asc", "desc"] | None
    """Sort order for results. Defaults to 'desc'."""

    order_by: Literal["created_at", "updated_at"] | None
    """Field to sort by. Defaults to 'created_at'."""


# Evals API Response Types
class Eval(BaseModel):
    """Represents an evaluation from the OpenAI Evals API"""

    id: str
    """Unique identifier for the evaluation"""

    object: str = "eval"
    """Object type, always 'eval'"""

    created_at: int
    """Unix timestamp of when the evaluation was created"""

    updated_at: int | None = None
    """Unix timestamp of when the evaluation was last updated"""

    name: str | None = None
    """The name of the evaluation"""

    data_source_config: dict[str, Any]
    """Configuration for the data source"""

    testing_criteria: list[dict[str, Any]]
    """List of graders for the evaluation"""

    metadata: dict[str, Any] | None = None
    """Additional metadata"""


class ListEvalsResponse(BaseModel):
    """Response from listing evaluations"""

    object: str = "list"
    """Object type, always 'list'"""

    data: list[Eval]
    """List of evaluations"""

    first_id: str | None = None
    """ID of the first evaluation in the list"""

    last_id: str | None = None
    """ID of the last evaluation in the list"""

    has_more: bool = False
    """Whether there are more evaluations available"""


class DeleteEvalResponse(BaseModel):
    """Response from deleting an evaluation"""

    eval_id: str
    """The ID of the deleted evaluation"""

    object: str = "eval.deleted"
    """Object type, always 'eval.deleted'"""

    deleted: bool
    """Whether the evaluation was successfully deleted"""


class CancelEvalResponse(BaseModel):
    """Response from cancelling an evaluation"""

    id: str
    """The ID of the cancelled evaluation"""

    object: str = "eval"
    """Object type, always 'eval'"""

    status: Literal["cancelled"]
    """Status after cancellation, always 'cancelled'"""


# Run API Request Types
class DataSourceDatasetConfig(TypedDict, total=False):
    """Data source configuration for dataset-based runs"""

    type: Required[Literal["dataset"]]
    """Data source type - dataset"""

    dataset_id: Required[str]
    """ID of the dataset to use for the run"""


class DataSourceSampleSetConfig(TypedDict, total=False):
    """Data source configuration for sample set-based runs"""

    type: Required[Literal["sample_set"]]
    """Data source type - sample_set"""

    sample_set_id: Required[str]
    """ID of the sample set to use for the run"""


class DataSourceInlineConfig(TypedDict, total=False):
    """Data source configuration for inline samples"""

    type: Required[Literal["inline"]]
    """Data source type - inline"""

    samples: Required[list[dict[str, Any]]]
    """List of inline samples to use for the run"""


RunDataSourceConfig = DataSourceDatasetConfig | DataSourceSampleSetConfig | DataSourceInlineConfig


class CompletionConfig(TypedDict, total=False):
    """Configuration for model completions in a run"""

    model: Required[str]
    """Model to use for completions"""

    temperature: float | None
    """Sampling temperature (0-2)"""

    max_tokens: int | None
    """Maximum tokens to generate"""

    top_p: float | None
    """Nucleus sampling parameter"""

    frequency_penalty: float | None
    """Frequency penalty (-2.0 to 2.0)"""

    presence_penalty: float | None
    """Presence penalty (-2.0 to 2.0)"""


class CreateRunRequest(TypedDict, total=False):
    """Request parameters for creating a run"""

    data_source: Required[dict[str, Any]]
    """Data source configuration for the run (can be jsonl, completions, or responses type)"""

    name: str | None
    """Optional name for the run"""

    metadata: dict[str, Any] | None
    """Optional metadata for the run"""


class ListRunsParams(TypedDict, total=False):
    """Query parameters for listing runs"""

    limit: int | None
    """Number of results to return per page. Maximum value is 100. Defaults to 20."""

    after: str | None
    """Cursor for pagination - returns runs after this ID"""

    before: str | None
    """Cursor for pagination - returns runs before this ID"""

    order: Literal["asc", "desc"] | None
    """Sort order for results. Defaults to 'desc'."""


# Run API Response Types
class ResultCounts(BaseModel):
    """Result counts for a run"""

    total: int
    """Total number of results"""

    passed: int = 0
    """Number of passed results"""

    failed: int = 0
    """Number of failed results"""

    error: int = 0
    """Number of error results"""


class PerTestingCriteriaResult(BaseModel):
    """Results for a specific testing criteria"""

    testing_criteria_index: int
    """Index of the testing criteria"""

    result_counts: ResultCounts
    """Result counts for this criteria"""

    average_score: float | None = None
    """Average score for this criteria"""


class Run(BaseModel):
    """Represents a run from the OpenAI Evals API"""

    id: str
    """Unique identifier for the run"""

    object: str = "eval.run"
    """Object type, always 'eval.run'"""

    created_at: int
    """Unix timestamp of when the run was created"""

    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    """Current status of the run"""

    data_source: dict[str, Any]
    """Data source configuration used for the run"""

    eval_id: str
    """ID of the evaluation this run belongs to"""

    name: str | None = None
    """Name of the run"""

    started_at: int | None = None
    """Unix timestamp of when the run started"""

    completed_at: int | None = None
    """Unix timestamp of when the run completed"""

    model: str | None = None
    """Model used for the run, if any"""

    per_model_usage: Any | None = None
    """Model usage details per model, if available"""

    per_testing_criteria_results: list[PerTestingCriteriaResult] | None = None
    """Per-criteria results"""

    report_url: str | None = None
    """URL for the evaluation report"""

    result_counts: dict[str, int] | None = None
    """Aggregate result counts (e.g., {"passed": 0, "failed": 0, "errored": 0, "total": 0})"""

    shared_with_openai: bool | None = None
    """Whether run is shared with OpenAI"""

    metadata: dict[str, Any] | None = None
    """Additional metadata"""

    error: dict[str, Any] | None = None
    """Error details if the run failed"""


class ListRunsResponse(BaseModel):
    """Response from listing runs"""

    object: str = "list"
    """Object type, always 'list'"""

    data: list[Run]
    """List of runs"""

    first_id: str | None = None
    """ID of the first run in the list"""

    last_id: str | None = None
    """ID of the last run in the list"""

    has_more: bool = False
    """Whether there are more runs available"""


class CancelRunResponse(BaseModel):
    """Response from cancelling a run"""

    id: str
    """The ID of the cancelled run"""

    object: str = "eval.run"
    """Object type, always 'eval.run'"""

    status: Literal["cancelled"]
    """Status after cancellation, always 'cancelled'"""


class RunDeleteResponse(BaseModel):
    """Response from deleting a run"""

    run_id: str
    """The ID of the deleted run"""

    object: str | None = "eval.run.deleted"
    """Object type, always 'eval.run.deleted'"""

    deleted: bool | None = True
    """Whether the run was successfully deleted"""
