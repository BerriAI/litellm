"""
Type definitions for OpenAI Evals API
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Required, TypedDict


# Evals API Request Types
class DataSourceConfigCustom(TypedDict, total=False):
    """Data source configuration for custom data sources"""

    type: Required[Literal["custom"]]
    """Data source type - custom"""

    item_schema: Required[Dict[str, Any]]
    """JSON schema describing the structure of each row in the dataset"""

    include_sample_schema: Optional[bool]
    """Whether eval expects sample schema population"""


class DataSourceConfigLogs(TypedDict, total=False):
    """Data source configuration for logs-based evals"""

    type: Required[Literal["logs"]]
    """Data source type - logs"""

    metadata: Optional[Dict[str, Any]]
    """Optional metadata for filtering logs"""


class DataSourceConfigStoredCompletions(TypedDict, total=False):
    """Data source configuration for stored completions (deprecated)"""

    type: Required[Literal["stored_completions"]]
    """Data source type - stored_completions (deprecated)"""

    metadata: Optional[Dict[str, Any]]
    """Optional metadata for filtering stored completions"""


DataSourceConfig = Union[DataSourceConfigCustom, DataSourceConfigLogs, DataSourceConfigStoredCompletions]


class LLMAsJudgeGraderConfig(TypedDict, total=False):
    """Configuration for LLM as judge grading"""

    type: Required[Literal["llm_as_judge"]]
    """Grader type - LLM as judge"""

    model: Optional[str]
    """Model to use as judge (e.g., 'gpt-4')"""

    prompt: Optional[str]
    """Custom prompt for the judge model"""


class GroundTruthGraderConfig(TypedDict, total=False):
    """Configuration for ground truth grading"""

    type: Required[Literal["ground_truth"]]
    """Grader type - ground truth comparison"""

    metric: Optional[Literal["exact_match", "f1_score", "bleu"]]
    """Metric to use for comparison"""


class CustomGraderConfig(TypedDict, total=False):
    """Configuration for custom grading function"""

    type: Required[Literal["custom"]]
    """Grader type - custom"""

    function_id: Required[str]
    """ID of the custom grading function"""


GraderConfig = Union[LLMAsJudgeGraderConfig, GroundTruthGraderConfig, CustomGraderConfig]


class CreateEvalRequest(TypedDict, total=False):
    """Request parameters for creating an evaluation"""

    name: Optional[str]
    """The name of the evaluation"""

    data_source_config: Required[DataSourceConfig]
    """Configuration for the data source"""

    testing_criteria: Required[List[GraderConfig]]
    """List of graders for all eval runs"""

    metadata: Optional[Dict[str, Any]]
    """Set of 16 key-value pairs that can be attached to an object (max 64 char keys, 512 char values)"""


class UpdateEvalRequest(TypedDict, total=False):
    """Request parameters for updating an evaluation"""

    name: Optional[str]
    """Updated name"""

    metadata: Optional[Dict[str, Any]]
    """Updated metadata"""


class ListEvalsParams(TypedDict, total=False):
    """Query parameters for listing evaluations"""

    limit: Optional[int]
    """Number of results to return per page. Maximum value is 100. Defaults to 20."""

    after: Optional[str]
    """Cursor for pagination - returns evals after this ID"""

    before: Optional[str]
    """Cursor for pagination - returns evals before this ID"""

    order: Optional[Literal["asc", "desc"]]
    """Sort order for results. Defaults to 'desc'."""

    order_by: Optional[Literal["created_at", "updated_at"]]
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

    updated_at: Optional[int] = None
    """Unix timestamp of when the evaluation was last updated"""

    name: Optional[str] = None
    """The name of the evaluation"""

    data_source_config: Dict[str, Any]
    """Configuration for the data source"""

    testing_criteria: List[Dict[str, Any]]
    """List of graders for the evaluation"""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata"""


class ListEvalsResponse(BaseModel):
    """Response from listing evaluations"""

    object: str = "list"
    """Object type, always 'list'"""

    data: List[Eval]
    """List of evaluations"""

    first_id: Optional[str] = None
    """ID of the first evaluation in the list"""

    last_id: Optional[str] = None
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
