# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Union, Optional
from typing_extensions import Literal, Annotated, TypeAlias

from pydantic import Field as FieldInfo

from .._utils import PropertyInfo
from .._models import BaseModel
from .shared.metadata import Metadata
from .graders.python_grader import PythonGrader
from .graders.label_model_grader import LabelModelGrader
from .graders.score_model_grader import ScoreModelGrader
from .graders.string_check_grader import StringCheckGrader
from .eval_custom_data_source_config import EvalCustomDataSourceConfig
from .graders.text_similarity_grader import TextSimilarityGrader
from .eval_stored_completions_data_source_config import (
    EvalStoredCompletionsDataSourceConfig,
)

__all__ = [
    "EvalUpdateResponse",
    "DataSourceConfig",
    "DataSourceConfigLogs",
    "TestingCriterion",
    "TestingCriterionEvalGraderTextSimilarity",
    "TestingCriterionEvalGraderPython",
    "TestingCriterionEvalGraderScoreModel",
]


class DataSourceConfigLogs(BaseModel):
    schema_: Dict[str, object] = FieldInfo(alias="schema")
    """
    The json schema for the run data source items. Learn how to build JSON schemas
    [here](https://json-schema.org/).
    """

    type: Literal["logs"]
    """The type of data source. Always `logs`."""

    metadata: Optional[Metadata] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """


DataSourceConfig: TypeAlias = Annotated[
    Union[
        EvalCustomDataSourceConfig,
        DataSourceConfigLogs,
        EvalStoredCompletionsDataSourceConfig,
    ],
    PropertyInfo(discriminator="type"),
]


class TestingCriterionEvalGraderTextSimilarity(TextSimilarityGrader):
    __test__ = False
    pass_threshold: float
    """The threshold for the score."""


class TestingCriterionEvalGraderPython(PythonGrader):
    __test__ = False
    pass_threshold: Optional[float] = None
    """The threshold for the score."""


class TestingCriterionEvalGraderScoreModel(ScoreModelGrader):
    __test__ = False
    pass_threshold: Optional[float] = None
    """The threshold for the score."""


TestingCriterion: TypeAlias = Union[
    LabelModelGrader,
    StringCheckGrader,
    TestingCriterionEvalGraderTextSimilarity,
    TestingCriterionEvalGraderPython,
    TestingCriterionEvalGraderScoreModel,
]


class EvalUpdateResponse(BaseModel):
    id: str
    """Unique identifier for the evaluation."""

    created_at: int
    """The Unix timestamp (in seconds) for when the eval was created."""

    data_source_config: DataSourceConfig
    """Configuration of data sources used in runs of the evaluation."""

    metadata: Optional[Metadata] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    name: str
    """The name of the evaluation."""

    object: Literal["eval"]
    """The object type."""

    testing_criteria: List[TestingCriterion]
    """A list of testing criteria."""
