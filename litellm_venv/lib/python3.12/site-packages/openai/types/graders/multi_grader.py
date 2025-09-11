# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Literal, TypeAlias

from ..._models import BaseModel
from .python_grader import PythonGrader
from .label_model_grader import LabelModelGrader
from .score_model_grader import ScoreModelGrader
from .string_check_grader import StringCheckGrader
from .text_similarity_grader import TextSimilarityGrader

__all__ = ["MultiGrader", "Graders"]

Graders: TypeAlias = Union[
    StringCheckGrader,
    TextSimilarityGrader,
    PythonGrader,
    ScoreModelGrader,
    LabelModelGrader,
]


class MultiGrader(BaseModel):
    calculate_output: str
    """A formula to calculate the output based on grader results."""

    graders: Graders
    """
    A StringCheckGrader object that performs a string comparison between input and
    reference using a specified operation.
    """

    name: str
    """The name of the grader."""

    type: Literal["multi"]
    """The object type, which is always `multi`."""
