# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import Literal, Required, TypeAlias, TypedDict

from .python_grader_param import PythonGraderParam
from .label_model_grader_param import LabelModelGraderParam
from .score_model_grader_param import ScoreModelGraderParam
from .string_check_grader_param import StringCheckGraderParam
from .text_similarity_grader_param import TextSimilarityGraderParam

__all__ = ["MultiGraderParam", "Graders"]

Graders: TypeAlias = Union[
    StringCheckGraderParam,
    TextSimilarityGraderParam,
    PythonGraderParam,
    ScoreModelGraderParam,
    LabelModelGraderParam,
]


class MultiGraderParam(TypedDict, total=False):
    calculate_output: Required[str]
    """A formula to calculate the output based on grader results."""

    graders: Required[Graders]
    """
    A StringCheckGrader object that performs a string comparison between input and
    reference using a specified operation.
    """

    name: Required[str]
    """The name of the grader."""

    type: Required[Literal["multi"]]
    """The object type, which is always `multi`."""
