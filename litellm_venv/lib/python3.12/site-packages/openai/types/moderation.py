# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = ["Moderation", "Categories", "CategoryAppliedInputTypes", "CategoryScores"]


class Categories(BaseModel):
    harassment: bool
    """
    Content that expresses, incites, or promotes harassing language towards any
    target.
    """

    harassment_threatening: bool = FieldInfo(alias="harassment/threatening")
    """
    Harassment content that also includes violence or serious harm towards any
    target.
    """

    hate: bool
    """
    Content that expresses, incites, or promotes hate based on race, gender,
    ethnicity, religion, nationality, sexual orientation, disability status, or
    caste. Hateful content aimed at non-protected groups (e.g., chess players) is
    harassment.
    """

    hate_threatening: bool = FieldInfo(alias="hate/threatening")
    """
    Hateful content that also includes violence or serious harm towards the targeted
    group based on race, gender, ethnicity, religion, nationality, sexual
    orientation, disability status, or caste.
    """

    illicit: Optional[bool] = None
    """
    Content that includes instructions or advice that facilitate the planning or
    execution of wrongdoing, or that gives advice or instruction on how to commit
    illicit acts. For example, "how to shoplift" would fit this category.
    """

    illicit_violent: Optional[bool] = FieldInfo(alias="illicit/violent", default=None)
    """
    Content that includes instructions or advice that facilitate the planning or
    execution of wrongdoing that also includes violence, or that gives advice or
    instruction on the procurement of any weapon.
    """

    self_harm: bool = FieldInfo(alias="self-harm")
    """
    Content that promotes, encourages, or depicts acts of self-harm, such as
    suicide, cutting, and eating disorders.
    """

    self_harm_instructions: bool = FieldInfo(alias="self-harm/instructions")
    """
    Content that encourages performing acts of self-harm, such as suicide, cutting,
    and eating disorders, or that gives instructions or advice on how to commit such
    acts.
    """

    self_harm_intent: bool = FieldInfo(alias="self-harm/intent")
    """
    Content where the speaker expresses that they are engaging or intend to engage
    in acts of self-harm, such as suicide, cutting, and eating disorders.
    """

    sexual: bool
    """
    Content meant to arouse sexual excitement, such as the description of sexual
    activity, or that promotes sexual services (excluding sex education and
    wellness).
    """

    sexual_minors: bool = FieldInfo(alias="sexual/minors")
    """Sexual content that includes an individual who is under 18 years old."""

    violence: bool
    """Content that depicts death, violence, or physical injury."""

    violence_graphic: bool = FieldInfo(alias="violence/graphic")
    """Content that depicts death, violence, or physical injury in graphic detail."""


class CategoryAppliedInputTypes(BaseModel):
    harassment: List[Literal["text"]]
    """The applied input type(s) for the category 'harassment'."""

    harassment_threatening: List[Literal["text"]] = FieldInfo(
        alias="harassment/threatening"
    )
    """The applied input type(s) for the category 'harassment/threatening'."""

    hate: List[Literal["text"]]
    """The applied input type(s) for the category 'hate'."""

    hate_threatening: List[Literal["text"]] = FieldInfo(alias="hate/threatening")
    """The applied input type(s) for the category 'hate/threatening'."""

    illicit: List[Literal["text"]]
    """The applied input type(s) for the category 'illicit'."""

    illicit_violent: List[Literal["text"]] = FieldInfo(alias="illicit/violent")
    """The applied input type(s) for the category 'illicit/violent'."""

    self_harm: List[Literal["text", "image"]] = FieldInfo(alias="self-harm")
    """The applied input type(s) for the category 'self-harm'."""

    self_harm_instructions: List[Literal["text", "image"]] = FieldInfo(
        alias="self-harm/instructions"
    )
    """The applied input type(s) for the category 'self-harm/instructions'."""

    self_harm_intent: List[Literal["text", "image"]] = FieldInfo(
        alias="self-harm/intent"
    )
    """The applied input type(s) for the category 'self-harm/intent'."""

    sexual: List[Literal["text", "image"]]
    """The applied input type(s) for the category 'sexual'."""

    sexual_minors: List[Literal["text"]] = FieldInfo(alias="sexual/minors")
    """The applied input type(s) for the category 'sexual/minors'."""

    violence: List[Literal["text", "image"]]
    """The applied input type(s) for the category 'violence'."""

    violence_graphic: List[Literal["text", "image"]] = FieldInfo(
        alias="violence/graphic"
    )
    """The applied input type(s) for the category 'violence/graphic'."""


class CategoryScores(BaseModel):
    harassment: float
    """The score for the category 'harassment'."""

    harassment_threatening: float = FieldInfo(alias="harassment/threatening")
    """The score for the category 'harassment/threatening'."""

    hate: float
    """The score for the category 'hate'."""

    hate_threatening: float = FieldInfo(alias="hate/threatening")
    """The score for the category 'hate/threatening'."""

    illicit: float
    """The score for the category 'illicit'."""

    illicit_violent: float = FieldInfo(alias="illicit/violent")
    """The score for the category 'illicit/violent'."""

    self_harm: float = FieldInfo(alias="self-harm")
    """The score for the category 'self-harm'."""

    self_harm_instructions: float = FieldInfo(alias="self-harm/instructions")
    """The score for the category 'self-harm/instructions'."""

    self_harm_intent: float = FieldInfo(alias="self-harm/intent")
    """The score for the category 'self-harm/intent'."""

    sexual: float
    """The score for the category 'sexual'."""

    sexual_minors: float = FieldInfo(alias="sexual/minors")
    """The score for the category 'sexual/minors'."""

    violence: float
    """The score for the category 'violence'."""

    violence_graphic: float = FieldInfo(alias="violence/graphic")
    """The score for the category 'violence/graphic'."""


class Moderation(BaseModel):
    categories: Categories
    """A list of the categories, and whether they are flagged or not."""

    category_applied_input_types: CategoryAppliedInputTypes
    """
    A list of the categories along with the input type(s) that the score applies to.
    """

    category_scores: CategoryScores
    """A list of the categories along with their scores as predicted by model."""

    flagged: bool
    """Whether any of the below categories are flagged."""
