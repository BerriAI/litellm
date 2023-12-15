# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import Mapping

from google.ai import generativelanguage as glm
from google.generativeai import string_utils

import typing
from typing import Iterable, Dict, Iterable, List, TypedDict, Union

__all__ = [
    "HarmCategory",
    "HarmProbability",
    "HarmBlockThreshold",
    "BlockedReason",
    "ContentFilterDict",
    "SafetyRatingDict",
    "SafetySettingDict",
    "SafetyFeedbackDict",
]

# These are basic python enums, it's okay to expose them
HarmCategory = glm.HarmCategory
HarmProbability = glm.SafetyRating.HarmProbability
HarmBlockThreshold = glm.SafetySetting.HarmBlockThreshold
BlockedReason = glm.ContentFilter.BlockedReason

HarmCategoryOptions = Union[str, int, HarmCategory]

# fmt: off
_HARM_CATEGORIES: Dict[HarmCategoryOptions, HarmCategory] = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmCategory.HARM_CATEGORY_UNSPECIFIED,
    0: HarmCategory.HARM_CATEGORY_UNSPECIFIED,
    "harm_category_unspecified": HarmCategory.HARM_CATEGORY_UNSPECIFIED,
    "unspecified": HarmCategory.HARM_CATEGORY_UNSPECIFIED,

    HarmCategory.HARM_CATEGORY_DEROGATORY: HarmCategory.HARM_CATEGORY_DEROGATORY,
    1: HarmCategory.HARM_CATEGORY_DEROGATORY,
    "harm_category_derogatory": HarmCategory.HARM_CATEGORY_DEROGATORY,
    "derogatory": HarmCategory.HARM_CATEGORY_DEROGATORY,

    HarmCategory.HARM_CATEGORY_TOXICITY: HarmCategory.HARM_CATEGORY_TOXICITY,
    2: HarmCategory.HARM_CATEGORY_TOXICITY,
    "harm_category_toxicity": HarmCategory.HARM_CATEGORY_TOXICITY,
    "toxicity": HarmCategory.HARM_CATEGORY_TOXICITY,
    "toxic": HarmCategory.HARM_CATEGORY_TOXICITY,

    HarmCategory.HARM_CATEGORY_VIOLENCE: HarmCategory.HARM_CATEGORY_VIOLENCE,
    3: HarmCategory.HARM_CATEGORY_VIOLENCE,
    "harm_category_violence": HarmCategory.HARM_CATEGORY_VIOLENCE,
    "violence": HarmCategory.HARM_CATEGORY_VIOLENCE,
    "violent": HarmCategory.HARM_CATEGORY_VIOLENCE,

    HarmCategory.HARM_CATEGORY_SEXUAL: HarmCategory.HARM_CATEGORY_SEXUAL,
    4: HarmCategory.HARM_CATEGORY_SEXUAL,
    "harm_category_sexual": HarmCategory.HARM_CATEGORY_SEXUAL,
    "sexual": HarmCategory.HARM_CATEGORY_SEXUAL,
    "sex": HarmCategory.HARM_CATEGORY_SEXUAL,

    HarmCategory.HARM_CATEGORY_MEDICAL: HarmCategory.HARM_CATEGORY_MEDICAL,
    5: HarmCategory.HARM_CATEGORY_MEDICAL,
    "harm_category_medical": HarmCategory.HARM_CATEGORY_MEDICAL,
    "medical": HarmCategory.HARM_CATEGORY_MEDICAL,
    "med": HarmCategory.HARM_CATEGORY_MEDICAL,

    HarmCategory.HARM_CATEGORY_DANGEROUS: HarmCategory.HARM_CATEGORY_DANGEROUS,
    6: HarmCategory.HARM_CATEGORY_DANGEROUS,
    "harm_category_dangerous": HarmCategory.HARM_CATEGORY_DANGEROUS,
    "dangerous": HarmCategory.HARM_CATEGORY_DANGEROUS,
    "danger": HarmCategory.HARM_CATEGORY_DANGEROUS,
}
# fmt: on


def to_harm_category(x: HarmCategoryOptions) -> HarmCategory:
    if isinstance(x, str):
        x = x.lower()
    return _HARM_CATEGORIES[x]


HarmBlockThresholdOptions = Union[str, int, HarmBlockThreshold]

# fmt: off
_BLOCK_THRESHOLDS: Dict[HarmBlockThresholdOptions, HarmBlockThreshold] = {
    HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED: HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED,
    0: HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED,
    "harm_block_threshold_unspecified": HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED,
    "block_threshold_unspecified": HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED,
    "unspecified": HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED,

    HarmBlockThreshold.BLOCK_LOW_AND_ABOVE: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    1: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    "block_low_and_above": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    "low": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,

    HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    2: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "block_medium_and_above": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "med": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,

    HarmBlockThreshold.BLOCK_ONLY_HIGH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    3: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    "block_only_high": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    "high": HarmBlockThreshold.BLOCK_ONLY_HIGH,

    HarmBlockThreshold.BLOCK_NONE: HarmBlockThreshold.BLOCK_NONE,
    4: HarmBlockThreshold.BLOCK_NONE,
    "block_none": HarmBlockThreshold.BLOCK_NONE,
}
# fmt: on


def to_block_threshold(x: HarmBlockThresholdOptions) -> HarmCategory:
    if isinstance(x, str):
        x = x.lower()
    return _BLOCK_THRESHOLDS[x]


class ContentFilterDict(TypedDict):
    reason: BlockedReason
    message: str

    __doc__ = string_utils.strip_oneof(glm.ContentFilter.__doc__)


def convert_filters_to_enums(
    filters: Iterable[dict],
) -> List[ContentFilterDict]:
    result = []
    for f in filters:
        f = f.copy()
        f["reason"] = BlockedReason(f["reason"])
        f = typing.cast(ContentFilterDict, f)
        result.append(f)
    return result


class SafetyRatingDict(TypedDict):
    category: HarmCategory
    probability: HarmProbability

    __doc__ = string_utils.strip_oneof(glm.SafetyRating.__doc__)


def convert_rating_to_enum(rating: dict) -> SafetyRatingDict:
    return {
        "category": HarmCategory(rating["category"]),
        "probability": HarmProbability(rating["probability"]),
    }


def convert_ratings_to_enum(ratings: Iterable[dict]) -> List[SafetyRatingDict]:
    result = []
    for r in ratings:
        result.append(convert_rating_to_enum(r))
    return result


class SafetySettingDict(TypedDict):
    category: HarmCategory
    threshold: HarmBlockThreshold

    __doc__ = string_utils.strip_oneof(glm.SafetySetting.__doc__)


class LooseSafetySettingDict(TypedDict):
    category: HarmCategoryOptions
    threshold: HarmBlockThresholdOptions


EasySafetySetting = Mapping[HarmCategoryOptions, HarmBlockThresholdOptions]

SafetySettingOptions = Union[EasySafetySetting, Iterable[LooseSafetySettingDict], None]


def normalize_safety_settings(
    settings: SafetySettingOptions,
) -> list[SafetySettingDict] | None:
    if settings is None:
        return None
    if isinstance(settings, Mapping):
        return [
            {
                "category": to_harm_category(key),
                "threshold": to_block_threshold(value),
            }
            for key, value in settings.items()
        ]
    return [
        {
            "category": to_harm_category(d["category"]),
            "threshold": to_block_threshold(d["threshold"]),
        }
        for d in settings
    ]


def convert_setting_to_enum(setting: dict) -> SafetySettingDict:
    return {
        "category": HarmCategory(setting["category"]),
        "threshold": HarmBlockThreshold(setting["threshold"]),
    }


class SafetyFeedbackDict(TypedDict):
    rating: SafetyRatingDict
    setting: SafetySettingDict

    __doc__ = string_utils.strip_oneof(glm.SafetyFeedback.__doc__)


def convert_safety_feedback_to_enums(
    safety_feedback: Iterable[dict],
) -> List[SafetyFeedbackDict]:
    result = []
    for sf in safety_feedback:
        result.append(
            {
                "rating": convert_rating_to_enum(sf["rating"]),
                "setting": convert_setting_to_enum(sf["setting"]),
            }
        )
    return result


def convert_candidate_enums(candidates):
    result = []
    for candidate in candidates:
        candidate = candidate.copy()
        candidate["safety_ratings"] = convert_ratings_to_enum(candidate["safety_ratings"])
        result.append(candidate)
    return result
