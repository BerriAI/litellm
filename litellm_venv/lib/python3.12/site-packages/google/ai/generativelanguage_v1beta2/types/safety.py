# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
#
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta2",
    manifest={
        "HarmCategory",
        "ContentFilter",
        "SafetyFeedback",
        "SafetyRating",
        "SafetySetting",
    },
)


class HarmCategory(proto.Enum):
    r"""The category of a rating.

    These categories cover various kinds of harms that developers
    may wish to adjust.

    Values:
        HARM_CATEGORY_UNSPECIFIED (0):
            Category is unspecified.
        HARM_CATEGORY_DEROGATORY (1):
            Negative or harmful comments targeting
            identity and/or protected attribute.
        HARM_CATEGORY_TOXICITY (2):
            Content that is rude, disrepspectful, or
            profane.
        HARM_CATEGORY_VIOLENCE (3):
            Describes scenarios depictng violence against
            an individual or group, or general descriptions
            of gore.
        HARM_CATEGORY_SEXUAL (4):
            Contains references to sexual acts or other
            lewd content.
        HARM_CATEGORY_MEDICAL (5):
            Promotes unchecked medical advice.
        HARM_CATEGORY_DANGEROUS (6):
            Dangerous content that promotes, facilitates,
            or encourages harmful acts.
    """
    HARM_CATEGORY_UNSPECIFIED = 0
    HARM_CATEGORY_DEROGATORY = 1
    HARM_CATEGORY_TOXICITY = 2
    HARM_CATEGORY_VIOLENCE = 3
    HARM_CATEGORY_SEXUAL = 4
    HARM_CATEGORY_MEDICAL = 5
    HARM_CATEGORY_DANGEROUS = 6


class ContentFilter(proto.Message):
    r"""Content filtering metadata associated with processing a
    single request.
    ContentFilter contains a reason and an optional supporting
    string. The reason may be unspecified.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        reason (google.ai.generativelanguage_v1beta2.types.ContentFilter.BlockedReason):
            The reason content was blocked during request
            processing.
        message (str):
            A string that describes the filtering
            behavior in more detail.

            This field is a member of `oneof`_ ``_message``.
    """

    class BlockedReason(proto.Enum):
        r"""A list of reasons why content may have been blocked.

        Values:
            BLOCKED_REASON_UNSPECIFIED (0):
                A blocked reason was not specified.
            SAFETY (1):
                Content was blocked by safety settings.
            OTHER (2):
                Content was blocked, but the reason is
                uncategorized.
        """
        BLOCKED_REASON_UNSPECIFIED = 0
        SAFETY = 1
        OTHER = 2

    reason: BlockedReason = proto.Field(
        proto.ENUM,
        number=1,
        enum=BlockedReason,
    )
    message: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class SafetyFeedback(proto.Message):
    r"""Safety feedback for an entire request.

    This field is populated if content in the input and/or response
    is blocked due to safety settings. SafetyFeedback may not exist
    for every HarmCategory. Each SafetyFeedback will return the
    safety settings used by the request as well as the lowest
    HarmProbability that should be allowed in order to return a
    result.

    Attributes:
        rating (google.ai.generativelanguage_v1beta2.types.SafetyRating):
            Safety rating evaluated from content.
        setting (google.ai.generativelanguage_v1beta2.types.SafetySetting):
            Safety settings applied to the request.
    """

    rating: "SafetyRating" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SafetyRating",
    )
    setting: "SafetySetting" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SafetySetting",
    )


class SafetyRating(proto.Message):
    r"""Safety rating for a piece of content.

    The safety rating contains the category of harm and the harm
    probability level in that category for a piece of content.
    Content is classified for safety across a number of harm
    categories and the probability of the harm classification is
    included here.

    Attributes:
        category (google.ai.generativelanguage_v1beta2.types.HarmCategory):
            Required. The category for this rating.
        probability (google.ai.generativelanguage_v1beta2.types.SafetyRating.HarmProbability):
            Required. The probability of harm for this
            content.
    """

    class HarmProbability(proto.Enum):
        r"""The probability that a piece of content is harmful.

        The classification system gives the probability of the content
        being unsafe. This does not indicate the severity of harm for a
        piece of content.

        Values:
            HARM_PROBABILITY_UNSPECIFIED (0):
                Probability is unspecified.
            NEGLIGIBLE (1):
                Content has a negligible chance of being
                unsafe.
            LOW (2):
                Content has a low chance of being unsafe.
            MEDIUM (3):
                Content has a medium chance of being unsafe.
            HIGH (4):
                Content has a high chance of being unsafe.
        """
        HARM_PROBABILITY_UNSPECIFIED = 0
        NEGLIGIBLE = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4

    category: "HarmCategory" = proto.Field(
        proto.ENUM,
        number=3,
        enum="HarmCategory",
    )
    probability: HarmProbability = proto.Field(
        proto.ENUM,
        number=4,
        enum=HarmProbability,
    )


class SafetySetting(proto.Message):
    r"""Safety setting, affecting the safety-blocking behavior.

    Passing a safety setting for a category changes the allowed
    proability that content is blocked.

    Attributes:
        category (google.ai.generativelanguage_v1beta2.types.HarmCategory):
            Required. The category for this setting.
        threshold (google.ai.generativelanguage_v1beta2.types.SafetySetting.HarmBlockThreshold):
            Required. Controls the probability threshold
            at which harm is blocked.
    """

    class HarmBlockThreshold(proto.Enum):
        r"""Block at and beyond a specified harm probability.

        Values:
            HARM_BLOCK_THRESHOLD_UNSPECIFIED (0):
                Threshold is unspecified.
            BLOCK_LOW_AND_ABOVE (1):
                Content with NEGLIGIBLE will be allowed.
            BLOCK_MEDIUM_AND_ABOVE (2):
                Content with NEGLIGIBLE and LOW will be
                allowed.
            BLOCK_ONLY_HIGH (3):
                Content with NEGLIGIBLE, LOW, and MEDIUM will
                be allowed.
        """
        HARM_BLOCK_THRESHOLD_UNSPECIFIED = 0
        BLOCK_LOW_AND_ABOVE = 1
        BLOCK_MEDIUM_AND_ABOVE = 2
        BLOCK_ONLY_HIGH = 3

    category: "HarmCategory" = proto.Field(
        proto.ENUM,
        number=3,
        enum="HarmCategory",
    )
    threshold: HarmBlockThreshold = proto.Field(
        proto.ENUM,
        number=4,
        enum=HarmBlockThreshold,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
