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
    package="google.cloud.aiplatform.v1.schema.predict.prediction",
    manifest={
        "TextSentimentPredictionResult",
    },
)


class TextSentimentPredictionResult(proto.Message):
    r"""Prediction output format for Text Sentiment

    Attributes:
        sentiment (int):
            The integer sentiment labels between 0
            (inclusive) and sentimentMax label (inclusive),
            while 0 maps to the least positive sentiment and
            sentimentMax maps to the most positive one. The
            higher the score is, the more positive the
            sentiment in the text snippet is. Note:
            sentimentMax is an integer value between 1
            (inclusive) and 10 (inclusive).
    """

    sentiment: int = proto.Field(
        proto.INT32,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
