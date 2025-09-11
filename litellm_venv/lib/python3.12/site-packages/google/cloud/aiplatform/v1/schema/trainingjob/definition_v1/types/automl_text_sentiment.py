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
    package="google.cloud.aiplatform.v1.schema.trainingjob.definition",
    manifest={
        "AutoMlTextSentiment",
        "AutoMlTextSentimentInputs",
    },
)


class AutoMlTextSentiment(proto.Message):
    r"""A TrainingJob that trains and uploads an AutoML Text
    Sentiment Model.

    Attributes:
        inputs (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTextSentimentInputs):
            The input parameters of this TrainingJob.
    """

    inputs: "AutoMlTextSentimentInputs" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="AutoMlTextSentimentInputs",
    )


class AutoMlTextSentimentInputs(proto.Message):
    r"""

    Attributes:
        sentiment_max (int):
            A sentiment is expressed as an integer
            ordinal, where higher value means a more
            positive sentiment. The range of sentiments that
            will be used is between 0 and sentimentMax
            (inclusive on both ends), and all the values in
            the range must be represented in the dataset
            before a model can be created.
            Only the Annotations with this sentimentMax will
            be used for training. sentimentMax value must be
            between 1 and 10 (inclusive).
    """

    sentiment_max: int = proto.Field(
        proto.INT32,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
