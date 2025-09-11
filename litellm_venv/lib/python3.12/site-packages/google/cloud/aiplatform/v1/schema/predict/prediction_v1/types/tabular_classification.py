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
        "TabularClassificationPredictionResult",
    },
)


class TabularClassificationPredictionResult(proto.Message):
    r"""Prediction output format for Tabular Classification.

    Attributes:
        classes (MutableSequence[str]):
            The name of the classes being classified,
            contains all possible values of the target
            column.
        scores (MutableSequence[float]):
            The model's confidence in each class being
            correct, higher value means higher confidence.
            The N-th score corresponds to the N-th class in
            classes.
    """

    classes: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    scores: MutableSequence[float] = proto.RepeatedField(
        proto.FLOAT,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
