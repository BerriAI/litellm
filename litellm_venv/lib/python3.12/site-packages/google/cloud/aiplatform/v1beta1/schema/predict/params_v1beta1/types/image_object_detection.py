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
    package="google.cloud.aiplatform.v1beta1.schema.predict.params",
    manifest={
        "ImageObjectDetectionPredictionParams",
    },
)


class ImageObjectDetectionPredictionParams(proto.Message):
    r"""Prediction model parameters for Image Object Detection.

    Attributes:
        confidence_threshold (float):
            The Model only returns predictions with at
            least this confidence score. Default value is
            0.0
        max_predictions (int):
            The Model only returns up to that many top,
            by confidence score, predictions per instance.
            Note that number of returned predictions is also
            limited by metadata's predictionsLimit. Default
            value is 10.
    """

    confidence_threshold: float = proto.Field(
        proto.FLOAT,
        number=1,
    )
    max_predictions: int = proto.Field(
        proto.INT32,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
