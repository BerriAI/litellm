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
        "VideoObjectTrackingPredictionParams",
    },
)


class VideoObjectTrackingPredictionParams(proto.Message):
    r"""Prediction model parameters for Video Object Tracking.

    Attributes:
        confidence_threshold (float):
            The Model only returns predictions with at
            least this confidence score. Default value is
            0.0
        max_predictions (int):
            The model only returns up to that many top,
            by confidence score, predictions per frame of
            the video. If this number is very high, the
            Model may return fewer predictions per frame.
            Default value is 50.
        min_bounding_box_size (float):
            Only bounding boxes with shortest edge at
            least that long as a relative value of video
            frame size are returned. Default value is 0.0.
    """

    confidence_threshold: float = proto.Field(
        proto.FLOAT,
        number=1,
    )
    max_predictions: int = proto.Field(
        proto.INT32,
        number=2,
    )
    min_bounding_box_size: float = proto.Field(
        proto.FLOAT,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
