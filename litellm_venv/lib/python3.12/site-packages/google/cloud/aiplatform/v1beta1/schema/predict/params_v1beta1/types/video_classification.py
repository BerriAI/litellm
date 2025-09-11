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
        "VideoClassificationPredictionParams",
    },
)


class VideoClassificationPredictionParams(proto.Message):
    r"""Prediction model parameters for Video Classification.

    Attributes:
        confidence_threshold (float):
            The Model only returns predictions with at
            least this confidence score. Default value is
            0.0
        max_predictions (int):
            The Model only returns up to that many top,
            by confidence score, predictions per instance.
            If this number is very high, the Model may
            return fewer predictions. Default value is
            10,000.
        segment_classification (bool):
            Set to true to request segment-level
            classification. Vertex AI returns labels and
            their confidence scores for the entire time
            segment of the video that user specified in the
            input instance. Default value is true
        shot_classification (bool):
            Set to true to request shot-level
            classification. Vertex AI determines the
            boundaries for each camera shot in the entire
            time segment of the video that user specified in
            the input instance. Vertex AI then returns
            labels and their confidence scores for each
            detected shot, along with the start and end time
            of the shot.
            WARNING: Model evaluation is not done for this
            classification type, the quality of it depends
            on the training data, but there are no metrics
            provided to describe that quality.
            Default value is false
        one_sec_interval_classification (bool):
            Set to true to request classification for a
            video at one-second intervals. Vertex AI returns
            labels and their confidence scores for each
            second of the entire time segment of the video
            that user specified in the input WARNING: Model
            evaluation is not done for this classification
            type, the quality of it depends on the training
            data, but there are no metrics provided to
            describe that quality. Default value is false
    """

    confidence_threshold: float = proto.Field(
        proto.FLOAT,
        number=1,
    )
    max_predictions: int = proto.Field(
        proto.INT32,
        number=2,
    )
    segment_classification: bool = proto.Field(
        proto.BOOL,
        number=3,
    )
    shot_classification: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    one_sec_interval_classification: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
