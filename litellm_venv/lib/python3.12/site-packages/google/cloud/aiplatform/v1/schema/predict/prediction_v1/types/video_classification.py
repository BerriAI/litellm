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

from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1.schema.predict.prediction",
    manifest={
        "VideoClassificationPredictionResult",
    },
)


class VideoClassificationPredictionResult(proto.Message):
    r"""Prediction output format for Video Classification.

    Attributes:
        id (str):
            The resource ID of the AnnotationSpec that
            had been identified.
        display_name (str):
            The display name of the AnnotationSpec that
            had been identified.
        type_ (str):
            The type of the prediction. The requested
            types can be configured via parameters. This
            will be one of
            - segment-classification
            - shot-classification
            - one-sec-interval-classification
        time_segment_start (google.protobuf.duration_pb2.Duration):
            The beginning, inclusive, of the video's time
            segment in which the AnnotationSpec has been
            identified. Expressed as a number of seconds as
            measured from the start of the video, with
            fractions up to a microsecond precision, and
            with "s" appended at the end. Note that for
            'segment-classification' prediction type, this
            equals the original 'timeSegmentStart' from the
            input instance, for other types it is the start
            of a shot or a 1 second interval respectively.
        time_segment_end (google.protobuf.duration_pb2.Duration):
            The end, exclusive, of the video's time
            segment in which the AnnotationSpec has been
            identified. Expressed as a number of seconds as
            measured from the start of the video, with
            fractions up to a microsecond precision, and
            with "s" appended at the end. Note that for
            'segment-classification' prediction type, this
            equals the original 'timeSegmentEnd' from the
            input instance, for other types it is the end of
            a shot or a 1 second interval respectively.
        confidence (google.protobuf.wrappers_pb2.FloatValue):
            The Model's confidence in correction of this
            prediction, higher value means higher
            confidence.
    """

    id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    type_: str = proto.Field(
        proto.STRING,
        number=3,
    )
    time_segment_start: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=4,
        message=duration_pb2.Duration,
    )
    time_segment_end: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=5,
        message=duration_pb2.Duration,
    )
    confidence: wrappers_pb2.FloatValue = proto.Field(
        proto.MESSAGE,
        number=6,
        message=wrappers_pb2.FloatValue,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
