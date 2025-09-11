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
    package="google.cloud.aiplatform.v1beta1.schema.predict.instance",
    manifest={
        "VideoObjectTrackingPredictionInstance",
    },
)


class VideoObjectTrackingPredictionInstance(proto.Message):
    r"""Prediction input format for Video Object Tracking.

    Attributes:
        content (str):
            The Google Cloud Storage location of the
            video on which to perform the prediction.
        mime_type (str):
            The MIME type of the content of the video.
            Only the following are supported: video/mp4
            video/avi video/quicktime
        time_segment_start (str):
            The beginning, inclusive, of the video's time
            segment on which to perform the prediction.
            Expressed as a number of seconds as measured
            from the start of the video, with "s" appended
            at the end. Fractions are allowed, up to a
            microsecond precision.
        time_segment_end (str):
            The end, exclusive, of the video's time
            segment on which to perform the prediction.
            Expressed as a number of seconds as measured
            from the start of the video, with "s" appended
            at the end. Fractions are allowed, up to a
            microsecond precision, and "inf" or "Infinity"
            is allowed, which means the end of the video.
    """

    content: str = proto.Field(
        proto.STRING,
        number=1,
    )
    mime_type: str = proto.Field(
        proto.STRING,
        number=2,
    )
    time_segment_start: str = proto.Field(
        proto.STRING,
        number=3,
    )
    time_segment_end: str = proto.Field(
        proto.STRING,
        number=4,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
