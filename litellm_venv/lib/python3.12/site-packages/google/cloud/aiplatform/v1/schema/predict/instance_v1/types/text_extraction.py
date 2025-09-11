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
    package="google.cloud.aiplatform.v1.schema.predict.instance",
    manifest={
        "TextExtractionPredictionInstance",
    },
)


class TextExtractionPredictionInstance(proto.Message):
    r"""Prediction input format for Text Extraction.

    Attributes:
        content (str):
            The text snippet to make the predictions on.
        mime_type (str):
            The MIME type of the text snippet. The
            supported MIME types are listed below.
            - text/plain
        key (str):
            This field is only used for batch prediction.
            If a key is provided, the batch prediction
            result will by mapped to this key. If omitted,
            then the batch prediction result will contain
            the entire input instance. Vertex AI will not
            check if keys in the request are duplicates, so
            it is up to the caller to ensure the keys are
            unique.
    """

    content: str = proto.Field(
        proto.STRING,
        number=1,
    )
    mime_type: str = proto.Field(
        proto.STRING,
        number=2,
    )
    key: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
