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
    package="google.cloud.aiplatform.v1beta1.schema.predict.prediction",
    manifest={
        "ImageSegmentationPredictionResult",
    },
)


class ImageSegmentationPredictionResult(proto.Message):
    r"""Prediction output format for Image Segmentation.

    Attributes:
        category_mask (str):
            A PNG image where each pixel in the mask
            represents the category in which the pixel in
            the original image was predicted to belong to.
            The size of this image will be the same as the
            original image. The mapping between the
            AnntoationSpec and the color can be found in
            model's metadata. The model will choose the most
            likely category and if none of the categories
            reach the confidence threshold, the pixel will
            be marked as background.
        confidence_mask (str):
            A one channel image which is encoded as an
            8bit lossless PNG. The size of the image will be
            the same as the original image. For a specific
            pixel, darker color means less confidence in
            correctness of the cateogry in the categoryMask
            for the corresponding pixel. Black means no
            confidence and white means complete confidence.
    """

    category_mask: str = proto.Field(
        proto.STRING,
        number=1,
    )
    confidence_mask: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
