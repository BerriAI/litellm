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
        "AutoMlVideoActionRecognition",
        "AutoMlVideoActionRecognitionInputs",
    },
)


class AutoMlVideoActionRecognition(proto.Message):
    r"""A TrainingJob that trains and uploads an AutoML Video Action
    Recognition Model.

    Attributes:
        inputs (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlVideoActionRecognitionInputs):
            The input parameters of this TrainingJob.
    """

    inputs: "AutoMlVideoActionRecognitionInputs" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="AutoMlVideoActionRecognitionInputs",
    )


class AutoMlVideoActionRecognitionInputs(proto.Message):
    r"""

    Attributes:
        model_type (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlVideoActionRecognitionInputs.ModelType):

    """

    class ModelType(proto.Enum):
        r"""

        Values:
            MODEL_TYPE_UNSPECIFIED (0):
                Should not be set.
            CLOUD (1):
                A model best tailored to be used within
                Google Cloud, and which c annot be exported.
                Default.
            MOBILE_VERSATILE_1 (2):
                A model that, in addition to being available
                within Google Cloud, can also be exported (see
                ModelService.ExportModel) as a TensorFlow or
                TensorFlow Lite model and used on a mobile or
                edge device afterwards.
            MOBILE_JETSON_VERSATILE_1 (3):
                A model that, in addition to being available
                within Google Cloud, can also be exported (see
                ModelService.ExportModel) to a Jetson device
                afterwards.
            MOBILE_CORAL_VERSATILE_1 (4):
                A model that, in addition to being available
                within Google Cloud, can also be exported (see
                ModelService.ExportModel) as a TensorFlow or
                TensorFlow Lite model and used on a Coral device
                afterwards.
        """
        MODEL_TYPE_UNSPECIFIED = 0
        CLOUD = 1
        MOBILE_VERSATILE_1 = 2
        MOBILE_JETSON_VERSATILE_1 = 3
        MOBILE_CORAL_VERSATILE_1 = 4

    model_type: ModelType = proto.Field(
        proto.ENUM,
        number=1,
        enum=ModelType,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
