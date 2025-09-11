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
    package="google.cloud.aiplatform.v1beta1.schema.trainingjob.definition",
    manifest={
        "AutoMlImageObjectDetection",
        "AutoMlImageObjectDetectionInputs",
        "AutoMlImageObjectDetectionMetadata",
    },
)


class AutoMlImageObjectDetection(proto.Message):
    r"""A TrainingJob that trains and uploads an AutoML Image Object
    Detection Model.

    Attributes:
        inputs (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlImageObjectDetectionInputs):
            The input parameters of this TrainingJob.
        metadata (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlImageObjectDetectionMetadata):
            The metadata information
    """

    inputs: "AutoMlImageObjectDetectionInputs" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="AutoMlImageObjectDetectionInputs",
    )
    metadata: "AutoMlImageObjectDetectionMetadata" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="AutoMlImageObjectDetectionMetadata",
    )


class AutoMlImageObjectDetectionInputs(proto.Message):
    r"""

    Attributes:
        model_type (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlImageObjectDetectionInputs.ModelType):

        budget_milli_node_hours (int):
            The training budget of creating this model, expressed in
            milli node hours i.e. 1,000 value in this field means 1 node
            hour. The actual metadata.costMilliNodeHours will be equal
            or less than this value. If further model training ceases to
            provide any improvements, it will stop without using the
            full budget and the metadata.successfulStopReason will be
            ``model-converged``. Note, node_hour = actual_hour \*
            number_of_nodes_involved. For modelType
            ``cloud``\ (default), the budget must be between 20,000 and
            900,000 milli node hours, inclusive. The default value is
            216,000 which represents one day in wall time, considering 9
            nodes are used. For model types ``mobile-tf-low-latency-1``,
            ``mobile-tf-versatile-1``, ``mobile-tf-high-accuracy-1`` the
            training budget must be between 1,000 and 100,000 milli node
            hours, inclusive. The default value is 24,000 which
            represents one day in wall time on a single node that is
            used.
        disable_early_stopping (bool):
            Use the entire training budget. This disables
            the early stopping feature. When false the early
            stopping feature is enabled, which means that
            AutoML Image Object Detection might stop
            training before the entire training budget has
            been used.
    """

    class ModelType(proto.Enum):
        r"""

        Values:
            MODEL_TYPE_UNSPECIFIED (0):
                Should not be set.
            CLOUD_HIGH_ACCURACY_1 (1):
                A model best tailored to be used within
                Google Cloud, and which cannot be exported.
                Expected to have a higher latency, but should
                also have a higher prediction quality than other
                cloud models.
            CLOUD_LOW_LATENCY_1 (2):
                A model best tailored to be used within
                Google Cloud, and which cannot be exported.
                Expected to have a low latency, but may have
                lower prediction quality than other cloud
                models.
            MOBILE_TF_LOW_LATENCY_1 (3):
                A model that, in addition to being available
                within Google Cloud can also be exported (see
                ModelService.ExportModel) and used on a mobile
                or edge device with TensorFlow afterwards.
                Expected to have low latency, but may have lower
                prediction quality than other mobile models.
            MOBILE_TF_VERSATILE_1 (4):
                A model that, in addition to being available
                within Google Cloud can also be exported (see
                ModelService.ExportModel) and used on a mobile
                or edge device with TensorFlow afterwards.
            MOBILE_TF_HIGH_ACCURACY_1 (5):
                A model that, in addition to being available
                within Google Cloud, can also be exported (see
                ModelService.ExportModel) and used on a mobile
                or edge device with TensorFlow afterwards.
                Expected to have a higher latency, but should
                also have a higher prediction quality than other
                mobile models.
        """
        MODEL_TYPE_UNSPECIFIED = 0
        CLOUD_HIGH_ACCURACY_1 = 1
        CLOUD_LOW_LATENCY_1 = 2
        MOBILE_TF_LOW_LATENCY_1 = 3
        MOBILE_TF_VERSATILE_1 = 4
        MOBILE_TF_HIGH_ACCURACY_1 = 5

    model_type: ModelType = proto.Field(
        proto.ENUM,
        number=1,
        enum=ModelType,
    )
    budget_milli_node_hours: int = proto.Field(
        proto.INT64,
        number=2,
    )
    disable_early_stopping: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class AutoMlImageObjectDetectionMetadata(proto.Message):
    r"""

    Attributes:
        cost_milli_node_hours (int):
            The actual training cost of creating this
            model, expressed in milli node hours, i.e. 1,000
            value in this field means 1 node hour.
            Guaranteed to not exceed
            inputs.budgetMilliNodeHours.
        successful_stop_reason (google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1.types.AutoMlImageObjectDetectionMetadata.SuccessfulStopReason):
            For successful job completions, this is the
            reason why the job has finished.
    """

    class SuccessfulStopReason(proto.Enum):
        r"""

        Values:
            SUCCESSFUL_STOP_REASON_UNSPECIFIED (0):
                Should not be set.
            BUDGET_REACHED (1):
                The inputs.budgetMilliNodeHours had been
                reached.
            MODEL_CONVERGED (2):
                Further training of the Model ceased to
                increase its quality, since it already has
                converged.
        """
        SUCCESSFUL_STOP_REASON_UNSPECIFIED = 0
        BUDGET_REACHED = 1
        MODEL_CONVERGED = 2

    cost_milli_node_hours: int = proto.Field(
        proto.INT64,
        number=1,
    )
    successful_stop_reason: SuccessfulStopReason = proto.Field(
        proto.ENUM,
        number=2,
        enum=SuccessfulStopReason,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
