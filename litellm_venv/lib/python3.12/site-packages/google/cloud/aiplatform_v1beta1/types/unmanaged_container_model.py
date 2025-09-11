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

from google.cloud.aiplatform_v1beta1.types import model


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "UnmanagedContainerModel",
    },
)


class UnmanagedContainerModel(proto.Message):
    r"""Contains model information necessary to perform batch
    prediction without requiring a full model import.

    Attributes:
        artifact_uri (str):
            The path to the directory containing the
            Model artifact and any of its supporting files.
        predict_schemata (google.cloud.aiplatform_v1beta1.types.PredictSchemata):
            Contains the schemata used in Model's
            predictions and explanations
        container_spec (google.cloud.aiplatform_v1beta1.types.ModelContainerSpec):
            Input only. The specification of the
            container that is to be used when deploying this
            Model.
    """

    artifact_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    predict_schemata: model.PredictSchemata = proto.Field(
        proto.MESSAGE,
        number=2,
        message=model.PredictSchemata,
    )
    container_spec: model.ModelContainerSpec = proto.Field(
        proto.MESSAGE,
        number=3,
        message=model.ModelContainerSpec,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
