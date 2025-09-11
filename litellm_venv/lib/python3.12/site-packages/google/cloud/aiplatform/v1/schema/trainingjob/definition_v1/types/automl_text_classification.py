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
        "AutoMlTextClassification",
        "AutoMlTextClassificationInputs",
    },
)


class AutoMlTextClassification(proto.Message):
    r"""A TrainingJob that trains and uploads an AutoML Text
    Classification Model.

    Attributes:
        inputs (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlTextClassificationInputs):
            The input parameters of this TrainingJob.
    """

    inputs: "AutoMlTextClassificationInputs" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="AutoMlTextClassificationInputs",
    )


class AutoMlTextClassificationInputs(proto.Message):
    r"""

    Attributes:
        multi_label (bool):

    """

    multi_label: bool = proto.Field(
        proto.BOOL,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
