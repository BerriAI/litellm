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
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ManualBatchTuningParameters",
    },
)


class ManualBatchTuningParameters(proto.Message):
    r"""Manual batch tuning parameters.

    Attributes:
        batch_size (int):
            Immutable. The number of the records (e.g.
            instances) of the operation given in each batch
            to a machine replica. Machine type, and size of
            a single record should be considered when
            setting this parameter, higher value speeds up
            the batch operation's execution, but too high
            value will result in a whole batch not fitting
            in a machine's memory, and the whole operation
            will fail.
            The default value is 64.
    """

    batch_size: int = proto.Field(
        proto.INT32,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
