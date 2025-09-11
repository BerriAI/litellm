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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "NotebookIdleShutdownConfig",
    },
)


class NotebookIdleShutdownConfig(proto.Message):
    r"""The idle shutdown configuration of NotebookRuntimeTemplate, which
    contains the idle_timeout as required field.

    Attributes:
        idle_timeout (google.protobuf.duration_pb2.Duration):
            Required. Duration is accurate to the second. In Notebook,
            Idle Timeout is accurate to minute so the range of
            idle_timeout (second) is: 10 \* 60 ~ 1440

            -

               60.
        idle_shutdown_disabled (bool):
            Whether Idle Shutdown is disabled in this
            NotebookRuntimeTemplate.
    """

    idle_timeout: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )
    idle_shutdown_disabled: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
