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

from google.protobuf import struct_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "QueryReasoningEngineRequest",
        "QueryReasoningEngineResponse",
    },
)


class QueryReasoningEngineRequest(proto.Message):
    r"""Request message for [ReasoningEngineExecutionService.Query][].

    Attributes:
        name (str):
            Required. The name of the ReasoningEngine resource to use.
            Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
        input (google.protobuf.struct_pb2.Struct):
            Optional. Input content provided by users in
            JSON object format. Examples include text query,
            function calling parameters, media bytes, etc.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    input: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Struct,
    )


class QueryReasoningEngineResponse(proto.Message):
    r"""Response message for [ReasoningEngineExecutionService.Query][]

    Attributes:
        output (google.protobuf.struct_pb2.Value):
            Response provided by users in JSON object
            format.
    """

    output: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=1,
        message=struct_pb2.Value,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
