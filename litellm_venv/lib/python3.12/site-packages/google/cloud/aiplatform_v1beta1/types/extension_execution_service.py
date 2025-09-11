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

from google.cloud.aiplatform_v1beta1.types import content as gca_content
from google.cloud.aiplatform_v1beta1.types import extension
from google.protobuf import struct_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ExecuteExtensionRequest",
        "ExecuteExtensionResponse",
        "QueryExtensionRequest",
        "QueryExtensionResponse",
    },
)


class ExecuteExtensionRequest(proto.Message):
    r"""Request message for
    [ExtensionExecutionService.ExecuteExtension][google.cloud.aiplatform.v1beta1.ExtensionExecutionService.ExecuteExtension].

    Attributes:
        name (str):
            Required. Name (identifier) of the extension; Format:
            ``projects/{project}/locations/{location}/extensions/{extension}``
        operation_id (str):
            Required. The desired ID of the operation to be executed in
            this extension as defined in
            [ExtensionOperation.operation_id][google.cloud.aiplatform.v1beta1.ExtensionOperation.operation_id].
        operation_params (google.protobuf.struct_pb2.Struct):
            Optional. Request parameters that will be
            used for executing this operation.

            The struct should be in a form of map with param
            name as the key and actual param value as the
            value.
            E.g. If this operation requires a param "name"
            to be set to "abc". you can set this to
            something like {"name": "abc"}.
        runtime_auth_config (google.cloud.aiplatform_v1beta1.types.AuthConfig):
            Optional. Auth config provided at runtime to override the
            default value in [Extension.manifest.auth_config][]. The
            AuthConfig.auth_type should match the value in
            [Extension.manifest.auth_config][].
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    operation_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    operation_params: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Struct,
    )
    runtime_auth_config: extension.AuthConfig = proto.Field(
        proto.MESSAGE,
        number=4,
        message=extension.AuthConfig,
    )


class ExecuteExtensionResponse(proto.Message):
    r"""Response message for
    [ExtensionExecutionService.ExecuteExtension][google.cloud.aiplatform.v1beta1.ExtensionExecutionService.ExecuteExtension].

    Attributes:
        content (str):
            Response content from the extension. The
            content should be conformant to the
            response.content schema in the extension's
            manifest/OpenAPI spec.
    """

    content: str = proto.Field(
        proto.STRING,
        number=2,
    )


class QueryExtensionRequest(proto.Message):
    r"""Request message for
    [ExtensionExecutionService.QueryExtension][google.cloud.aiplatform.v1beta1.ExtensionExecutionService.QueryExtension].

    Attributes:
        name (str):
            Required. Name (identifier) of the extension; Format:
            ``projects/{project}/locations/{location}/extensions/{extension}``
        contents (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
            Required. The content of the current
            conversation with the model.
            For single-turn queries, this is a single
            instance. For multi-turn queries, this is a
            repeated field that contains conversation
            history + latest request.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    contents: MutableSequence[gca_content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=gca_content.Content,
    )


class QueryExtensionResponse(proto.Message):
    r"""Response message for
    [ExtensionExecutionService.QueryExtension][google.cloud.aiplatform.v1beta1.ExtensionExecutionService.QueryExtension].

    Attributes:
        steps (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
            Steps of extension or LLM interaction, can
            contain function call, function response, or
            text response. The last step contains the final
            response to the query.
        failure_message (str):
            Failure message if any.
    """

    steps: MutableSequence[gca_content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_content.Content,
    )
    failure_message: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
