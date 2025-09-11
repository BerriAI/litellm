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
        "ComputeTokensRequest",
        "TokensInfo",
        "ComputeTokensResponse",
    },
)


class ComputeTokensRequest(proto.Message):
    r"""Request message for ComputeTokens RPC call.

    Attributes:
        endpoint (str):
            Required. The name of the Endpoint requested
            to get lists of tokens and token ids.
        instances (MutableSequence[google.protobuf.struct_pb2.Value]):
            Required. The instances that are the input to
            token computing API call. Schema is identical to
            the prediction schema of the text model, even
            for the non-text models, like chat models, or
            Codey models.
    """

    endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    instances: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Value,
    )


class TokensInfo(proto.Message):
    r"""Tokens info with a list of tokens and the corresponding list
    of token ids.

    Attributes:
        tokens (MutableSequence[bytes]):
            A list of tokens from the input.
        token_ids (MutableSequence[int]):
            A list of token ids from the input.
    """

    tokens: MutableSequence[bytes] = proto.RepeatedField(
        proto.BYTES,
        number=1,
    )
    token_ids: MutableSequence[int] = proto.RepeatedField(
        proto.INT64,
        number=2,
    )


class ComputeTokensResponse(proto.Message):
    r"""Response message for ComputeTokens RPC call.

    Attributes:
        tokens_info (MutableSequence[google.cloud.aiplatform_v1beta1.types.TokensInfo]):
            Lists of tokens info from the input. A
            ComputeTokensRequest could have multiple
            instances with a prompt in each instance. We
            also need to return lists of tokens info for the
            request with multiple instances.
    """

    tokens_info: MutableSequence["TokensInfo"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TokensInfo",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
