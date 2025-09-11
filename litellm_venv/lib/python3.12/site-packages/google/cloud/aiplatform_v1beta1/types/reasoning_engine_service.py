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

from google.cloud.aiplatform_v1beta1.types import operation
from google.cloud.aiplatform_v1beta1.types import (
    reasoning_engine as gca_reasoning_engine,
)


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateReasoningEngineRequest",
        "CreateReasoningEngineOperationMetadata",
        "GetReasoningEngineRequest",
        "ListReasoningEnginesRequest",
        "ListReasoningEnginesResponse",
        "DeleteReasoningEngineRequest",
    },
)


class CreateReasoningEngineRequest(proto.Message):
    r"""Request message for
    [ReasoningEngineService.CreateReasoningEngine][google.cloud.aiplatform.v1beta1.ReasoningEngineService.CreateReasoningEngine].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            ReasoningEngine in. Format:
            ``projects/{project}/locations/{location}``
        reasoning_engine (google.cloud.aiplatform_v1beta1.types.ReasoningEngine):
            Required. The ReasoningEngine to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    reasoning_engine: gca_reasoning_engine.ReasoningEngine = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_reasoning_engine.ReasoningEngine,
    )


class CreateReasoningEngineOperationMetadata(proto.Message):
    r"""Details of
    [ReasoningEngineService.CreateReasoningEngine][google.cloud.aiplatform.v1beta1.ReasoningEngineService.CreateReasoningEngine]
    operation.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GetReasoningEngineRequest(proto.Message):
    r"""Request message for
    [ReasoningEngineService.GetReasoningEngine][google.cloud.aiplatform.v1beta1.ReasoningEngineService.GetReasoningEngine].

    Attributes:
        name (str):
            Required. The name of the ReasoningEngine resource. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListReasoningEnginesRequest(proto.Message):
    r"""Request message for
    [ReasoningEngineService.ListReasoningEngines][google.cloud.aiplatform.v1beta1.ReasoningEngineService.ListReasoningEngines].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            ReasoningEngines from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Optional. The standard list filter. More detail in
            `AIP-160 <https://google.aip.dev/160>`__.
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListReasoningEnginesResponse(proto.Message):
    r"""Response message for
    [ReasoningEngineService.ListReasoningEngines][google.cloud.aiplatform.v1beta1.ReasoningEngineService.ListReasoningEngines]

    Attributes:
        reasoning_engines (MutableSequence[google.cloud.aiplatform_v1beta1.types.ReasoningEngine]):
            List of ReasoningEngines in the requested
            page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListReasoningEnginesRequest.page_token][google.cloud.aiplatform.v1beta1.ListReasoningEnginesRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    reasoning_engines: MutableSequence[
        gca_reasoning_engine.ReasoningEngine
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_reasoning_engine.ReasoningEngine,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteReasoningEngineRequest(proto.Message):
    r"""Request message for
    [ReasoningEngineService.DeleteReasoningEngine][google.cloud.aiplatform.v1beta1.ReasoningEngineService.DeleteReasoningEngine].

    Attributes:
        name (str):
            Required. The name of the ReasoningEngine resource to be
            deleted. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
