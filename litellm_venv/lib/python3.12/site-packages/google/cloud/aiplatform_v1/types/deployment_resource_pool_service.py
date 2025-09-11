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

from google.cloud.aiplatform_v1.types import deployed_model_ref
from google.cloud.aiplatform_v1.types import (
    deployment_resource_pool as gca_deployment_resource_pool,
)
from google.cloud.aiplatform_v1.types import endpoint
from google.cloud.aiplatform_v1.types import operation


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "CreateDeploymentResourcePoolRequest",
        "CreateDeploymentResourcePoolOperationMetadata",
        "GetDeploymentResourcePoolRequest",
        "ListDeploymentResourcePoolsRequest",
        "ListDeploymentResourcePoolsResponse",
        "UpdateDeploymentResourcePoolOperationMetadata",
        "DeleteDeploymentResourcePoolRequest",
        "QueryDeployedModelsRequest",
        "QueryDeployedModelsResponse",
    },
)


class CreateDeploymentResourcePoolRequest(proto.Message):
    r"""Request message for CreateDeploymentResourcePool method.

    Attributes:
        parent (str):
            Required. The parent location resource where this
            DeploymentResourcePool will be created. Format:
            ``projects/{project}/locations/{location}``
        deployment_resource_pool (google.cloud.aiplatform_v1.types.DeploymentResourcePool):
            Required. The DeploymentResourcePool to
            create.
        deployment_resource_pool_id (str):
            Required. The ID to use for the DeploymentResourcePool,
            which will become the final component of the
            DeploymentResourcePool's resource name.

            The maximum length is 63 characters, and valid characters
            are ``/^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    deployment_resource_pool: gca_deployment_resource_pool.DeploymentResourcePool = (
        proto.Field(
            proto.MESSAGE,
            number=2,
            message=gca_deployment_resource_pool.DeploymentResourcePool,
        )
    )
    deployment_resource_pool_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CreateDeploymentResourcePoolOperationMetadata(proto.Message):
    r"""Runtime operation information for
    CreateDeploymentResourcePool method.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GetDeploymentResourcePoolRequest(proto.Message):
    r"""Request message for GetDeploymentResourcePool method.

    Attributes:
        name (str):
            Required. The name of the DeploymentResourcePool to
            retrieve. Format:
            ``projects/{project}/locations/{location}/deploymentResourcePools/{deployment_resource_pool}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListDeploymentResourcePoolsRequest(proto.Message):
    r"""Request message for ListDeploymentResourcePools method.

    Attributes:
        parent (str):
            Required. The parent Location which owns this collection of
            DeploymentResourcePools. Format:
            ``projects/{project}/locations/{location}``
        page_size (int):
            The maximum number of DeploymentResourcePools
            to return. The service may return fewer than
            this value.
        page_token (str):
            A page token, received from a previous
            ``ListDeploymentResourcePools`` call. Provide this to
            retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``ListDeploymentResourcePools`` must match the call that
            provided the page token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListDeploymentResourcePoolsResponse(proto.Message):
    r"""Response message for ListDeploymentResourcePools method.

    Attributes:
        deployment_resource_pools (MutableSequence[google.cloud.aiplatform_v1.types.DeploymentResourcePool]):
            The DeploymentResourcePools from the
            specified location.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    deployment_resource_pools: MutableSequence[
        gca_deployment_resource_pool.DeploymentResourcePool
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_deployment_resource_pool.DeploymentResourcePool,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateDeploymentResourcePoolOperationMetadata(proto.Message):
    r"""Runtime operation information for
    UpdateDeploymentResourcePool method.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class DeleteDeploymentResourcePoolRequest(proto.Message):
    r"""Request message for DeleteDeploymentResourcePool method.

    Attributes:
        name (str):
            Required. The name of the DeploymentResourcePool to delete.
            Format:
            ``projects/{project}/locations/{location}/deploymentResourcePools/{deployment_resource_pool}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class QueryDeployedModelsRequest(proto.Message):
    r"""Request message for QueryDeployedModels method.

    Attributes:
        deployment_resource_pool (str):
            Required. The name of the target DeploymentResourcePool to
            query. Format:
            ``projects/{project}/locations/{location}/deploymentResourcePools/{deployment_resource_pool}``
        page_size (int):
            The maximum number of DeployedModels to
            return. The service may return fewer than this
            value.
        page_token (str):
            A page token, received from a previous
            ``QueryDeployedModels`` call. Provide this to retrieve the
            subsequent page.

            When paginating, all other parameters provided to
            ``QueryDeployedModels`` must match the call that provided
            the page token.
    """

    deployment_resource_pool: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class QueryDeployedModelsResponse(proto.Message):
    r"""Response message for QueryDeployedModels method.

    Attributes:
        deployed_models (MutableSequence[google.cloud.aiplatform_v1.types.DeployedModel]):
            DEPRECATED Use deployed_model_refs instead.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
        deployed_model_refs (MutableSequence[google.cloud.aiplatform_v1.types.DeployedModelRef]):
            References to the DeployedModels that share
            the specified deploymentResourcePool.
        total_deployed_model_count (int):
            The total number of DeployedModels on this
            DeploymentResourcePool.
        total_endpoint_count (int):
            The total number of Endpoints that have
            DeployedModels on this DeploymentResourcePool.
    """

    @property
    def raw_page(self):
        return self

    deployed_models: MutableSequence[endpoint.DeployedModel] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=endpoint.DeployedModel,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    deployed_model_refs: MutableSequence[
        deployed_model_ref.DeployedModelRef
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=deployed_model_ref.DeployedModelRef,
    )
    total_deployed_model_count: int = proto.Field(
        proto.INT32,
        number=4,
    )
    total_endpoint_count: int = proto.Field(
        proto.INT32,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
