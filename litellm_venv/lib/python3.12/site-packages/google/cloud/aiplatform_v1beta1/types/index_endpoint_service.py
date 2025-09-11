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

from google.cloud.aiplatform_v1beta1.types import index_endpoint as gca_index_endpoint
from google.cloud.aiplatform_v1beta1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateIndexEndpointRequest",
        "CreateIndexEndpointOperationMetadata",
        "GetIndexEndpointRequest",
        "ListIndexEndpointsRequest",
        "ListIndexEndpointsResponse",
        "UpdateIndexEndpointRequest",
        "DeleteIndexEndpointRequest",
        "DeployIndexRequest",
        "DeployIndexResponse",
        "DeployIndexOperationMetadata",
        "UndeployIndexRequest",
        "UndeployIndexResponse",
        "UndeployIndexOperationMetadata",
        "MutateDeployedIndexRequest",
        "MutateDeployedIndexResponse",
        "MutateDeployedIndexOperationMetadata",
    },
)


class CreateIndexEndpointRequest(proto.Message):
    r"""Request message for
    [IndexEndpointService.CreateIndexEndpoint][google.cloud.aiplatform.v1beta1.IndexEndpointService.CreateIndexEndpoint].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            IndexEndpoint in. Format:
            ``projects/{project}/locations/{location}``
        index_endpoint (google.cloud.aiplatform_v1beta1.types.IndexEndpoint):
            Required. The IndexEndpoint to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    index_endpoint: gca_index_endpoint.IndexEndpoint = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_index_endpoint.IndexEndpoint,
    )


class CreateIndexEndpointOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [IndexEndpointService.CreateIndexEndpoint][google.cloud.aiplatform.v1beta1.IndexEndpointService.CreateIndexEndpoint].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GetIndexEndpointRequest(proto.Message):
    r"""Request message for
    [IndexEndpointService.GetIndexEndpoint][google.cloud.aiplatform.v1beta1.IndexEndpointService.GetIndexEndpoint]

    Attributes:
        name (str):
            Required. The name of the IndexEndpoint resource. Format:
            ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListIndexEndpointsRequest(proto.Message):
    r"""Request message for
    [IndexEndpointService.ListIndexEndpoints][google.cloud.aiplatform.v1beta1.IndexEndpointService.ListIndexEndpoints].

    Attributes:
        parent (str):
            Required. The resource name of the Location from which to
            list the IndexEndpoints. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Optional. An expression for filtering the results of the
            request. For field names both snake_case and camelCase are
            supported.

            -  ``index_endpoint`` supports = and !=. ``index_endpoint``
               represents the IndexEndpoint ID, ie. the last segment of
               the IndexEndpoint's
               [resourcename][google.cloud.aiplatform.v1beta1.IndexEndpoint.name].
            -  ``display_name`` supports =, != and regex() (uses
               `re2 <https://github.com/google/re2/wiki/Syntax>`__
               syntax)
            -  ``labels`` supports general map functions that is:
               ``labels.key=value`` - key:value equality
               ``labels.key:* or labels:key - key existence A key including a space must be quoted.``\ labels."a
               key"`.

            Some examples:

            -  ``index_endpoint="1"``
            -  ``display_name="myDisplayName"``
            -  \`regex(display_name, "^A") -> The display name starts
               with an A.
            -  ``labels.myKey="myValue"``
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token. Typically obtained
            via
            [ListIndexEndpointsResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListIndexEndpointsResponse.next_page_token]
            of the previous
            [IndexEndpointService.ListIndexEndpoints][google.cloud.aiplatform.v1beta1.IndexEndpointService.ListIndexEndpoints]
            call.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Mask specifying which fields to
            read.
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListIndexEndpointsResponse(proto.Message):
    r"""Response message for
    [IndexEndpointService.ListIndexEndpoints][google.cloud.aiplatform.v1beta1.IndexEndpointService.ListIndexEndpoints].

    Attributes:
        index_endpoints (MutableSequence[google.cloud.aiplatform_v1beta1.types.IndexEndpoint]):
            List of IndexEndpoints in the requested page.
        next_page_token (str):
            A token to retrieve next page of results. Pass to
            [ListIndexEndpointsRequest.page_token][google.cloud.aiplatform.v1beta1.ListIndexEndpointsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    index_endpoints: MutableSequence[
        gca_index_endpoint.IndexEndpoint
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_index_endpoint.IndexEndpoint,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateIndexEndpointRequest(proto.Message):
    r"""Request message for
    [IndexEndpointService.UpdateIndexEndpoint][google.cloud.aiplatform.v1beta1.IndexEndpointService.UpdateIndexEndpoint].

    Attributes:
        index_endpoint (google.cloud.aiplatform_v1beta1.types.IndexEndpoint):
            Required. The IndexEndpoint which replaces
            the resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The update mask applies to the resource. See
            [google.protobuf.FieldMask][google.protobuf.FieldMask].
    """

    index_endpoint: gca_index_endpoint.IndexEndpoint = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_index_endpoint.IndexEndpoint,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteIndexEndpointRequest(proto.Message):
    r"""Request message for
    [IndexEndpointService.DeleteIndexEndpoint][google.cloud.aiplatform.v1beta1.IndexEndpointService.DeleteIndexEndpoint].

    Attributes:
        name (str):
            Required. The name of the IndexEndpoint resource to be
            deleted. Format:
            ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeployIndexRequest(proto.Message):
    r"""Request message for
    [IndexEndpointService.DeployIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.DeployIndex].

    Attributes:
        index_endpoint (str):
            Required. The name of the IndexEndpoint resource into which
            to deploy an Index. Format:
            ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
        deployed_index (google.cloud.aiplatform_v1beta1.types.DeployedIndex):
            Required. The DeployedIndex to be created
            within the IndexEndpoint.
    """

    index_endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    deployed_index: gca_index_endpoint.DeployedIndex = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_index_endpoint.DeployedIndex,
    )


class DeployIndexResponse(proto.Message):
    r"""Response message for
    [IndexEndpointService.DeployIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.DeployIndex].

    Attributes:
        deployed_index (google.cloud.aiplatform_v1beta1.types.DeployedIndex):
            The DeployedIndex that had been deployed in
            the IndexEndpoint.
    """

    deployed_index: gca_index_endpoint.DeployedIndex = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_index_endpoint.DeployedIndex,
    )


class DeployIndexOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [IndexEndpointService.DeployIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.DeployIndex].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
        deployed_index_id (str):
            The unique index id specified by user
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    deployed_index_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UndeployIndexRequest(proto.Message):
    r"""Request message for
    [IndexEndpointService.UndeployIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.UndeployIndex].

    Attributes:
        index_endpoint (str):
            Required. The name of the IndexEndpoint resource from which
            to undeploy an Index. Format:
            ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
        deployed_index_id (str):
            Required. The ID of the DeployedIndex to be
            undeployed from the IndexEndpoint.
    """

    index_endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    deployed_index_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UndeployIndexResponse(proto.Message):
    r"""Response message for
    [IndexEndpointService.UndeployIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.UndeployIndex].

    """


class UndeployIndexOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [IndexEndpointService.UndeployIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.UndeployIndex].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class MutateDeployedIndexRequest(proto.Message):
    r"""Request message for
    [IndexEndpointService.MutateDeployedIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.MutateDeployedIndex].

    Attributes:
        index_endpoint (str):
            Required. The name of the IndexEndpoint resource into which
            to deploy an Index. Format:
            ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
        deployed_index (google.cloud.aiplatform_v1beta1.types.DeployedIndex):
            Required. The DeployedIndex to be updated within the
            IndexEndpoint. Currently, the updatable fields are
            [DeployedIndex][automatic_resources] and
            [DeployedIndex][dedicated_resources]
    """

    index_endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    deployed_index: gca_index_endpoint.DeployedIndex = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_index_endpoint.DeployedIndex,
    )


class MutateDeployedIndexResponse(proto.Message):
    r"""Response message for
    [IndexEndpointService.MutateDeployedIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.MutateDeployedIndex].

    Attributes:
        deployed_index (google.cloud.aiplatform_v1beta1.types.DeployedIndex):
            The DeployedIndex that had been updated in
            the IndexEndpoint.
    """

    deployed_index: gca_index_endpoint.DeployedIndex = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_index_endpoint.DeployedIndex,
    )


class MutateDeployedIndexOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [IndexEndpointService.MutateDeployedIndex][google.cloud.aiplatform.v1beta1.IndexEndpointService.MutateDeployedIndex].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
        deployed_index_id (str):
            The unique index id specified by user
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    deployed_index_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
