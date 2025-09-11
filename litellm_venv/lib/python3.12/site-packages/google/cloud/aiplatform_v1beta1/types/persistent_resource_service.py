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
    persistent_resource as gca_persistent_resource,
)
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreatePersistentResourceRequest",
        "CreatePersistentResourceOperationMetadata",
        "UpdatePersistentResourceOperationMetadata",
        "RebootPersistentResourceOperationMetadata",
        "GetPersistentResourceRequest",
        "ListPersistentResourcesRequest",
        "ListPersistentResourcesResponse",
        "DeletePersistentResourceRequest",
        "UpdatePersistentResourceRequest",
        "RebootPersistentResourceRequest",
    },
)


class CreatePersistentResourceRequest(proto.Message):
    r"""Request message for
    [PersistentResourceService.CreatePersistentResource][google.cloud.aiplatform.v1beta1.PersistentResourceService.CreatePersistentResource].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            PersistentResource in. Format:
            ``projects/{project}/locations/{location}``
        persistent_resource (google.cloud.aiplatform_v1beta1.types.PersistentResource):
            Required. The PersistentResource to create.
        persistent_resource_id (str):
            Required. The ID to use for the PersistentResource, which
            become the final component of the PersistentResource's
            resource name.

            The maximum length is 63 characters, and valid characters
            are ``/^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    persistent_resource: gca_persistent_resource.PersistentResource = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_persistent_resource.PersistentResource,
    )
    persistent_resource_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CreatePersistentResourceOperationMetadata(proto.Message):
    r"""Details of operations that perform create PersistentResource.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            Operation metadata for PersistentResource.
        progress_message (str):
            Progress Message for Create LRO
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    progress_message: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdatePersistentResourceOperationMetadata(proto.Message):
    r"""Details of operations that perform update PersistentResource.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            Operation metadata for PersistentResource.
        progress_message (str):
            Progress Message for Update LRO
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    progress_message: str = proto.Field(
        proto.STRING,
        number=2,
    )


class RebootPersistentResourceOperationMetadata(proto.Message):
    r"""Details of operations that perform reboot PersistentResource.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            Operation metadata for PersistentResource.
        progress_message (str):
            Progress Message for Reboot LRO
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    progress_message: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetPersistentResourceRequest(proto.Message):
    r"""Request message for
    [PersistentResourceService.GetPersistentResource][google.cloud.aiplatform.v1beta1.PersistentResourceService.GetPersistentResource].

    Attributes:
        name (str):
            Required. The name of the PersistentResource resource.
            Format:
            ``projects/{project_id_or_number}/locations/{location_id}/persistentResources/{persistent_resource_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListPersistentResourcesRequest(proto.Message):
    r"""Request message for
    [PersistentResourceService.ListPersistentResource][].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            PersistentResources from. Format:
            ``projects/{project}/locations/{location}``
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token. Typically obtained
            via [ListPersistentResourceResponse.next_page_token][] of
            the previous
            [PersistentResourceService.ListPersistentResource][] call.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListPersistentResourcesResponse(proto.Message):
    r"""Response message for
    [PersistentResourceService.ListPersistentResources][google.cloud.aiplatform.v1beta1.PersistentResourceService.ListPersistentResources]

    Attributes:
        persistent_resources (MutableSequence[google.cloud.aiplatform_v1beta1.types.PersistentResource]):

        next_page_token (str):
            A token to retrieve next page of results. Pass to
            [ListPersistentResourcesRequest.page_token][google.cloud.aiplatform.v1beta1.ListPersistentResourcesRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    persistent_resources: MutableSequence[
        gca_persistent_resource.PersistentResource
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_persistent_resource.PersistentResource,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeletePersistentResourceRequest(proto.Message):
    r"""Request message for
    [PersistentResourceService.DeletePersistentResource][google.cloud.aiplatform.v1beta1.PersistentResourceService.DeletePersistentResource].

    Attributes:
        name (str):
            Required. The name of the PersistentResource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/persistentResources/{persistent_resource}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdatePersistentResourceRequest(proto.Message):
    r"""Request message for UpdatePersistentResource method.

    Attributes:
        persistent_resource (google.cloud.aiplatform_v1beta1.types.PersistentResource):
            Required. The PersistentResource to update.

            The PersistentResource's ``name`` field is used to identify
            the PersistentResource to update. Format:
            ``projects/{project}/locations/{location}/persistentResources/{persistent_resource}``
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. Specify the fields to be
            overwritten in the PersistentResource by the
            update method.
    """

    persistent_resource: gca_persistent_resource.PersistentResource = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_persistent_resource.PersistentResource,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class RebootPersistentResourceRequest(proto.Message):
    r"""Request message for
    [PersistentResourceService.RebootPersistentResource][google.cloud.aiplatform.v1beta1.PersistentResourceService.RebootPersistentResource].

    Attributes:
        name (str):
            Required. The name of the PersistentResource resource.
            Format:
            ``projects/{project_id_or_number}/locations/{location_id}/persistentResources/{persistent_resource_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
