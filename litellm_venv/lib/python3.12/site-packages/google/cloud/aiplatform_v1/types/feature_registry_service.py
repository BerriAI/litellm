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

from google.cloud.aiplatform_v1.types import feature_group as gca_feature_group
from google.cloud.aiplatform_v1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "CreateFeatureGroupRequest",
        "GetFeatureGroupRequest",
        "ListFeatureGroupsRequest",
        "ListFeatureGroupsResponse",
        "UpdateFeatureGroupRequest",
        "DeleteFeatureGroupRequest",
        "CreateFeatureGroupOperationMetadata",
        "UpdateFeatureGroupOperationMetadata",
        "CreateRegistryFeatureOperationMetadata",
        "UpdateFeatureOperationMetadata",
    },
)


class CreateFeatureGroupRequest(proto.Message):
    r"""Request message for
    [FeatureRegistryService.CreateFeatureGroup][google.cloud.aiplatform.v1.FeatureRegistryService.CreateFeatureGroup].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create
            FeatureGroups. Format:
            ``projects/{project}/locations/{location}'``
        feature_group (google.cloud.aiplatform_v1.types.FeatureGroup):
            Required. The FeatureGroup to create.
        feature_group_id (str):
            Required. The ID to use for this FeatureGroup, which will
            become the final component of the FeatureGroup's resource
            name.

            This value may be up to 60 characters, and valid characters
            are ``[a-z0-9_]``. The first character cannot be a number.

            The value must be unique within the project and location.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    feature_group: gca_feature_group.FeatureGroup = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_feature_group.FeatureGroup,
    )
    feature_group_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class GetFeatureGroupRequest(proto.Message):
    r"""Request message for
    [FeatureRegistryService.GetFeatureGroup][google.cloud.aiplatform.v1.FeatureRegistryService.GetFeatureGroup].

    Attributes:
        name (str):
            Required. The name of the FeatureGroup
            resource.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListFeatureGroupsRequest(proto.Message):
    r"""Request message for
    [FeatureRegistryService.ListFeatureGroups][google.cloud.aiplatform.v1.FeatureRegistryService.ListFeatureGroups].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list
            FeatureGroups. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Lists the FeatureGroups that match the filter expression.
            The following fields are supported:

            -  ``create_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``update_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``labels``: Supports key-value equality and key presence.

            Examples:

            -  ``create_time > "2020-01-01" OR update_time > "2020-01-01"``
               FeatureGroups created or updated after 2020-01-01.
            -  ``labels.env = "prod"`` FeatureGroups with label "env"
               set to "prod".
        page_size (int):
            The maximum number of FeatureGroups to
            return. The service may return fewer than this
            value. If unspecified, at most 100 FeatureGroups
            will be returned. The maximum value is 100; any
            value greater than 100 will be coerced to 100.
        page_token (str):
            A page token, received from a previous
            [FeatureGroupAdminService.ListFeatureGroups][] call. Provide
            this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [FeatureGroupAdminService.ListFeatureGroups][] must match
            the call that provided the page token.
        order_by (str):
            A comma-separated list of fields to order by, sorted in
            ascending order. Use "desc" after a field name for
            descending. Supported Fields:

            -  ``create_time``
            -  ``update_time``
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
    order_by: str = proto.Field(
        proto.STRING,
        number=5,
    )


class ListFeatureGroupsResponse(proto.Message):
    r"""Response message for
    [FeatureRegistryService.ListFeatureGroups][google.cloud.aiplatform.v1.FeatureRegistryService.ListFeatureGroups].

    Attributes:
        feature_groups (MutableSequence[google.cloud.aiplatform_v1.types.FeatureGroup]):
            The FeatureGroups matching the request.
        next_page_token (str):
            A token, which can be sent as
            [ListFeatureGroupsRequest.page_token][google.cloud.aiplatform.v1.ListFeatureGroupsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    feature_groups: MutableSequence[
        gca_feature_group.FeatureGroup
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_feature_group.FeatureGroup,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateFeatureGroupRequest(proto.Message):
    r"""Request message for
    [FeatureRegistryService.UpdateFeatureGroup][google.cloud.aiplatform.v1.FeatureRegistryService.UpdateFeatureGroup].

    Attributes:
        feature_group (google.cloud.aiplatform_v1.types.FeatureGroup):
            Required. The FeatureGroup's ``name`` field is used to
            identify the FeatureGroup to be updated. Format:
            ``projects/{project}/locations/{location}/featureGroups/{feature_group}``
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Field mask is used to specify the fields to be overwritten
            in the FeatureGroup resource by the update. The fields
            specified in the update_mask are relative to the resource,
            not the full request. A field will be overwritten if it is
            in the mask. If the user does not provide a mask then only
            the non-empty fields present in the request will be
            overwritten. Set the update_mask to ``*`` to override all
            fields.

            Updatable fields:

            -  ``labels``
    """

    feature_group: gca_feature_group.FeatureGroup = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_feature_group.FeatureGroup,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteFeatureGroupRequest(proto.Message):
    r"""Request message for
    [FeatureRegistryService.DeleteFeatureGroup][google.cloud.aiplatform.v1.FeatureRegistryService.DeleteFeatureGroup].

    Attributes:
        name (str):
            Required. The name of the FeatureGroup to be deleted.
            Format:
            ``projects/{project}/locations/{location}/featureGroups/{feature_group}``
        force (bool):
            If set to true, any Features under this
            FeatureGroup will also be deleted. (Otherwise,
            the request will only work if the FeatureGroup
            has no Features.)
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class CreateFeatureGroupOperationMetadata(proto.Message):
    r"""Details of operations that perform create FeatureGroup.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            Operation metadata for FeatureGroup.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class UpdateFeatureGroupOperationMetadata(proto.Message):
    r"""Details of operations that perform update FeatureGroup.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            Operation metadata for FeatureGroup.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class CreateRegistryFeatureOperationMetadata(proto.Message):
    r"""Details of operations that perform create FeatureGroup.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            Operation metadata for Feature.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class UpdateFeatureOperationMetadata(proto.Message):
    r"""Details of operations that perform update Feature.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            Operation metadata for Feature Update.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
