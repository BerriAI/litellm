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

from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.resourcemanager.v3",
    manifest={
        "Purpose",
        "TagKey",
        "ListTagKeysRequest",
        "ListTagKeysResponse",
        "GetTagKeyRequest",
        "GetNamespacedTagKeyRequest",
        "CreateTagKeyRequest",
        "CreateTagKeyMetadata",
        "UpdateTagKeyRequest",
        "UpdateTagKeyMetadata",
        "DeleteTagKeyRequest",
        "DeleteTagKeyMetadata",
    },
)


class Purpose(proto.Enum):
    r"""A purpose for each policy engine requiring such an
    integration. A single policy engine may have multiple purposes
    defined, however a TagKey may only specify a single purpose.

    Values:
        PURPOSE_UNSPECIFIED (0):
            Unspecified purpose.
        GCE_FIREWALL (1):
            Purpose for Compute Engine firewalls. A corresponding
            ``purpose_data`` should be set for the network the tag is
            intended for. The key should be ``network`` and the value
            should be in either of these two formats:

            -

            ``https://www.googleapis.com/compute/{compute_version}/projects/{project_id}/global/networks/{network_id}``

            -  ``{project_id}/{network_name}``

            Examples:

            -

            ``https://www.googleapis.com/compute/staging_v1/projects/fail-closed-load-testing/global/networks/6992953698831725600``

            -  ``fail-closed-load-testing/load-testing-network``
    """
    PURPOSE_UNSPECIFIED = 0
    GCE_FIREWALL = 1


class TagKey(proto.Message):
    r"""A TagKey, used to group a set of TagValues.

    Attributes:
        name (str):
            Immutable. The resource name for a TagKey. Must be in the
            format ``tagKeys/{tag_key_id}``, where ``tag_key_id`` is the
            generated numeric id for the TagKey.
        parent (str):
            Immutable. The resource name of the TagKey's parent. A
            TagKey can be parented by an Organization or a Project. For
            a TagKey parented by an Organization, its parent must be in
            the form ``organizations/{org_id}``. For a TagKey parented
            by a Project, its parent can be in the form
            ``projects/{project_id}`` or ``projects/{project_number}``.
        short_name (str):
            Required. Immutable. The user friendly name for a TagKey.
            The short name should be unique for TagKeys within the same
            tag namespace.

            The short name must be 1-63 characters, beginning and ending
            with an alphanumeric character ([a-z0-9A-Z]) with dashes
            (-), underscores (_), dots (.), and alphanumerics between.
        namespaced_name (str):
            Output only. Immutable. Namespaced name of
            the TagKey.
        description (str):
            Optional. User-assigned description of the
            TagKey. Must not exceed 256 characters.

            Read-write.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Creation time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Update time.
        etag (str):
            Optional. Entity tag which users can pass to
            prevent race conditions. This field is always
            set in server responses. See UpdateTagKeyRequest
            for details.
        purpose (google.cloud.resourcemanager_v3.types.Purpose):
            Optional. A purpose denotes that this Tag is
            intended for use in policies of a specific
            policy engine, and will involve that policy
            engine in management operations involving this
            Tag. A purpose does not grant a policy engine
            exclusive rights to the Tag, and it may be
            referenced by other policy engines.

            A purpose cannot be changed once set.
        purpose_data (MutableMapping[str, str]):
            Optional. Purpose data corresponds to the policy system that
            the tag is intended for. See documentation for ``Purpose``
            for formatting of this field.

            Purpose data cannot be changed once set.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=2,
    )
    short_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    namespaced_name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    description: str = proto.Field(
        proto.STRING,
        number=5,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=8,
    )
    purpose: "Purpose" = proto.Field(
        proto.ENUM,
        number=11,
        enum="Purpose",
    )
    purpose_data: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=12,
    )


class ListTagKeysRequest(proto.Message):
    r"""The request message for listing all TagKeys under a parent
    resource.

    Attributes:
        parent (str):
            Required. The resource name of the TagKey's parent. Must be
            of the form ``organizations/{org_id}`` or
            ``projects/{project_id}`` or ``projects/{project_number}``
        page_size (int):
            Optional. The maximum number of TagKeys to
            return in the response. The server allows a
            maximum of 300 TagKeys to return. If
            unspecified, the server will use 100 as the
            default.
        page_token (str):
            Optional. A pagination token returned from a previous call
            to ``ListTagKey`` that indicates where this listing should
            continue from.
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


class ListTagKeysResponse(proto.Message):
    r"""The ListTagKeys response message.

    Attributes:
        tag_keys (MutableSequence[google.cloud.resourcemanager_v3.types.TagKey]):
            List of TagKeys that live under the specified
            parent in the request.
        next_page_token (str):
            A pagination token returned from a previous call to
            ``ListTagKeys`` that indicates from where listing should
            continue.
    """

    @property
    def raw_page(self):
        return self

    tag_keys: MutableSequence["TagKey"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TagKey",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetTagKeyRequest(proto.Message):
    r"""The request message for getting a TagKey.

    Attributes:
        name (str):
            Required. A resource name in the format ``tagKeys/{id}``,
            such as ``tagKeys/123``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetNamespacedTagKeyRequest(proto.Message):
    r"""The request message for getting a TagKey by its namespaced
    name.

    Attributes:
        name (str):
            Required. A namespaced tag key name in the format
            ``{parentId}/{tagKeyShort}``, such as ``42/foo`` for a key
            with short name "foo" under the organization with ID 42 or
            ``r2-d2/bar`` for a key with short name "bar" under the
            project ``r2-d2``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateTagKeyRequest(proto.Message):
    r"""The request message for creating a TagKey.

    Attributes:
        tag_key (google.cloud.resourcemanager_v3.types.TagKey):
            Required. The TagKey to be created. Only fields
            ``short_name``, ``description``, and ``parent`` are
            considered during the creation request.
        validate_only (bool):
            Optional. Set to true to perform validations
            necessary for creating the resource, but not
            actually perform the action.
    """

    tag_key: "TagKey" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TagKey",
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class CreateTagKeyMetadata(proto.Message):
    r"""Runtime operation information for creating a TagKey."""


class UpdateTagKeyRequest(proto.Message):
    r"""The request message for updating a TagKey.

    Attributes:
        tag_key (google.cloud.resourcemanager_v3.types.TagKey):
            Required. The new definition of the TagKey. Only the
            ``description`` and ``etag`` fields can be updated by this
            request. If the ``etag`` field is not empty, it must match
            the ``etag`` field of the existing tag key. Otherwise,
            ``ABORTED`` will be returned.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Fields to be updated. The mask may only contain
            ``description`` or ``etag``. If omitted entirely, both
            ``description`` and ``etag`` are assumed to be significant.
        validate_only (bool):
            Set as true to perform validations necessary
            for updating the resource, but not actually
            perform the action.
    """

    tag_key: "TagKey" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TagKey",
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class UpdateTagKeyMetadata(proto.Message):
    r"""Runtime operation information for updating a TagKey."""


class DeleteTagKeyRequest(proto.Message):
    r"""The request message for deleting a TagKey.

    Attributes:
        name (str):
            Required. The resource name of a TagKey to be deleted in the
            format ``tagKeys/123``. The TagKey cannot be a parent of any
            existing TagValues or it will not be deleted successfully.
        validate_only (bool):
            Optional. Set as true to perform validations
            necessary for deletion, but not actually perform
            the action.
        etag (str):
            Optional. The etag known to the client for
            the expected state of the TagKey. This is to be
            used for optimistic concurrency.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=3,
    )


class DeleteTagKeyMetadata(proto.Message):
    r"""Runtime operation information for deleting a TagKey."""


__all__ = tuple(sorted(__protobuf__.manifest))
