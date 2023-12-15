# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
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
import proto  # type: ignore

from google.ai.generativelanguage_v1beta3.types import permission as gag_permission

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta3",
    manifest={
        "CreatePermissionRequest",
        "GetPermissionRequest",
        "ListPermissionsRequest",
        "ListPermissionsResponse",
        "UpdatePermissionRequest",
        "DeletePermissionRequest",
        "TransferOwnershipRequest",
        "TransferOwnershipResponse",
    },
)


class CreatePermissionRequest(proto.Message):
    r"""Request to create a ``Permission``.

    Attributes:
        parent (str):
            Required. The parent resource of the ``Permission``. Format:
            tunedModels/{tuned_model}
        permission (google.ai.generativelanguage_v1beta3.types.Permission):
            Required. The permission to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    permission: gag_permission.Permission = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gag_permission.Permission,
    )


class GetPermissionRequest(proto.Message):
    r"""Request for getting information about a specific ``Permission``.

    Attributes:
        name (str):
            Required. The resource name of the permission.

            Format:
            ``tunedModels/{tuned_model}permissions/{permission}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListPermissionsRequest(proto.Message):
    r"""Request for listing permissions.

    Attributes:
        parent (str):
            Required. The parent resource of the permissions. Format:
            tunedModels/{tuned_model}
        page_size (int):
            Optional. The maximum number of ``Permission``\ s to return
            (per page). The service may return fewer permissions.

            If unspecified, at most 10 permissions will be returned.
            This method returns at most 1000 permissions per page, even
            if you pass larger page_size.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListPermissions`` call.

            Provide the ``page_token`` returned by one request as an
            argument to the next request to retrieve the next page.

            When paginating, all other parameters provided to
            ``ListPermissions`` must match the call that provided the
            page token.
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


class ListPermissionsResponse(proto.Message):
    r"""Response from ``ListPermissions`` containing a paginated list of
    permissions.

    Attributes:
        permissions (MutableSequence[google.ai.generativelanguage_v1beta3.types.Permission]):
            Returned permissions.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page.

            If this field is omitted, there are no more pages.
    """

    @property
    def raw_page(self):
        return self

    permissions: MutableSequence[gag_permission.Permission] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gag_permission.Permission,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdatePermissionRequest(proto.Message):
    r"""Request to update the ``Permission``.

    Attributes:
        permission (google.ai.generativelanguage_v1beta3.types.Permission):
            Required. The permission to update.

            The permission's ``name`` field is used to identify the
            permission to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The list of fields to update. Accepted ones:

            -  role (``Permission.role`` field)
    """

    permission: gag_permission.Permission = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gag_permission.Permission,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeletePermissionRequest(proto.Message):
    r"""Request to delete the ``Permission``.

    Attributes:
        name (str):
            Required. The resource name of the permission. Format:
            ``tunedModels/{tuned_model}/permissions/{permission}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class TransferOwnershipRequest(proto.Message):
    r"""Request to transfer the ownership of the tuned model.

    Attributes:
        name (str):
            Required. The resource name of the tuned model to transfer
            ownership .

            Format: ``tunedModels/my-model-id``
        email_address (str):
            Required. The email address of the user to
            whom the tuned model is being transferred to.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    email_address: str = proto.Field(
        proto.STRING,
        number=2,
    )


class TransferOwnershipResponse(proto.Message):
    r"""Response from ``TransferOwnership``."""


__all__ = tuple(sorted(__protobuf__.manifest))
