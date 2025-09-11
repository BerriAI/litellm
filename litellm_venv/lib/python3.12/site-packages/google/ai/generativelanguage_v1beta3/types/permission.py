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

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta3",
    manifest={
        "Permission",
    },
)


class Permission(proto.Message):
    r"""Permission resource grants user, group or the rest of the
    world access to the PaLM API resource (e.g. a tuned model,
    file).

    A role is a collection of permitted operations that allows users
    to perform specific actions on PaLM API resources. To make them
    available to users, groups, or service accounts, you assign
    roles. When you assign a role, you grant permissions that the
    role contains.

    There are three concentric roles. Each role is a superset of the
    previous role's permitted operations:

    - reader can use the resource (e.g. tuned model) for inference
    - writer has reader's permissions and additionally can edit and
      share
    - owner has writer's permissions and additionally can delete


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Output only. The permission name. A unique name will be
            generated on create. Example:
            tunedModels/{tuned_model}permssions/{permission} Output
            only.
        grantee_type (google.ai.generativelanguage_v1beta3.types.Permission.GranteeType):
            Required. Immutable. The type of the grantee.

            This field is a member of `oneof`_ ``_grantee_type``.
        email_address (str):
            Optional. Immutable. The email address of the
            user of group which this permission refers.
            Field is not set when permission's grantee type
            is EVERYONE.

            This field is a member of `oneof`_ ``_email_address``.
        role (google.ai.generativelanguage_v1beta3.types.Permission.Role):
            Required. The role granted by this
            permission.

            This field is a member of `oneof`_ ``_role``.
    """

    class GranteeType(proto.Enum):
        r"""Defines types of the grantee of this permission.

        Values:
            GRANTEE_TYPE_UNSPECIFIED (0):
                The default value. This value is unused.
            USER (1):
                Represents a user. When set, you must provide email_address
                for the user.
            GROUP (2):
                Represents a group. When set, you must provide email_address
                for the group.
            EVERYONE (3):
                Represents access to everyone. No extra
                information is required.
        """
        GRANTEE_TYPE_UNSPECIFIED = 0
        USER = 1
        GROUP = 2
        EVERYONE = 3

    class Role(proto.Enum):
        r"""Defines the role granted by this permission.

        Values:
            ROLE_UNSPECIFIED (0):
                The default value. This value is unused.
            OWNER (1):
                Owner can use, update, share and delete the
                resource.
            WRITER (2):
                Writer can use, update and share the
                resource.
            READER (3):
                Reader can use the resource.
        """
        ROLE_UNSPECIFIED = 0
        OWNER = 1
        WRITER = 2
        READER = 3

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    grantee_type: GranteeType = proto.Field(
        proto.ENUM,
        number=2,
        optional=True,
        enum=GranteeType,
    )
    email_address: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    role: Role = proto.Field(
        proto.ENUM,
        number=4,
        optional=True,
        enum=Role,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
