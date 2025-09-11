# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
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
    package="google.iam.admin.v1",
    manifest={
        "AuditData",
    },
)


class AuditData(proto.Message):
    r"""Audit log information specific to Cloud IAM admin APIs. This message
    is serialized as an ``Any`` type in the ``ServiceData`` message of
    an ``AuditLog`` message.

    Attributes:
        permission_delta (google.cloud.iam_admin_v1.types.AuditData.PermissionDelta):
            The permission_delta when when creating or updating a Role.
    """

    class PermissionDelta(proto.Message):
        r"""A PermissionDelta message to record the added_permissions and
        removed_permissions inside a role.

        Attributes:
            added_permissions (MutableSequence[str]):
                Added permissions.
            removed_permissions (MutableSequence[str]):
                Removed permissions.
        """

        added_permissions: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )
        removed_permissions: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )

    permission_delta: PermissionDelta = proto.Field(
        proto.MESSAGE,
        number=1,
        message=PermissionDelta,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
