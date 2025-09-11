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
from __future__ import annotations

import dataclasses
from typing import Optional, Union, Any

import google.ai.generativelanguage as glm

from google.protobuf import field_mask_pb2

from google.generativeai.client import get_dafault_permission_client
from google.generativeai.client import get_dafault_permission_async_client
from google.generativeai.utils import flatten_update_paths
from google.generativeai import string_utils


GranteeType = glm.Permission.GranteeType
Role = glm.Permission.Role

GranteeTypeOptions = Union[str, int, GranteeType]
RoleOptions = Union[str, int, Role]

_GRANTEE_TYPE: dict[GranteeTypeOptions, GranteeType] = {
    GranteeType.GRANTEE_TYPE_UNSPECIFIED: GranteeType.GRANTEE_TYPE_UNSPECIFIED,
    0: GranteeType.GRANTEE_TYPE_UNSPECIFIED,
    "grantee_type_unspecified": GranteeType.GRANTEE_TYPE_UNSPECIFIED,
    "unspecified": GranteeType.GRANTEE_TYPE_UNSPECIFIED,
    GranteeType.USER: GranteeType.USER,
    1: GranteeType.USER,
    "user": GranteeType.USER,
    GranteeType.GROUP: GranteeType.GROUP,
    2: GranteeType.GROUP,
    "group": GranteeType.GROUP,
    GranteeType.EVERYONE: GranteeType.EVERYONE,
    3: GranteeType.EVERYONE,
    "everyone": GranteeType.EVERYONE,
}

_ROLE: dict[RoleOptions, Role] = {
    Role.ROLE_UNSPECIFIED: Role.ROLE_UNSPECIFIED,
    0: Role.ROLE_UNSPECIFIED,
    "role_unspecified": Role.ROLE_UNSPECIFIED,
    "unspecified": Role.ROLE_UNSPECIFIED,
    Role.OWNER: Role.OWNER,
    1: Role.OWNER,
    "owner": Role.OWNER,
    Role.WRITER: Role.WRITER,
    2: Role.WRITER,
    "writer": Role.WRITER,
    Role.READER: Role.READER,
    3: Role.READER,
    "reader": Role.READER,
}


def to_grantee_type(x: GranteeTypeOptions) -> GranteeType:
    if isinstance(x, str):
        x = x.lower()
    return _GRANTEE_TYPE[x]


def to_role(x: RoleOptions) -> Role:
    if isinstance(x, str):
        x = x.lower()
    return _ROLE[x]


@string_utils.prettyprint
@dataclasses.dataclass
class Permission:
    """
    A permission to access a resource.
    """

    name: str
    role: Role
    grantee_type: Optional[GranteeType]
    email_address: Optional[str] = None

    def delete(
        self,
        client: glm.PermissionServiceClient | None = None,
    ) -> None:
        """
        Delete permission (self).
        """
        if client is None:
            client = get_dafault_permission_client()
        delete_request = glm.DeletePermissionRequest(name=self.name)
        client.delete_permission(request=delete_request)

    async def delete_async(
        self,
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> None:
        """
        This is the async version of `Permission.delete`.
        """
        if client is None:
            client = get_dafault_permission_async_client()
        delete_request = glm.DeletePermissionRequest(name=self.name)
        await client.delete_permission(request=delete_request)

    # TODO (magashe): Add a method to validate update value. As of now only `role` is supported as a mask path
    def _apply_update(self, path, value):
        parts = path.split(".")
        for part in parts[:-1]:
            self = getattr(self, part)
        setattr(self, parts[-1], value)

    def update(
        self,
        updates: dict[str, Any],
        client: glm.PermissionServiceClient | None = None,
    ) -> Permission:
        """
        Update a list of fields for a specified permission.

        Args:
            updates: The list of fields to update.
                     Currently only `role` is supported as an update path.

        Returns:
            `Permission` object with specified updates.
        """
        if client is None:
            client = get_dafault_permission_client()

        updates = flatten_update_paths(updates)
        for update_path in updates:
            if update_path != "role":
                raise ValueError(
                    f"As of now, only `role` can be updated for `Permission`. Got: `{update_path}` instead."
                )
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        update_request = glm.UpdatePermissionRequest(
            permission=self.to_dict(), update_mask=field_mask
        )
        client.update_permission(request=update_request)
        return self

    async def update_async(
        self,
        updates: dict[str, Any],
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> Permission:
        """
        This is the async version of `Permission.update`.
        """
        if client is None:
            client = get_dafault_permission_async_client()

        updates = flatten_update_paths(updates)
        for update_path in updates:
            if update_path != "role":
                raise ValueError(
                    f"As of now, only `role` can be updated for `Permission`. Got: `{update_path}` instead."
                )
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        update_request = glm.UpdatePermissionRequest(
            permission=self.to_dict(), update_mask=field_mask
        )
        await client.update_permission(request=update_request)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "grantee_type": self.grantee_type,
            "email_address": self.email_address,
        }

    @classmethod
    def get(
        cls,
        name: str,
        client: glm.PermissionServiceClient | None = None,
    ) -> Permission:
        """
        Get information about a specific permission.

        Args:
            name: The name of the permission to get.

        Returns:
            Requested permission as an instance of `Permission`.
        """
        if client is None:
            client = get_dafault_permission_client()
        get_perm_request = glm.GetPermissionRequest(name=name)
        get_perm_response = client.get_permission(request=get_perm_request)
        get_perm_response = type(get_perm_response).to_dict(get_perm_response)
        return cls(**get_perm_response)

    @classmethod
    async def get_async(
        cls,
        name: str,
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> Permission:
        """
        This is the async version of `Permission.get`.
        """
        if client is None:
            client = get_dafault_permission_async_client()
        get_perm_request = glm.GetPermissionRequest(name=name)
        get_perm_response = await client.get_permission(request=get_perm_request)
        get_perm_response = type(get_perm_response).to_dict(get_perm_response)
        return cls(**get_perm_response)
