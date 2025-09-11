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

import google.ai.generativelanguage as glm

from google.generativeai.types import permission_types


def get_permission(
    name: str,
    client: glm.PermissionServiceClient | None = None,
) -> permission_types.Permission:
    """Get a permission by name.

    Args:
        name: The name of the permission.

    Returns:
        The permission as an instance of `permission_types.Permission`.
    """
    return permission_types.Permission.get(name=name, client=client)


async def get_permission_async(
    name: str,
    client: glm.PermissionServiceAsyncClient | None = None,
) -> permission_types.Permission:
    """
    This is the async version of `permission.get_permission`.
    """
    return await permission_types.Permission.get_async(name=name, client=client)
