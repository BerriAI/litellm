# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
from typing import Optional, Any

from .._internal.managed_identity_base import AsyncManagedIdentityBase
from .._internal.managed_identity_client import AsyncManagedIdentityClient
from ..._credentials.service_fabric import _get_client_args


class ServiceFabricCredential(AsyncManagedIdentityBase):
    def get_client(self, **kwargs: Any) -> Optional[AsyncManagedIdentityClient]:
        client_args = _get_client_args(**kwargs)
        if client_args:
            return AsyncManagedIdentityClient(**client_args)
        return None

    def get_unavailable_message(self) -> str:
        return "Service Fabric managed identity configuration not found in environment"
