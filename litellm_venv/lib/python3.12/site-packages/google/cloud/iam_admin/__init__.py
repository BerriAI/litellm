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
from google.cloud.iam_admin import gapic_version as package_version

__version__ = package_version.__version__


from google.cloud.iam_admin_v1.services.iam.async_client import IAMAsyncClient
from google.cloud.iam_admin_v1.services.iam.client import IAMClient
from google.cloud.iam_admin_v1.types.audit_data import AuditData
from google.cloud.iam_admin_v1.types.iam import (
    CreateRoleRequest,
    CreateServiceAccountKeyRequest,
    CreateServiceAccountRequest,
    DeleteRoleRequest,
    DeleteServiceAccountKeyRequest,
    DeleteServiceAccountRequest,
    DisableServiceAccountKeyRequest,
    DisableServiceAccountRequest,
    EnableServiceAccountKeyRequest,
    EnableServiceAccountRequest,
    GetRoleRequest,
    GetServiceAccountKeyRequest,
    GetServiceAccountRequest,
    LintPolicyRequest,
    LintPolicyResponse,
    LintResult,
    ListRolesRequest,
    ListRolesResponse,
    ListServiceAccountKeysRequest,
    ListServiceAccountKeysResponse,
    ListServiceAccountsRequest,
    ListServiceAccountsResponse,
    PatchServiceAccountRequest,
    Permission,
    QueryAuditableServicesRequest,
    QueryAuditableServicesResponse,
    QueryGrantableRolesRequest,
    QueryGrantableRolesResponse,
    QueryTestablePermissionsRequest,
    QueryTestablePermissionsResponse,
    Role,
    RoleView,
    ServiceAccount,
    ServiceAccountKey,
    ServiceAccountKeyAlgorithm,
    ServiceAccountKeyOrigin,
    ServiceAccountPrivateKeyType,
    ServiceAccountPublicKeyType,
    SignBlobRequest,
    SignBlobResponse,
    SignJwtRequest,
    SignJwtResponse,
    UndeleteRoleRequest,
    UndeleteServiceAccountRequest,
    UndeleteServiceAccountResponse,
    UpdateRoleRequest,
    UploadServiceAccountKeyRequest,
)

__all__ = (
    "IAMClient",
    "IAMAsyncClient",
    "AuditData",
    "CreateRoleRequest",
    "CreateServiceAccountKeyRequest",
    "CreateServiceAccountRequest",
    "DeleteRoleRequest",
    "DeleteServiceAccountKeyRequest",
    "DeleteServiceAccountRequest",
    "DisableServiceAccountKeyRequest",
    "DisableServiceAccountRequest",
    "EnableServiceAccountKeyRequest",
    "EnableServiceAccountRequest",
    "GetRoleRequest",
    "GetServiceAccountKeyRequest",
    "GetServiceAccountRequest",
    "LintPolicyRequest",
    "LintPolicyResponse",
    "LintResult",
    "ListRolesRequest",
    "ListRolesResponse",
    "ListServiceAccountKeysRequest",
    "ListServiceAccountKeysResponse",
    "ListServiceAccountsRequest",
    "ListServiceAccountsResponse",
    "PatchServiceAccountRequest",
    "Permission",
    "QueryAuditableServicesRequest",
    "QueryAuditableServicesResponse",
    "QueryGrantableRolesRequest",
    "QueryGrantableRolesResponse",
    "QueryTestablePermissionsRequest",
    "QueryTestablePermissionsResponse",
    "Role",
    "ServiceAccount",
    "ServiceAccountKey",
    "SignBlobRequest",
    "SignBlobResponse",
    "SignJwtRequest",
    "SignJwtResponse",
    "UndeleteRoleRequest",
    "UndeleteServiceAccountRequest",
    "UndeleteServiceAccountResponse",
    "UpdateRoleRequest",
    "UploadServiceAccountKeyRequest",
    "RoleView",
    "ServiceAccountKeyAlgorithm",
    "ServiceAccountKeyOrigin",
    "ServiceAccountPrivateKeyType",
    "ServiceAccountPublicKeyType",
)
