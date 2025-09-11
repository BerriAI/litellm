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
from google.cloud.iam_v3beta import gapic_version as package_version

__version__ = package_version.__version__


from .services.policy_bindings import PolicyBindingsAsyncClient, PolicyBindingsClient
from .services.principal_access_boundary_policies import (
    PrincipalAccessBoundaryPoliciesAsyncClient,
    PrincipalAccessBoundaryPoliciesClient,
)
from .types.operation_metadata import OperationMetadata
from .types.policy_binding_resources import PolicyBinding
from .types.policy_bindings_service import (
    CreatePolicyBindingRequest,
    DeletePolicyBindingRequest,
    GetPolicyBindingRequest,
    ListPolicyBindingsRequest,
    ListPolicyBindingsResponse,
    SearchTargetPolicyBindingsRequest,
    SearchTargetPolicyBindingsResponse,
    UpdatePolicyBindingRequest,
)
from .types.principal_access_boundary_policies_service import (
    CreatePrincipalAccessBoundaryPolicyRequest,
    DeletePrincipalAccessBoundaryPolicyRequest,
    GetPrincipalAccessBoundaryPolicyRequest,
    ListPrincipalAccessBoundaryPoliciesRequest,
    ListPrincipalAccessBoundaryPoliciesResponse,
    SearchPrincipalAccessBoundaryPolicyBindingsRequest,
    SearchPrincipalAccessBoundaryPolicyBindingsResponse,
    UpdatePrincipalAccessBoundaryPolicyRequest,
)
from .types.principal_access_boundary_policy_resources import (
    PrincipalAccessBoundaryPolicy,
    PrincipalAccessBoundaryPolicyDetails,
    PrincipalAccessBoundaryPolicyRule,
)

__all__ = (
    "PolicyBindingsAsyncClient",
    "PrincipalAccessBoundaryPoliciesAsyncClient",
    "CreatePolicyBindingRequest",
    "CreatePrincipalAccessBoundaryPolicyRequest",
    "DeletePolicyBindingRequest",
    "DeletePrincipalAccessBoundaryPolicyRequest",
    "GetPolicyBindingRequest",
    "GetPrincipalAccessBoundaryPolicyRequest",
    "ListPolicyBindingsRequest",
    "ListPolicyBindingsResponse",
    "ListPrincipalAccessBoundaryPoliciesRequest",
    "ListPrincipalAccessBoundaryPoliciesResponse",
    "OperationMetadata",
    "PolicyBinding",
    "PolicyBindingsClient",
    "PrincipalAccessBoundaryPoliciesClient",
    "PrincipalAccessBoundaryPolicy",
    "PrincipalAccessBoundaryPolicyDetails",
    "PrincipalAccessBoundaryPolicyRule",
    "SearchPrincipalAccessBoundaryPolicyBindingsRequest",
    "SearchPrincipalAccessBoundaryPolicyBindingsResponse",
    "SearchTargetPolicyBindingsRequest",
    "SearchTargetPolicyBindingsResponse",
    "UpdatePolicyBindingRequest",
    "UpdatePrincipalAccessBoundaryPolicyRequest",
)
