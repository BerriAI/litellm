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
from google.cloud.iam import gapic_version as package_version

__version__ = package_version.__version__


from google.cloud.iam_v2.services.policies.async_client import PoliciesAsyncClient
from google.cloud.iam_v2.services.policies.client import PoliciesClient
from google.cloud.iam_v2.types.deny import DenyRule
from google.cloud.iam_v2.types.policy import (
    CreatePolicyRequest,
    DeletePolicyRequest,
    GetPolicyRequest,
    ListPoliciesRequest,
    ListPoliciesResponse,
    Policy,
    PolicyOperationMetadata,
    PolicyRule,
    UpdatePolicyRequest,
)

__all__ = (
    "PoliciesClient",
    "PoliciesAsyncClient",
    "DenyRule",
    "CreatePolicyRequest",
    "DeletePolicyRequest",
    "GetPolicyRequest",
    "ListPoliciesRequest",
    "ListPoliciesResponse",
    "Policy",
    "PolicyOperationMetadata",
    "PolicyRule",
    "UpdatePolicyRequest",
)
