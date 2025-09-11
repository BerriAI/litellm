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
from google.cloud.iam_v2beta import gapic_version as package_version

__version__ = package_version.__version__


from .services.policies import PoliciesAsyncClient, PoliciesClient
from .types.deny import DenyRule
from .types.policy import (
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
    "PoliciesAsyncClient",
    "CreatePolicyRequest",
    "DeletePolicyRequest",
    "DenyRule",
    "GetPolicyRequest",
    "ListPoliciesRequest",
    "ListPoliciesResponse",
    "PoliciesClient",
    "Policy",
    "PolicyOperationMetadata",
    "PolicyRule",
    "UpdatePolicyRequest",
)
