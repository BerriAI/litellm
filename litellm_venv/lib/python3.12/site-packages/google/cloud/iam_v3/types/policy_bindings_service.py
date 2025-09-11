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

from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.iam_v3.types import policy_binding_resources

__protobuf__ = proto.module(
    package="google.iam.v3",
    manifest={
        "CreatePolicyBindingRequest",
        "GetPolicyBindingRequest",
        "UpdatePolicyBindingRequest",
        "DeletePolicyBindingRequest",
        "ListPolicyBindingsRequest",
        "ListPolicyBindingsResponse",
        "SearchTargetPolicyBindingsRequest",
        "SearchTargetPolicyBindingsResponse",
    },
)


class CreatePolicyBindingRequest(proto.Message):
    r"""Request message for CreatePolicyBinding method.

    Attributes:
        parent (str):
            Required. The parent resource where this policy binding will
            be created. The binding parent is the closest Resource
            Manager resource (project, folder or organization) to the
            binding target.

            Format:

            -  ``projects/{project_id}/locations/{location}``
            -  ``projects/{project_number}/locations/{location}``
            -  ``folders/{folder_id}/locations/{location}``
            -  ``organizations/{organization_id}/locations/{location}``
        policy_binding_id (str):
            Required. The ID to use for the policy binding, which will
            become the final component of the policy binding's resource
            name.

            This value must start with a lowercase letter followed by up
            to 62 lowercase letters, numbers, hyphens, or dots. Pattern,
            /[a-z][a-z0-9-.]{2,62}/.
        policy_binding (google.cloud.iam_v3.types.PolicyBinding):
            Required. The policy binding to create.
        validate_only (bool):
            Optional. If set, validate the request and
            preview the creation, but do not actually post
            it.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    policy_binding_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    policy_binding: policy_binding_resources.PolicyBinding = proto.Field(
        proto.MESSAGE,
        number=3,
        message=policy_binding_resources.PolicyBinding,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class GetPolicyBindingRequest(proto.Message):
    r"""Request message for GetPolicyBinding method.

    Attributes:
        name (str):
            Required. The name of the policy binding to retrieve.

            Format:

            -  ``projects/{project_id}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``projects/{project_number}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``folders/{folder_id}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``organizations/{organization_id}/locations/{location}/policyBindings/{policy_binding_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdatePolicyBindingRequest(proto.Message):
    r"""Request message for UpdatePolicyBinding method.

    Attributes:
        policy_binding (google.cloud.iam_v3.types.PolicyBinding):
            Required. The policy binding to update.

            The policy binding's ``name`` field is used to identify the
            policy binding to update.
        validate_only (bool):
            Optional. If set, validate the request and
            preview the update, but do not actually post it.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. The list of fields to update
    """

    policy_binding: policy_binding_resources.PolicyBinding = proto.Field(
        proto.MESSAGE,
        number=1,
        message=policy_binding_resources.PolicyBinding,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=3,
        message=field_mask_pb2.FieldMask,
    )


class DeletePolicyBindingRequest(proto.Message):
    r"""Request message for DeletePolicyBinding method.

    Attributes:
        name (str):
            Required. The name of the policy binding to delete.

            Format:

            -  ``projects/{project_id}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``projects/{project_number}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``folders/{folder_id}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``organizations/{organization_id}/locations/{location}/policyBindings/{policy_binding_id}``
        etag (str):
            Optional. The etag of the policy binding.
            If this is provided, it must match the server's
            etag.
        validate_only (bool):
            Optional. If set, validate the request and
            preview the deletion, but do not actually post
            it.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=2,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class ListPolicyBindingsRequest(proto.Message):
    r"""Request message for ListPolicyBindings method.

    Attributes:
        parent (str):
            Required. The parent resource, which owns the collection of
            policy bindings.

            Format:

            -  ``projects/{project_id}/locations/{location}``
            -  ``projects/{project_number}/locations/{location}``
            -  ``folders/{folder_id}/locations/{location}``
            -  ``organizations/{organization_id}/locations/{location}``
        page_size (int):
            Optional. The maximum number of policy
            bindings to return. The service may return fewer
            than this value.

            If unspecified, at most 50 policy bindings will
            be returned. The maximum value is 1000; values
            above 1000 will be coerced to 1000.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListPolicyBindings`` call. Provide this to retrieve the
            subsequent page.

            When paginating, all other parameters provided to
            ``ListPolicyBindings`` must match the call that provided the
            page token.
        filter (str):
            Optional. An expression for filtering the results of the
            request. Filter rules are case insensitive. Some eligible
            fields for filtering are:

            -  ``target``
            -  ``policy``

            Some examples of filter queries:

            -  ``target:ex*``: The binding target's name starts with
               "ex".
            -  ``target:example``: The binding target's name is
               ``example``.
            -  ``policy:example``: The binding policy's name is
               ``example``.
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
    filter: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListPolicyBindingsResponse(proto.Message):
    r"""Response message for ListPolicyBindings method.

    Attributes:
        policy_bindings (MutableSequence[google.cloud.iam_v3.types.PolicyBinding]):
            The policy bindings from the specified
            parent.
        next_page_token (str):
            Optional. A token, which can be sent as ``page_token`` to
            retrieve the next page. If this field is omitted, there are
            no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    policy_bindings: MutableSequence[
        policy_binding_resources.PolicyBinding
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=policy_binding_resources.PolicyBinding,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SearchTargetPolicyBindingsRequest(proto.Message):
    r"""Request message for SearchTargetPolicyBindings method.

    Attributes:
        target (str):
            Required. The target resource, which is bound to the policy
            in the binding.

            Format:

            -  ``//iam.googleapis.com/locations/global/workforcePools/POOL_ID``
            -  ``//iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID``
            -  ``//iam.googleapis.com/locations/global/workspace/WORKSPACE_ID``
            -  ``//cloudresourcemanager.googleapis.com/projects/{project_number}``
            -  ``//cloudresourcemanager.googleapis.com/folders/{folder_id}``
            -  ``//cloudresourcemanager.googleapis.com/organizations/{organization_id}``
        page_size (int):
            Optional. The maximum number of policy
            bindings to return. The service may return fewer
            than this value.

            If unspecified, at most 50 policy bindings will
            be returned. The maximum value is 1000; values
            above 1000 will be coerced to 1000.
        page_token (str):
            Optional. A page token, received from a previous
            ``SearchTargetPolicyBindingsRequest`` call. Provide this to
            retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``SearchTargetPolicyBindingsRequest`` must match the call
            that provided the page token.
        parent (str):
            Required. The parent resource where this search will be
            performed. This should be the nearest Resource Manager
            resource (project, folder, or organization) to the target.

            Format:

            -  ``projects/{project_id}/locations/{location}``
            -  ``projects/{project_number}/locations/{location}``
            -  ``folders/{folder_id}/locations/{location}``
            -  ``organizations/{organization_id}/locations/{location}``
    """

    target: str = proto.Field(
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
    parent: str = proto.Field(
        proto.STRING,
        number=5,
    )


class SearchTargetPolicyBindingsResponse(proto.Message):
    r"""Response message for SearchTargetPolicyBindings method.

    Attributes:
        policy_bindings (MutableSequence[google.cloud.iam_v3.types.PolicyBinding]):
            The policy bindings bound to the specified
            target.
        next_page_token (str):
            Optional. A token, which can be sent as ``page_token`` to
            retrieve the next page. If this field is omitted, there are
            no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    policy_bindings: MutableSequence[
        policy_binding_resources.PolicyBinding
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=policy_binding_resources.PolicyBinding,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
