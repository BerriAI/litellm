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

from google.cloud.iam_v3beta.types import (
    policy_binding_resources,
    principal_access_boundary_policy_resources,
)

__protobuf__ = proto.module(
    package="google.iam.v3beta",
    manifest={
        "CreatePrincipalAccessBoundaryPolicyRequest",
        "GetPrincipalAccessBoundaryPolicyRequest",
        "UpdatePrincipalAccessBoundaryPolicyRequest",
        "DeletePrincipalAccessBoundaryPolicyRequest",
        "ListPrincipalAccessBoundaryPoliciesRequest",
        "ListPrincipalAccessBoundaryPoliciesResponse",
        "SearchPrincipalAccessBoundaryPolicyBindingsRequest",
        "SearchPrincipalAccessBoundaryPolicyBindingsResponse",
    },
)


class CreatePrincipalAccessBoundaryPolicyRequest(proto.Message):
    r"""Request message for
    CreatePrincipalAccessBoundaryPolicyRequest method.

    Attributes:
        parent (str):
            Required. The parent resource where this principal access
            boundary policy will be created. Only organizations are
            supported.

            Format:
            ``organizations/{organization_id}/locations/{location}``
        principal_access_boundary_policy_id (str):
            Required. The ID to use for the principal access boundary
            policy, which will become the final component of the
            principal access boundary policy's resource name.

            This value must start with a lowercase letter followed by up
            to 62 lowercase letters, numbers, hyphens, or dots. Pattern,
            /[a-z][a-z0-9-.]{2,62}/.
        principal_access_boundary_policy (google.cloud.iam_v3beta.types.PrincipalAccessBoundaryPolicy):
            Required. The principal access boundary
            policy to create.
        validate_only (bool):
            Optional. If set, validate the request and
            preview the creation, but do not actually post
            it.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    principal_access_boundary_policy_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    principal_access_boundary_policy: principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy = proto.Field(
        proto.MESSAGE,
        number=3,
        message=principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class GetPrincipalAccessBoundaryPolicyRequest(proto.Message):
    r"""Request message for GetPrincipalAccessBoundaryPolicy method.

    Attributes:
        name (str):
            Required. The name of the principal access boundary policy
            to retrieve.

            Format:
            ``organizations/{organization_id}/locations/{location}/principalAccessBoundaryPolicies/{principal_access_boundary_policy_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdatePrincipalAccessBoundaryPolicyRequest(proto.Message):
    r"""Request message for UpdatePrincipalAccessBoundaryPolicy
    method.

    Attributes:
        principal_access_boundary_policy (google.cloud.iam_v3beta.types.PrincipalAccessBoundaryPolicy):
            Required. The principal access boundary policy to update.

            The principal access boundary policy's ``name`` field is
            used to identify the policy to update.
        validate_only (bool):
            Optional. If set, validate the request and
            preview the update, but do not actually post it.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. The list of fields to update
    """

    principal_access_boundary_policy: principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy = proto.Field(
        proto.MESSAGE,
        number=1,
        message=principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
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


class DeletePrincipalAccessBoundaryPolicyRequest(proto.Message):
    r"""Request message for DeletePrincipalAccessBoundaryPolicy
    method.

    Attributes:
        name (str):
            Required. The name of the principal access boundary policy
            to delete.

            Format:
            ``organizations/{organization_id}/locations/{location}/principalAccessBoundaryPolicies/{principal_access_boundary_policy_id}``
        etag (str):
            Optional. The etag of the principal access
            boundary policy. If this is provided, it must
            match the server's etag.
        validate_only (bool):
            Optional. If set, validate the request and
            preview the deletion, but do not actually post
            it.
        force (bool):
            Optional. If set to true, the request will
            force the deletion of the policy even if the
            policy is referenced in policy bindings.
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
    force: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class ListPrincipalAccessBoundaryPoliciesRequest(proto.Message):
    r"""Request message for ListPrincipalAccessBoundaryPolicies
    method.

    Attributes:
        parent (str):
            Required. The parent resource, which owns the collection of
            principal access boundary policies.

            Format:
            ``organizations/{organization_id}/locations/{location}``
        page_size (int):
            Optional. The maximum number of principal
            access boundary policies to return. The service
            may return fewer than this value.

            If unspecified, at most 50 principal access
            boundary policies will be returned. The maximum
            value is 1000; values above 1000 will be coerced
            to 1000.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListPrincipalAccessBoundaryPolicies`` call. Provide this
            to retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``ListPrincipalAccessBoundaryPolicies`` must match the call
            that provided the page token.
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


class ListPrincipalAccessBoundaryPoliciesResponse(proto.Message):
    r"""Response message for ListPrincipalAccessBoundaryPolicies
    method.

    Attributes:
        principal_access_boundary_policies (MutableSequence[google.cloud.iam_v3beta.types.PrincipalAccessBoundaryPolicy]):
            The principal access boundary policies from
            the specified parent.
        next_page_token (str):
            Optional. A token, which can be sent as ``page_token`` to
            retrieve the next page. If this field is omitted, there are
            no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    principal_access_boundary_policies: MutableSequence[
        principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SearchPrincipalAccessBoundaryPolicyBindingsRequest(proto.Message):
    r"""Request message for
    SearchPrincipalAccessBoundaryPolicyBindings rpc.

    Attributes:
        name (str):
            Required. The name of the principal access boundary policy.
            Format:
            ``organizations/{organization_id}/locations/{location}/principalAccessBoundaryPolicies/{principal_access_boundary_policy_id}``
        page_size (int):
            Optional. The maximum number of policy
            bindings to return. The service may return fewer
            than this value.

            If unspecified, at most 50 policy bindings will
            be returned. The maximum value is 1000; values
            above 1000 will be coerced to 1000.
        page_token (str):
            Optional. A page token, received from a previous
            ``SearchPrincipalAccessBoundaryPolicyBindingsRequest`` call.
            Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``SearchPrincipalAccessBoundaryPolicyBindingsRequest`` must
            match the call that provided the page token.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class SearchPrincipalAccessBoundaryPolicyBindingsResponse(proto.Message):
    r"""Response message for
    SearchPrincipalAccessBoundaryPolicyBindings rpc.

    Attributes:
        policy_bindings (MutableSequence[google.cloud.iam_v3beta.types.PolicyBinding]):
            The policy bindings that reference the
            specified policy.
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
