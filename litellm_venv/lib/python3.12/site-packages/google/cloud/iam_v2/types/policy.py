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

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.iam_v2.types import deny

__protobuf__ = proto.module(
    package="google.iam.v2",
    manifest={
        "Policy",
        "PolicyRule",
        "ListPoliciesRequest",
        "ListPoliciesResponse",
        "GetPolicyRequest",
        "CreatePolicyRequest",
        "UpdatePolicyRequest",
        "DeletePolicyRequest",
        "PolicyOperationMetadata",
    },
)


class Policy(proto.Message):
    r"""Data for an IAM policy.

    Attributes:
        name (str):
            Immutable. The resource name of the ``Policy``, which must
            be unique. Format:
            ``policies/{attachment_point}/denypolicies/{policy_id}``

            The attachment point is identified by its URL-encoded full
            resource name, which means that the forward-slash character,
            ``/``, must be written as ``%2F``. For example,
            ``policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-project/denypolicies/my-deny-policy``.

            For organizations and folders, use the numeric ID in the
            full resource name. For projects, requests can use the
            alphanumeric or the numeric ID. Responses always contain the
            numeric ID.
        uid (str):
            Immutable. The globally unique ID of the ``Policy``.
            Assigned automatically when the ``Policy`` is created.
        kind (str):
            Output only. The kind of the ``Policy``. Always contains the
            value ``DenyPolicy``.
        display_name (str):
            A user-specified description of the ``Policy``. This value
            can be up to 63 characters.
        annotations (MutableMapping[str, str]):
            A key-value map to store arbitrary metadata for the
            ``Policy``. Keys can be up to 63 characters. Values can be
            up to 255 characters.
        etag (str):
            An opaque tag that identifies the current version of the
            ``Policy``. IAM uses this value to help manage concurrent
            updates, so they do not cause one update to be overwritten
            by another.

            If this field is present in a [CreatePolicy][] request, the
            value is ignored.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time when the ``Policy`` was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time when the ``Policy`` was last updated.
        delete_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time when the ``Policy`` was deleted. Empty
            if the policy is not deleted.
        rules (MutableSequence[google.cloud.iam_v2.types.PolicyRule]):
            A list of rules that specify the behavior of the ``Policy``.
            All of the rules should be of the ``kind`` specified in the
            ``Policy``.
        managing_authority (str):
            Immutable. Specifies that this policy is
            managed by an authority and can only be modified
            by that authority. Usage is restricted.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    uid: str = proto.Field(
        proto.STRING,
        number=2,
    )
    kind: str = proto.Field(
        proto.STRING,
        number=3,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    annotations: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=5,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=6,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    delete_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    rules: MutableSequence["PolicyRule"] = proto.RepeatedField(
        proto.MESSAGE,
        number=10,
        message="PolicyRule",
    )
    managing_authority: str = proto.Field(
        proto.STRING,
        number=11,
    )


class PolicyRule(proto.Message):
    r"""A single rule in a ``Policy``.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        deny_rule (google.cloud.iam_v2.types.DenyRule):
            A rule for a deny policy.

            This field is a member of `oneof`_ ``kind``.
        description (str):
            A user-specified description of the rule.
            This value can be up to 256 characters.
    """

    deny_rule: deny.DenyRule = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="kind",
        message=deny.DenyRule,
    )
    description: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListPoliciesRequest(proto.Message):
    r"""Request message for ``ListPolicies``.

    Attributes:
        parent (str):
            Required. The resource that the policy is attached to, along
            with the kind of policy to list. Format:
            ``policies/{attachment_point}/denypolicies``

            The attachment point is identified by its URL-encoded full
            resource name, which means that the forward-slash character,
            ``/``, must be written as ``%2F``. For example,
            ``policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-project/denypolicies``.

            For organizations and folders, use the numeric ID in the
            full resource name. For projects, you can use the
            alphanumeric or the numeric ID.
        page_size (int):
            The maximum number of policies to return. IAM
            ignores this value and uses the value 1000.
        page_token (str):
            A page token received in a
            [ListPoliciesResponse][google.iam.v2.ListPoliciesResponse].
            Provide this token to retrieve the next page.
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


class ListPoliciesResponse(proto.Message):
    r"""Response message for ``ListPolicies``.

    Attributes:
        policies (MutableSequence[google.cloud.iam_v2.types.Policy]):
            Metadata for the policies that are attached
            to the resource.
        next_page_token (str):
            A page token that you can use in a
            [ListPoliciesRequest][google.iam.v2.ListPoliciesRequest] to
            retrieve the next page. If this field is omitted, there are
            no additional pages.
    """

    @property
    def raw_page(self):
        return self

    policies: MutableSequence["Policy"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Policy",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetPolicyRequest(proto.Message):
    r"""Request message for ``GetPolicy``.

    Attributes:
        name (str):
            Required. The resource name of the policy to retrieve.
            Format:
            ``policies/{attachment_point}/denypolicies/{policy_id}``

            Use the URL-encoded full resource name, which means that the
            forward-slash character, ``/``, must be written as ``%2F``.
            For example,
            ``policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-project/denypolicies/my-policy``.

            For organizations and folders, use the numeric ID in the
            full resource name. For projects, you can use the
            alphanumeric or the numeric ID.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreatePolicyRequest(proto.Message):
    r"""Request message for ``CreatePolicy``.

    Attributes:
        parent (str):
            Required. The resource that the policy is attached to, along
            with the kind of policy to create. Format:
            ``policies/{attachment_point}/denypolicies``

            The attachment point is identified by its URL-encoded full
            resource name, which means that the forward-slash character,
            ``/``, must be written as ``%2F``. For example,
            ``policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-project/denypolicies``.

            For organizations and folders, use the numeric ID in the
            full resource name. For projects, you can use the
            alphanumeric or the numeric ID.
        policy (google.cloud.iam_v2.types.Policy):
            Required. The policy to create.
        policy_id (str):
            The ID to use for this policy, which will become the final
            component of the policy's resource name. The ID must contain
            3 to 63 characters. It can contain lowercase letters and
            numbers, as well as dashes (``-``) and periods (``.``). The
            first character must be a lowercase letter.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    policy: "Policy" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="Policy",
    )
    policy_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class UpdatePolicyRequest(proto.Message):
    r"""Request message for ``UpdatePolicy``.

    Attributes:
        policy (google.cloud.iam_v2.types.Policy):
            Required. The policy to update.

            To prevent conflicting updates, the ``etag`` value must
            match the value that is stored in IAM. If the ``etag``
            values do not match, the request fails with a ``409`` error
            code and ``ABORTED`` status.
    """

    policy: "Policy" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="Policy",
    )


class DeletePolicyRequest(proto.Message):
    r"""Request message for ``DeletePolicy``.

    Attributes:
        name (str):
            Required. The resource name of the policy to delete. Format:
            ``policies/{attachment_point}/denypolicies/{policy_id}``

            Use the URL-encoded full resource name, which means that the
            forward-slash character, ``/``, must be written as ``%2F``.
            For example,
            ``policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-project/denypolicies/my-policy``.

            For organizations and folders, use the numeric ID in the
            full resource name. For projects, you can use the
            alphanumeric or the numeric ID.
        etag (str):
            Optional. The expected ``etag`` of the policy to delete. If
            the value does not match the value that is stored in IAM,
            the request fails with a ``409`` error code and ``ABORTED``
            status.

            If you omit this field, the policy is deleted regardless of
            its current ``etag``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=2,
    )


class PolicyOperationMetadata(proto.Message):
    r"""Metadata for long-running ``Policy`` operations.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Timestamp when the ``google.longrunning.Operation`` was
            created.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
