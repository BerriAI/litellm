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

__protobuf__ = proto.module(
    package="google.iam.v3",
    manifest={
        "PrincipalAccessBoundaryPolicy",
        "PrincipalAccessBoundaryPolicyDetails",
        "PrincipalAccessBoundaryPolicyRule",
    },
)


class PrincipalAccessBoundaryPolicy(proto.Message):
    r"""An IAM principal access boundary policy resource.

    Attributes:
        name (str):
            Identifier. The resource name of the principal access
            boundary policy.

            The following format is supported:
            ``organizations/{organization_id}/locations/{location}/principalAccessBoundaryPolicies/{policy_id}``
        uid (str):
            Output only. The globally unique ID of the
            principal access boundary policy.
        etag (str):
            Optional. The etag for the principal access
            boundary. If this is provided on update, it must
            match the server's etag.
        display_name (str):
            Optional. The description of the principal
            access boundary policy. Must be less than or
            equal to 63 characters.
        annotations (MutableMapping[str, str]):
            Optional. User defined annotations. See
            https://google.aip.dev/148#annotations for more
            details such as format and size limitations
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time when the principal
            access boundary policy was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time when the principal
            access boundary policy was most recently
            updated.
        details (google.cloud.iam_v3.types.PrincipalAccessBoundaryPolicyDetails):
            Optional. The details for the principal
            access boundary policy.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    uid: str = proto.Field(
        proto.STRING,
        number=2,
    )
    etag: str = proto.Field(
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
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    details: "PrincipalAccessBoundaryPolicyDetails" = proto.Field(
        proto.MESSAGE,
        number=8,
        message="PrincipalAccessBoundaryPolicyDetails",
    )


class PrincipalAccessBoundaryPolicyDetails(proto.Message):
    r"""Principal access boundary policy details

    Attributes:
        rules (MutableSequence[google.cloud.iam_v3.types.PrincipalAccessBoundaryPolicyRule]):
            Required. A list of principal access boundary
            policy rules. The number of rules in a policy is
            limited to 500.
        enforcement_version (str):
            Optional. The version number (for example, ``1`` or
            ``latest``) that indicates which permissions are able to be
            blocked by the policy. If empty, the PAB policy version will
            be set to the most recent version number at the time of the
            policy's creation.
    """

    rules: MutableSequence["PrincipalAccessBoundaryPolicyRule"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="PrincipalAccessBoundaryPolicyRule",
    )
    enforcement_version: str = proto.Field(
        proto.STRING,
        number=4,
    )


class PrincipalAccessBoundaryPolicyRule(proto.Message):
    r"""Principal access boundary policy rule that defines the
    resource boundary.

    Attributes:
        description (str):
            Optional. The description of the principal
            access boundary policy rule. Must be less than
            or equal to 256 characters.
        resources (MutableSequence[str]):
            Required. A list of Resource Manager resources. If a
            resource is listed in the rule, then the rule applies for
            that resource and its descendants. The number of resources
            in a policy is limited to 500 across all rules in the
            policy.

            The following resource types are supported:

            -  Organizations, such as
               ``//cloudresourcemanager.googleapis.com/organizations/123``.
            -  Folders, such as
               ``//cloudresourcemanager.googleapis.com/folders/123``.
            -  Projects, such as
               ``//cloudresourcemanager.googleapis.com/projects/123`` or
               ``//cloudresourcemanager.googleapis.com/projects/my-project-id``.
        effect (google.cloud.iam_v3.types.PrincipalAccessBoundaryPolicyRule.Effect):
            Required. The access relationship of
            principals to the resources in this rule.
    """

    class Effect(proto.Enum):
        r"""An effect to describe the access relationship.

        Values:
            EFFECT_UNSPECIFIED (0):
                Effect unspecified.
            ALLOW (1):
                Allows access to the resources in this rule.
        """
        EFFECT_UNSPECIFIED = 0
        ALLOW = 1

    description: str = proto.Field(
        proto.STRING,
        number=1,
    )
    resources: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    effect: Effect = proto.Field(
        proto.ENUM,
        number=3,
        enum=Effect,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
