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
from google.type import expr_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.iam.v3beta",
    manifest={
        "PolicyBinding",
    },
)


class PolicyBinding(proto.Message):
    r"""IAM policy binding resource.

    Attributes:
        name (str):
            Identifier. The name of the policy binding, in the format
            ``{binding_parent/locations/{location}/policyBindings/{policy_binding_id}``.
            The binding parent is the closest Resource Manager resource
            (project, folder, or organization) to the binding target.

            Format:

            -  ``projects/{project_id}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``projects/{project_number}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``folders/{folder_id}/locations/{location}/policyBindings/{policy_binding_id}``
            -  ``organizations/{organization_id}/locations/{location}/policyBindings/{policy_binding_id}``
        uid (str):
            Output only. The globally unique ID of the
            policy binding. Assigned when the policy binding
            is created.
        etag (str):
            Optional. The etag for the policy binding.
            If this is provided on update, it must match the
            server's etag.
        display_name (str):
            Optional. The description of the policy
            binding. Must be less than or equal to 63
            characters.
        annotations (MutableMapping[str, str]):
            Optional. User-defined annotations. See
            https://google.aip.dev/148#annotations for more
            details such as format and size limitations
        target (google.cloud.iam_v3beta.types.PolicyBinding.Target):
            Required. Immutable. Target is the full
            resource name of the resource to which the
            policy will be bound. Immutable once set.
        policy_kind (google.cloud.iam_v3beta.types.PolicyBinding.PolicyKind):
            Immutable. The kind of the policy to attach
            in this binding. This field must be one of the
            following:

            - Left empty (will be automatically set to the
              policy kind)
            - The input policy kind
        policy (str):
            Required. Immutable. The resource name of the
            policy to be bound. The binding parent and
            policy must belong to the same organization.
        policy_uid (str):
            Output only. The globally unique ID of the
            policy to be bound.
        condition (google.type.expr_pb2.Expr):
            Optional. The condition to apply to the policy binding. When
            set, the ``expression`` field in the ``Expr`` must include
            from 1 to 10 subexpressions, joined by the "||"(Logical OR),
            "&&"(Logical AND) or "!"(Logical NOT) operators and cannot
            contain more than 250 characters.

            The condition is currently only supported when bound to
            policies of kind principal access boundary.

            When the bound policy is a principal access boundary policy,
            the only supported attributes in any subexpression are
            ``principal.type`` and ``principal.subject``. An example
            expression is: "principal.type ==
            'iam.googleapis.com/ServiceAccount'" or "principal.subject
            == 'bob@example.com'".

            Allowed operations for ``principal.subject``:

            -  ``principal.subject == <principal subject string>``
            -  ``principal.subject != <principal subject string>``
            -  ``principal.subject in [<list of principal subjects>]``
            -  ``principal.subject.startsWith(<string>)``
            -  ``principal.subject.endsWith(<string>)``

            Allowed operations for ``principal.type``:

            -  ``principal.type == <principal type string>``
            -  ``principal.type != <principal type string>``
            -  ``principal.type in [<list of principal types>]``

            Supported principal types are Workspace, Workforce Pool,
            Workload Pool and Service Account. Allowed string must be
            one of:

            -  iam.googleapis.com/WorkspaceIdentity
            -  iam.googleapis.com/WorkforcePoolIdentity
            -  iam.googleapis.com/WorkloadPoolIdentity
            -  iam.googleapis.com/ServiceAccount
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time when the policy binding
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time when the policy binding
            was most recently updated.
    """

    class PolicyKind(proto.Enum):
        r"""Different policy kinds supported in this binding.

        Values:
            POLICY_KIND_UNSPECIFIED (0):
                Unspecified policy kind; Not a valid state
            PRINCIPAL_ACCESS_BOUNDARY (1):
                Principal access boundary policy kind
        """
        POLICY_KIND_UNSPECIFIED = 0
        PRINCIPAL_ACCESS_BOUNDARY = 1

    class Target(proto.Message):
        r"""Target is the full resource name of the resource to which the
        policy will be bound. Immutable once set.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            principal_set (str):
                Immutable. Full Resource Name used for principal access
                boundary policy bindings. The principal set must be directly
                parented by the policy binding's parent or same as the
                parent if the target is a project/folder/organization.

                Examples:

                -  For binding's parented by an organization:

                   -  Organization:
                      ``//cloudresourcemanager.googleapis.com/organizations/ORGANIZATION_ID``
                   -  Workforce Identity:
                      ``//iam.googleapis.com/locations/global/workforcePools/WORKFORCE_POOL_ID``
                   -  Workspace Identity:
                      ``//iam.googleapis.com/locations/global/workspace/WORKSPACE_ID``

                -  For binding's parented by a folder:

                   -  Folder:
                      ``//cloudresourcemanager.googleapis.com/folders/FOLDER_ID``

                -  For binding's parented by a project:

                   -  Project:

                      -  ``//cloudresourcemanager.googleapis.com/projects/PROJECT_NUMBER``
                      -  ``//cloudresourcemanager.googleapis.com/projects/PROJECT_ID``

                   -  Workload Identity Pool:
                      ``//iam.googleapis.com/projects/PROJECT_NUMBER/locations/LOCATION/workloadIdentityPools/WORKLOAD_POOL_ID``

                This field is a member of `oneof`_ ``target``.
        """

        principal_set: str = proto.Field(
            proto.STRING,
            number=1,
            oneof="target",
        )

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
    target: Target = proto.Field(
        proto.MESSAGE,
        number=6,
        message=Target,
    )
    policy_kind: PolicyKind = proto.Field(
        proto.ENUM,
        number=11,
        enum=PolicyKind,
    )
    policy: str = proto.Field(
        proto.STRING,
        number=7,
    )
    policy_uid: str = proto.Field(
        proto.STRING,
        number=12,
    )
    condition: expr_pb2.Expr = proto.Field(
        proto.MESSAGE,
        number=8,
        message=expr_pb2.Expr,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=10,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
