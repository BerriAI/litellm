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

from google.type import expr_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.iam.v2",
    manifest={
        "DenyRule",
    },
)


class DenyRule(proto.Message):
    r"""A deny rule in an IAM deny policy.

    Attributes:
        denied_principals (MutableSequence[str]):
            The identities that are prevented from using one or more
            permissions on Google Cloud resources. This field can
            contain the following values:

            -  ``principalSet://goog/public:all``: A special identifier
               that represents any principal that is on the internet,
               even if they do not have a Google Account or are not
               logged in.

            -  ``principal://goog/subject/{email_id}``: A specific
               Google Account. Includes Gmail, Cloud Identity, and
               Google Workspace user accounts. For example,
               ``principal://goog/subject/alice@example.com``.

            -  ``deleted:principal://goog/subject/{email_id}?uid={uid}``:
               A specific Google Account that was deleted recently. For
               example,
               ``deleted:principal://goog/subject/alice@example.com?uid=1234567890``.
               If the Google Account is recovered, this identifier
               reverts to the standard identifier for a Google Account.

            -  ``principalSet://goog/group/{group_id}``: A Google group.
               For example,
               ``principalSet://goog/group/admins@example.com``.

            -  ``deleted:principalSet://goog/group/{group_id}?uid={uid}``:
               A Google group that was deleted recently. For example,
               ``deleted:principalSet://goog/group/admins@example.com?uid=1234567890``.
               If the Google group is restored, this identifier reverts
               to the standard identifier for a Google group.

            -  ``principal://iam.googleapis.com/projects/-/serviceAccounts/{service_account_id}``:
               A Google Cloud service account. For example,
               ``principal://iam.googleapis.com/projects/-/serviceAccounts/my-service-account@iam.gserviceaccount.com``.

            -  ``deleted:principal://iam.googleapis.com/projects/-/serviceAccounts/{service_account_id}?uid={uid}``:
               A Google Cloud service account that was deleted recently.
               For example,
               ``deleted:principal://iam.googleapis.com/projects/-/serviceAccounts/my-service-account@iam.gserviceaccount.com?uid=1234567890``.
               If the service account is undeleted, this identifier
               reverts to the standard identifier for a service account.

            -  ``principalSet://goog/cloudIdentityCustomerId/{customer_id}``:
               All of the principals associated with the specified
               Google Workspace or Cloud Identity customer ID. For
               example,
               ``principalSet://goog/cloudIdentityCustomerId/C01Abc35``.
        exception_principals (MutableSequence[str]):
            The identities that are excluded from the deny rule, even if
            they are listed in the ``denied_principals``. For example,
            you could add a Google group to the ``denied_principals``,
            then exclude specific users who belong to that group.

            This field can contain the same values as the
            ``denied_principals`` field, excluding
            ``principalSet://goog/public:all``, which represents all
            users on the internet.
        denied_permissions (MutableSequence[str]):
            The permissions that are explicitly denied by this rule.
            Each permission uses the format
            ``{service_fqdn}/{resource}.{verb}``, where
            ``{service_fqdn}`` is the fully qualified domain name for
            the service. For example, ``iam.googleapis.com/roles.list``.
        exception_permissions (MutableSequence[str]):
            Specifies the permissions that this rule excludes from the
            set of denied permissions given by ``denied_permissions``.
            If a permission appears in ``denied_permissions`` *and* in
            ``exception_permissions`` then it will *not* be denied.

            The excluded permissions can be specified using the same
            syntax as ``denied_permissions``.
        denial_condition (google.type.expr_pb2.Expr):
            The condition that determines whether this deny rule applies
            to a request. If the condition expression evaluates to
            ``true``, then the deny rule is applied; otherwise, the deny
            rule is not applied.

            Each deny rule is evaluated independently. If this deny rule
            does not apply to a request, other deny rules might still
            apply.

            The condition can use CEL functions that evaluate `resource
            tags <https://cloud.google.com/iam/help/conditions/resource-tags>`__.
            Other functions and operators are not supported.
    """

    denied_principals: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    exception_principals: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    denied_permissions: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )
    exception_permissions: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=4,
    )
    denial_condition: expr_pb2.Expr = proto.Field(
        proto.MESSAGE,
        number=5,
        message=expr_pb2.Expr,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
