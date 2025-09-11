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
import abc
from typing import Awaitable, Callable, Dict, Optional, Sequence, Union

import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf
from google.protobuf import empty_pb2  # type: ignore

from google.cloud.iam_admin_v1 import gapic_version as package_version
from google.cloud.iam_admin_v1.types import iam

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class IAMTransport(abc.ABC):
    """Abstract transport class for IAM."""

    AUTH_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

    DEFAULT_HOST: str = "iam.googleapis.com"

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        api_audience: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'iam.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A list of scopes.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
        """

        scopes_kwargs = {"scopes": scopes, "default_scopes": self.AUTH_SCOPES}

        # Save the scopes.
        self._scopes = scopes
        if not hasattr(self, "_ignore_credentials"):
            self._ignore_credentials: bool = False

        # If no credentials are provided, then determine the appropriate
        # defaults.
        if credentials and credentials_file:
            raise core_exceptions.DuplicateCredentialArgs(
                "'credentials_file' and 'credentials' are mutually exclusive"
            )

        if credentials_file is not None:
            credentials, _ = google.auth.load_credentials_from_file(
                credentials_file, **scopes_kwargs, quota_project_id=quota_project_id
            )
        elif credentials is None and not self._ignore_credentials:
            credentials, _ = google.auth.default(
                **scopes_kwargs, quota_project_id=quota_project_id
            )
            # Don't apply audience if the credentials file passed from user.
            if hasattr(credentials, "with_gdch_audience"):
                credentials = credentials.with_gdch_audience(
                    api_audience if api_audience else host
                )

        # If the credentials are service account credentials, then always try to use self signed JWT.
        if (
            always_use_jwt_access
            and isinstance(credentials, service_account.Credentials)
            and hasattr(service_account.Credentials, "with_always_use_jwt_access")
        ):
            credentials = credentials.with_always_use_jwt_access(True)

        # Save the credentials.
        self._credentials = credentials

        # Save the hostname. Default to port 443 (HTTPS) if none is specified.
        if ":" not in host:
            host += ":443"
        self._host = host

    @property
    def host(self):
        return self._host

    def _prep_wrapped_messages(self, client_info):
        # Precompute the wrapped methods.
        self._wrapped_methods = {
            self.list_service_accounts: gapic_v1.method.wrap_method(
                self.list_service_accounts,
                default_retry=retries.Retry(
                    initial=0.1,
                    maximum=60.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_service_account: gapic_v1.method.wrap_method(
                self.get_service_account,
                default_retry=retries.Retry(
                    initial=0.1,
                    maximum=60.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.create_service_account: gapic_v1.method.wrap_method(
                self.create_service_account,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.update_service_account: gapic_v1.method.wrap_method(
                self.update_service_account,
                default_retry=retries.Retry(
                    initial=0.1,
                    maximum=60.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.patch_service_account: gapic_v1.method.wrap_method(
                self.patch_service_account,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_service_account: gapic_v1.method.wrap_method(
                self.delete_service_account,
                default_retry=retries.Retry(
                    initial=0.1,
                    maximum=60.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.undelete_service_account: gapic_v1.method.wrap_method(
                self.undelete_service_account,
                default_timeout=None,
                client_info=client_info,
            ),
            self.enable_service_account: gapic_v1.method.wrap_method(
                self.enable_service_account,
                default_timeout=None,
                client_info=client_info,
            ),
            self.disable_service_account: gapic_v1.method.wrap_method(
                self.disable_service_account,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_service_account_keys: gapic_v1.method.wrap_method(
                self.list_service_account_keys,
                default_retry=retries.Retry(
                    initial=0.1,
                    maximum=60.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_service_account_key: gapic_v1.method.wrap_method(
                self.get_service_account_key,
                default_retry=retries.Retry(
                    initial=0.1,
                    maximum=60.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.create_service_account_key: gapic_v1.method.wrap_method(
                self.create_service_account_key,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.upload_service_account_key: gapic_v1.method.wrap_method(
                self.upload_service_account_key,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_service_account_key: gapic_v1.method.wrap_method(
                self.delete_service_account_key,
                default_retry=retries.Retry(
                    initial=0.1,
                    maximum=60.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.disable_service_account_key: gapic_v1.method.wrap_method(
                self.disable_service_account_key,
                default_timeout=None,
                client_info=client_info,
            ),
            self.enable_service_account_key: gapic_v1.method.wrap_method(
                self.enable_service_account_key,
                default_timeout=None,
                client_info=client_info,
            ),
            self.sign_blob: gapic_v1.method.wrap_method(
                self.sign_blob,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.sign_jwt: gapic_v1.method.wrap_method(
                self.sign_jwt,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_iam_policy: gapic_v1.method.wrap_method(
                self.get_iam_policy,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.set_iam_policy: gapic_v1.method.wrap_method(
                self.set_iam_policy,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.test_iam_permissions: gapic_v1.method.wrap_method(
                self.test_iam_permissions,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.query_grantable_roles: gapic_v1.method.wrap_method(
                self.query_grantable_roles,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.list_roles: gapic_v1.method.wrap_method(
                self.list_roles,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_role: gapic_v1.method.wrap_method(
                self.get_role,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_role: gapic_v1.method.wrap_method(
                self.create_role,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_role: gapic_v1.method.wrap_method(
                self.update_role,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_role: gapic_v1.method.wrap_method(
                self.delete_role,
                default_timeout=None,
                client_info=client_info,
            ),
            self.undelete_role: gapic_v1.method.wrap_method(
                self.undelete_role,
                default_timeout=None,
                client_info=client_info,
            ),
            self.query_testable_permissions: gapic_v1.method.wrap_method(
                self.query_testable_permissions,
                default_timeout=None,
                client_info=client_info,
            ),
            self.query_auditable_services: gapic_v1.method.wrap_method(
                self.query_auditable_services,
                default_timeout=None,
                client_info=client_info,
            ),
            self.lint_policy: gapic_v1.method.wrap_method(
                self.lint_policy,
                default_timeout=None,
                client_info=client_info,
            ),
        }

    def close(self):
        """Closes resources associated with the transport.

        .. warning::
             Only call this method if the transport is NOT shared
             with other clients - this may cause errors in other clients!
        """
        raise NotImplementedError()

    @property
    def list_service_accounts(
        self,
    ) -> Callable[
        [iam.ListServiceAccountsRequest],
        Union[
            iam.ListServiceAccountsResponse, Awaitable[iam.ListServiceAccountsResponse]
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_service_account(
        self,
    ) -> Callable[
        [iam.GetServiceAccountRequest],
        Union[iam.ServiceAccount, Awaitable[iam.ServiceAccount]],
    ]:
        raise NotImplementedError()

    @property
    def create_service_account(
        self,
    ) -> Callable[
        [iam.CreateServiceAccountRequest],
        Union[iam.ServiceAccount, Awaitable[iam.ServiceAccount]],
    ]:
        raise NotImplementedError()

    @property
    def update_service_account(
        self,
    ) -> Callable[
        [iam.ServiceAccount], Union[iam.ServiceAccount, Awaitable[iam.ServiceAccount]]
    ]:
        raise NotImplementedError()

    @property
    def patch_service_account(
        self,
    ) -> Callable[
        [iam.PatchServiceAccountRequest],
        Union[iam.ServiceAccount, Awaitable[iam.ServiceAccount]],
    ]:
        raise NotImplementedError()

    @property
    def delete_service_account(
        self,
    ) -> Callable[
        [iam.DeleteServiceAccountRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def undelete_service_account(
        self,
    ) -> Callable[
        [iam.UndeleteServiceAccountRequest],
        Union[
            iam.UndeleteServiceAccountResponse,
            Awaitable[iam.UndeleteServiceAccountResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def enable_service_account(
        self,
    ) -> Callable[
        [iam.EnableServiceAccountRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def disable_service_account(
        self,
    ) -> Callable[
        [iam.DisableServiceAccountRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def list_service_account_keys(
        self,
    ) -> Callable[
        [iam.ListServiceAccountKeysRequest],
        Union[
            iam.ListServiceAccountKeysResponse,
            Awaitable[iam.ListServiceAccountKeysResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_service_account_key(
        self,
    ) -> Callable[
        [iam.GetServiceAccountKeyRequest],
        Union[iam.ServiceAccountKey, Awaitable[iam.ServiceAccountKey]],
    ]:
        raise NotImplementedError()

    @property
    def create_service_account_key(
        self,
    ) -> Callable[
        [iam.CreateServiceAccountKeyRequest],
        Union[iam.ServiceAccountKey, Awaitable[iam.ServiceAccountKey]],
    ]:
        raise NotImplementedError()

    @property
    def upload_service_account_key(
        self,
    ) -> Callable[
        [iam.UploadServiceAccountKeyRequest],
        Union[iam.ServiceAccountKey, Awaitable[iam.ServiceAccountKey]],
    ]:
        raise NotImplementedError()

    @property
    def delete_service_account_key(
        self,
    ) -> Callable[
        [iam.DeleteServiceAccountKeyRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def disable_service_account_key(
        self,
    ) -> Callable[
        [iam.DisableServiceAccountKeyRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def enable_service_account_key(
        self,
    ) -> Callable[
        [iam.EnableServiceAccountKeyRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def sign_blob(
        self,
    ) -> Callable[
        [iam.SignBlobRequest],
        Union[iam.SignBlobResponse, Awaitable[iam.SignBlobResponse]],
    ]:
        raise NotImplementedError()

    @property
    def sign_jwt(
        self,
    ) -> Callable[
        [iam.SignJwtRequest], Union[iam.SignJwtResponse, Awaitable[iam.SignJwtResponse]]
    ]:
        raise NotImplementedError()

    @property
    def get_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.GetIamPolicyRequest],
        Union[policy_pb2.Policy, Awaitable[policy_pb2.Policy]],
    ]:
        raise NotImplementedError()

    @property
    def set_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.SetIamPolicyRequest],
        Union[policy_pb2.Policy, Awaitable[policy_pb2.Policy]],
    ]:
        raise NotImplementedError()

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        Union[
            iam_policy_pb2.TestIamPermissionsResponse,
            Awaitable[iam_policy_pb2.TestIamPermissionsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def query_grantable_roles(
        self,
    ) -> Callable[
        [iam.QueryGrantableRolesRequest],
        Union[
            iam.QueryGrantableRolesResponse, Awaitable[iam.QueryGrantableRolesResponse]
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_roles(
        self,
    ) -> Callable[
        [iam.ListRolesRequest],
        Union[iam.ListRolesResponse, Awaitable[iam.ListRolesResponse]],
    ]:
        raise NotImplementedError()

    @property
    def get_role(
        self,
    ) -> Callable[[iam.GetRoleRequest], Union[iam.Role, Awaitable[iam.Role]]]:
        raise NotImplementedError()

    @property
    def create_role(
        self,
    ) -> Callable[[iam.CreateRoleRequest], Union[iam.Role, Awaitable[iam.Role]]]:
        raise NotImplementedError()

    @property
    def update_role(
        self,
    ) -> Callable[[iam.UpdateRoleRequest], Union[iam.Role, Awaitable[iam.Role]]]:
        raise NotImplementedError()

    @property
    def delete_role(
        self,
    ) -> Callable[[iam.DeleteRoleRequest], Union[iam.Role, Awaitable[iam.Role]]]:
        raise NotImplementedError()

    @property
    def undelete_role(
        self,
    ) -> Callable[[iam.UndeleteRoleRequest], Union[iam.Role, Awaitable[iam.Role]]]:
        raise NotImplementedError()

    @property
    def query_testable_permissions(
        self,
    ) -> Callable[
        [iam.QueryTestablePermissionsRequest],
        Union[
            iam.QueryTestablePermissionsResponse,
            Awaitable[iam.QueryTestablePermissionsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def query_auditable_services(
        self,
    ) -> Callable[
        [iam.QueryAuditableServicesRequest],
        Union[
            iam.QueryAuditableServicesResponse,
            Awaitable[iam.QueryAuditableServicesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def lint_policy(
        self,
    ) -> Callable[
        [iam.LintPolicyRequest],
        Union[iam.LintPolicyResponse, Awaitable[iam.LintPolicyResponse]],
    ]:
        raise NotImplementedError()

    @property
    def kind(self) -> str:
        raise NotImplementedError()


__all__ = ("IAMTransport",)
