# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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

from google.cloud.aiplatform_v1beta1 import gapic_version as package_version

import google.auth  # type: ignore
import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.api_core import operations_v1
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore

from google.cloud.aiplatform_v1beta1.types import study
from google.cloud.aiplatform_v1beta1.types import study as gca_study
from google.cloud.aiplatform_v1beta1.types import vizier_service
from google.cloud.location import locations_pb2  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)


class VizierServiceTransport(abc.ABC):
    """Abstract transport class for VizierService."""

    AUTH_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

    DEFAULT_HOST: str = "aiplatform.googleapis.com"

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
                 The hostname to connect to (default: 'aiplatform.googleapis.com').
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
        elif credentials is None:
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
            self.create_study: gapic_v1.method.wrap_method(
                self.create_study,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_study: gapic_v1.method.wrap_method(
                self.get_study,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_studies: gapic_v1.method.wrap_method(
                self.list_studies,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.delete_study: gapic_v1.method.wrap_method(
                self.delete_study,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.lookup_study: gapic_v1.method.wrap_method(
                self.lookup_study,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.suggest_trials: gapic_v1.method.wrap_method(
                self.suggest_trials,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.create_trial: gapic_v1.method.wrap_method(
                self.create_trial,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_trial: gapic_v1.method.wrap_method(
                self.get_trial,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_trials: gapic_v1.method.wrap_method(
                self.list_trials,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.add_trial_measurement: gapic_v1.method.wrap_method(
                self.add_trial_measurement,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.complete_trial: gapic_v1.method.wrap_method(
                self.complete_trial,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.delete_trial: gapic_v1.method.wrap_method(
                self.delete_trial,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.check_trial_early_stopping_state: gapic_v1.method.wrap_method(
                self.check_trial_early_stopping_state,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.stop_trial: gapic_v1.method.wrap_method(
                self.stop_trial,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_optimal_trials: gapic_v1.method.wrap_method(
                self.list_optimal_trials,
                default_timeout=5.0,
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
    def operations_client(self):
        """Return the client designed to process long-running operations."""
        raise NotImplementedError()

    @property
    def create_study(
        self,
    ) -> Callable[
        [vizier_service.CreateStudyRequest],
        Union[gca_study.Study, Awaitable[gca_study.Study]],
    ]:
        raise NotImplementedError()

    @property
    def get_study(
        self,
    ) -> Callable[
        [vizier_service.GetStudyRequest], Union[study.Study, Awaitable[study.Study]]
    ]:
        raise NotImplementedError()

    @property
    def list_studies(
        self,
    ) -> Callable[
        [vizier_service.ListStudiesRequest],
        Union[
            vizier_service.ListStudiesResponse,
            Awaitable[vizier_service.ListStudiesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_study(
        self,
    ) -> Callable[
        [vizier_service.DeleteStudyRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def lookup_study(
        self,
    ) -> Callable[
        [vizier_service.LookupStudyRequest], Union[study.Study, Awaitable[study.Study]]
    ]:
        raise NotImplementedError()

    @property
    def suggest_trials(
        self,
    ) -> Callable[
        [vizier_service.SuggestTrialsRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def create_trial(
        self,
    ) -> Callable[
        [vizier_service.CreateTrialRequest], Union[study.Trial, Awaitable[study.Trial]]
    ]:
        raise NotImplementedError()

    @property
    def get_trial(
        self,
    ) -> Callable[
        [vizier_service.GetTrialRequest], Union[study.Trial, Awaitable[study.Trial]]
    ]:
        raise NotImplementedError()

    @property
    def list_trials(
        self,
    ) -> Callable[
        [vizier_service.ListTrialsRequest],
        Union[
            vizier_service.ListTrialsResponse,
            Awaitable[vizier_service.ListTrialsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def add_trial_measurement(
        self,
    ) -> Callable[
        [vizier_service.AddTrialMeasurementRequest],
        Union[study.Trial, Awaitable[study.Trial]],
    ]:
        raise NotImplementedError()

    @property
    def complete_trial(
        self,
    ) -> Callable[
        [vizier_service.CompleteTrialRequest],
        Union[study.Trial, Awaitable[study.Trial]],
    ]:
        raise NotImplementedError()

    @property
    def delete_trial(
        self,
    ) -> Callable[
        [vizier_service.DeleteTrialRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def check_trial_early_stopping_state(
        self,
    ) -> Callable[
        [vizier_service.CheckTrialEarlyStoppingStateRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def stop_trial(
        self,
    ) -> Callable[
        [vizier_service.StopTrialRequest], Union[study.Trial, Awaitable[study.Trial]]
    ]:
        raise NotImplementedError()

    @property
    def list_optimal_trials(
        self,
    ) -> Callable[
        [vizier_service.ListOptimalTrialsRequest],
        Union[
            vizier_service.ListOptimalTrialsResponse,
            Awaitable[vizier_service.ListOptimalTrialsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest],
        Union[
            operations_pb2.ListOperationsResponse,
            Awaitable[operations_pb2.ListOperationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_operation(
        self,
    ) -> Callable[
        [operations_pb2.GetOperationRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def cancel_operation(
        self,
    ) -> Callable[[operations_pb2.CancelOperationRequest], None,]:
        raise NotImplementedError()

    @property
    def delete_operation(
        self,
    ) -> Callable[[operations_pb2.DeleteOperationRequest], None,]:
        raise NotImplementedError()

    @property
    def wait_operation(
        self,
    ) -> Callable[
        [operations_pb2.WaitOperationRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
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
    def get_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.GetIamPolicyRequest],
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
    def get_location(
        self,
    ) -> Callable[
        [locations_pb2.GetLocationRequest],
        Union[locations_pb2.Location, Awaitable[locations_pb2.Location]],
    ]:
        raise NotImplementedError()

    @property
    def list_locations(
        self,
    ) -> Callable[
        [locations_pb2.ListLocationsRequest],
        Union[
            locations_pb2.ListLocationsResponse,
            Awaitable[locations_pb2.ListLocationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def kind(self) -> str:
        raise NotImplementedError()


__all__ = ("VizierServiceTransport",)
