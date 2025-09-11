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

from google.cloud.aiplatform_v1beta1.types import tensorboard
from google.cloud.aiplatform_v1beta1.types import tensorboard_experiment
from google.cloud.aiplatform_v1beta1.types import (
    tensorboard_experiment as gca_tensorboard_experiment,
)
from google.cloud.aiplatform_v1beta1.types import tensorboard_run
from google.cloud.aiplatform_v1beta1.types import tensorboard_run as gca_tensorboard_run
from google.cloud.aiplatform_v1beta1.types import tensorboard_service
from google.cloud.aiplatform_v1beta1.types import tensorboard_time_series
from google.cloud.aiplatform_v1beta1.types import (
    tensorboard_time_series as gca_tensorboard_time_series,
)
from google.cloud.location import locations_pb2  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)


class TensorboardServiceTransport(abc.ABC):
    """Abstract transport class for TensorboardService."""

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
            self.create_tensorboard: gapic_v1.method.wrap_method(
                self.create_tensorboard,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_tensorboard: gapic_v1.method.wrap_method(
                self.get_tensorboard,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_tensorboard: gapic_v1.method.wrap_method(
                self.update_tensorboard,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_tensorboards: gapic_v1.method.wrap_method(
                self.list_tensorboards,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_tensorboard: gapic_v1.method.wrap_method(
                self.delete_tensorboard,
                default_timeout=None,
                client_info=client_info,
            ),
            self.read_tensorboard_usage: gapic_v1.method.wrap_method(
                self.read_tensorboard_usage,
                default_timeout=None,
                client_info=client_info,
            ),
            self.read_tensorboard_size: gapic_v1.method.wrap_method(
                self.read_tensorboard_size,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_tensorboard_experiment: gapic_v1.method.wrap_method(
                self.create_tensorboard_experiment,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_tensorboard_experiment: gapic_v1.method.wrap_method(
                self.get_tensorboard_experiment,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_tensorboard_experiment: gapic_v1.method.wrap_method(
                self.update_tensorboard_experiment,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_tensorboard_experiments: gapic_v1.method.wrap_method(
                self.list_tensorboard_experiments,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_tensorboard_experiment: gapic_v1.method.wrap_method(
                self.delete_tensorboard_experiment,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_tensorboard_run: gapic_v1.method.wrap_method(
                self.create_tensorboard_run,
                default_timeout=None,
                client_info=client_info,
            ),
            self.batch_create_tensorboard_runs: gapic_v1.method.wrap_method(
                self.batch_create_tensorboard_runs,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_tensorboard_run: gapic_v1.method.wrap_method(
                self.get_tensorboard_run,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_tensorboard_run: gapic_v1.method.wrap_method(
                self.update_tensorboard_run,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_tensorboard_runs: gapic_v1.method.wrap_method(
                self.list_tensorboard_runs,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_tensorboard_run: gapic_v1.method.wrap_method(
                self.delete_tensorboard_run,
                default_timeout=None,
                client_info=client_info,
            ),
            self.batch_create_tensorboard_time_series: gapic_v1.method.wrap_method(
                self.batch_create_tensorboard_time_series,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_tensorboard_time_series: gapic_v1.method.wrap_method(
                self.create_tensorboard_time_series,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_tensorboard_time_series: gapic_v1.method.wrap_method(
                self.get_tensorboard_time_series,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_tensorboard_time_series: gapic_v1.method.wrap_method(
                self.update_tensorboard_time_series,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_tensorboard_time_series: gapic_v1.method.wrap_method(
                self.list_tensorboard_time_series,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_tensorboard_time_series: gapic_v1.method.wrap_method(
                self.delete_tensorboard_time_series,
                default_timeout=None,
                client_info=client_info,
            ),
            self.batch_read_tensorboard_time_series_data: gapic_v1.method.wrap_method(
                self.batch_read_tensorboard_time_series_data,
                default_timeout=None,
                client_info=client_info,
            ),
            self.read_tensorboard_time_series_data: gapic_v1.method.wrap_method(
                self.read_tensorboard_time_series_data,
                default_timeout=None,
                client_info=client_info,
            ),
            self.read_tensorboard_blob_data: gapic_v1.method.wrap_method(
                self.read_tensorboard_blob_data,
                default_timeout=None,
                client_info=client_info,
            ),
            self.write_tensorboard_experiment_data: gapic_v1.method.wrap_method(
                self.write_tensorboard_experiment_data,
                default_timeout=None,
                client_info=client_info,
            ),
            self.write_tensorboard_run_data: gapic_v1.method.wrap_method(
                self.write_tensorboard_run_data,
                default_timeout=None,
                client_info=client_info,
            ),
            self.export_tensorboard_time_series_data: gapic_v1.method.wrap_method(
                self.export_tensorboard_time_series_data,
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
    def operations_client(self):
        """Return the client designed to process long-running operations."""
        raise NotImplementedError()

    @property
    def create_tensorboard(
        self,
    ) -> Callable[
        [tensorboard_service.CreateTensorboardRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_tensorboard(
        self,
    ) -> Callable[
        [tensorboard_service.GetTensorboardRequest],
        Union[tensorboard.Tensorboard, Awaitable[tensorboard.Tensorboard]],
    ]:
        raise NotImplementedError()

    @property
    def update_tensorboard(
        self,
    ) -> Callable[
        [tensorboard_service.UpdateTensorboardRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def list_tensorboards(
        self,
    ) -> Callable[
        [tensorboard_service.ListTensorboardsRequest],
        Union[
            tensorboard_service.ListTensorboardsResponse,
            Awaitable[tensorboard_service.ListTensorboardsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_tensorboard(
        self,
    ) -> Callable[
        [tensorboard_service.DeleteTensorboardRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def read_tensorboard_usage(
        self,
    ) -> Callable[
        [tensorboard_service.ReadTensorboardUsageRequest],
        Union[
            tensorboard_service.ReadTensorboardUsageResponse,
            Awaitable[tensorboard_service.ReadTensorboardUsageResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def read_tensorboard_size(
        self,
    ) -> Callable[
        [tensorboard_service.ReadTensorboardSizeRequest],
        Union[
            tensorboard_service.ReadTensorboardSizeResponse,
            Awaitable[tensorboard_service.ReadTensorboardSizeResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_tensorboard_experiment(
        self,
    ) -> Callable[
        [tensorboard_service.CreateTensorboardExperimentRequest],
        Union[
            gca_tensorboard_experiment.TensorboardExperiment,
            Awaitable[gca_tensorboard_experiment.TensorboardExperiment],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_tensorboard_experiment(
        self,
    ) -> Callable[
        [tensorboard_service.GetTensorboardExperimentRequest],
        Union[
            tensorboard_experiment.TensorboardExperiment,
            Awaitable[tensorboard_experiment.TensorboardExperiment],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_tensorboard_experiment(
        self,
    ) -> Callable[
        [tensorboard_service.UpdateTensorboardExperimentRequest],
        Union[
            gca_tensorboard_experiment.TensorboardExperiment,
            Awaitable[gca_tensorboard_experiment.TensorboardExperiment],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_tensorboard_experiments(
        self,
    ) -> Callable[
        [tensorboard_service.ListTensorboardExperimentsRequest],
        Union[
            tensorboard_service.ListTensorboardExperimentsResponse,
            Awaitable[tensorboard_service.ListTensorboardExperimentsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_tensorboard_experiment(
        self,
    ) -> Callable[
        [tensorboard_service.DeleteTensorboardExperimentRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def create_tensorboard_run(
        self,
    ) -> Callable[
        [tensorboard_service.CreateTensorboardRunRequest],
        Union[
            gca_tensorboard_run.TensorboardRun,
            Awaitable[gca_tensorboard_run.TensorboardRun],
        ],
    ]:
        raise NotImplementedError()

    @property
    def batch_create_tensorboard_runs(
        self,
    ) -> Callable[
        [tensorboard_service.BatchCreateTensorboardRunsRequest],
        Union[
            tensorboard_service.BatchCreateTensorboardRunsResponse,
            Awaitable[tensorboard_service.BatchCreateTensorboardRunsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_tensorboard_run(
        self,
    ) -> Callable[
        [tensorboard_service.GetTensorboardRunRequest],
        Union[
            tensorboard_run.TensorboardRun, Awaitable[tensorboard_run.TensorboardRun]
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_tensorboard_run(
        self,
    ) -> Callable[
        [tensorboard_service.UpdateTensorboardRunRequest],
        Union[
            gca_tensorboard_run.TensorboardRun,
            Awaitable[gca_tensorboard_run.TensorboardRun],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_tensorboard_runs(
        self,
    ) -> Callable[
        [tensorboard_service.ListTensorboardRunsRequest],
        Union[
            tensorboard_service.ListTensorboardRunsResponse,
            Awaitable[tensorboard_service.ListTensorboardRunsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_tensorboard_run(
        self,
    ) -> Callable[
        [tensorboard_service.DeleteTensorboardRunRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def batch_create_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.BatchCreateTensorboardTimeSeriesRequest],
        Union[
            tensorboard_service.BatchCreateTensorboardTimeSeriesResponse,
            Awaitable[tensorboard_service.BatchCreateTensorboardTimeSeriesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.CreateTensorboardTimeSeriesRequest],
        Union[
            gca_tensorboard_time_series.TensorboardTimeSeries,
            Awaitable[gca_tensorboard_time_series.TensorboardTimeSeries],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.GetTensorboardTimeSeriesRequest],
        Union[
            tensorboard_time_series.TensorboardTimeSeries,
            Awaitable[tensorboard_time_series.TensorboardTimeSeries],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.UpdateTensorboardTimeSeriesRequest],
        Union[
            gca_tensorboard_time_series.TensorboardTimeSeries,
            Awaitable[gca_tensorboard_time_series.TensorboardTimeSeries],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.ListTensorboardTimeSeriesRequest],
        Union[
            tensorboard_service.ListTensorboardTimeSeriesResponse,
            Awaitable[tensorboard_service.ListTensorboardTimeSeriesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.DeleteTensorboardTimeSeriesRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def batch_read_tensorboard_time_series_data(
        self,
    ) -> Callable[
        [tensorboard_service.BatchReadTensorboardTimeSeriesDataRequest],
        Union[
            tensorboard_service.BatchReadTensorboardTimeSeriesDataResponse,
            Awaitable[tensorboard_service.BatchReadTensorboardTimeSeriesDataResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def read_tensorboard_time_series_data(
        self,
    ) -> Callable[
        [tensorboard_service.ReadTensorboardTimeSeriesDataRequest],
        Union[
            tensorboard_service.ReadTensorboardTimeSeriesDataResponse,
            Awaitable[tensorboard_service.ReadTensorboardTimeSeriesDataResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def read_tensorboard_blob_data(
        self,
    ) -> Callable[
        [tensorboard_service.ReadTensorboardBlobDataRequest],
        Union[
            tensorboard_service.ReadTensorboardBlobDataResponse,
            Awaitable[tensorboard_service.ReadTensorboardBlobDataResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def write_tensorboard_experiment_data(
        self,
    ) -> Callable[
        [tensorboard_service.WriteTensorboardExperimentDataRequest],
        Union[
            tensorboard_service.WriteTensorboardExperimentDataResponse,
            Awaitable[tensorboard_service.WriteTensorboardExperimentDataResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def write_tensorboard_run_data(
        self,
    ) -> Callable[
        [tensorboard_service.WriteTensorboardRunDataRequest],
        Union[
            tensorboard_service.WriteTensorboardRunDataResponse,
            Awaitable[tensorboard_service.WriteTensorboardRunDataResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def export_tensorboard_time_series_data(
        self,
    ) -> Callable[
        [tensorboard_service.ExportTensorboardTimeSeriesDataRequest],
        Union[
            tensorboard_service.ExportTensorboardTimeSeriesDataResponse,
            Awaitable[tensorboard_service.ExportTensorboardTimeSeriesDataResponse],
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


__all__ = ("TensorboardServiceTransport",)
