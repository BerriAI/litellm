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

from google.cloud.aiplatform_v1beta1.types import model
from google.cloud.aiplatform_v1beta1.types import model as gca_model
from google.cloud.aiplatform_v1beta1.types import model_evaluation
from google.cloud.aiplatform_v1beta1.types import (
    model_evaluation as gca_model_evaluation,
)
from google.cloud.aiplatform_v1beta1.types import model_evaluation_slice
from google.cloud.aiplatform_v1beta1.types import model_service
from google.cloud.location import locations_pb2  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)


class ModelServiceTransport(abc.ABC):
    """Abstract transport class for ModelService."""

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
            self.upload_model: gapic_v1.method.wrap_method(
                self.upload_model,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_model: gapic_v1.method.wrap_method(
                self.get_model,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_models: gapic_v1.method.wrap_method(
                self.list_models,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_model_versions: gapic_v1.method.wrap_method(
                self.list_model_versions,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_model: gapic_v1.method.wrap_method(
                self.update_model,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.update_explanation_dataset: gapic_v1.method.wrap_method(
                self.update_explanation_dataset,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_model: gapic_v1.method.wrap_method(
                self.delete_model,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.delete_model_version: gapic_v1.method.wrap_method(
                self.delete_model_version,
                default_timeout=None,
                client_info=client_info,
            ),
            self.merge_version_aliases: gapic_v1.method.wrap_method(
                self.merge_version_aliases,
                default_timeout=None,
                client_info=client_info,
            ),
            self.export_model: gapic_v1.method.wrap_method(
                self.export_model,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.copy_model: gapic_v1.method.wrap_method(
                self.copy_model,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.import_model_evaluation: gapic_v1.method.wrap_method(
                self.import_model_evaluation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.batch_import_model_evaluation_slices: gapic_v1.method.wrap_method(
                self.batch_import_model_evaluation_slices,
                default_timeout=None,
                client_info=client_info,
            ),
            self.batch_import_evaluated_annotations: gapic_v1.method.wrap_method(
                self.batch_import_evaluated_annotations,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_model_evaluation: gapic_v1.method.wrap_method(
                self.get_model_evaluation,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_model_evaluations: gapic_v1.method.wrap_method(
                self.list_model_evaluations,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_model_evaluation_slice: gapic_v1.method.wrap_method(
                self.get_model_evaluation_slice,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_model_evaluation_slices: gapic_v1.method.wrap_method(
                self.list_model_evaluation_slices,
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
    def upload_model(
        self,
    ) -> Callable[
        [model_service.UploadModelRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_model(
        self,
    ) -> Callable[
        [model_service.GetModelRequest], Union[model.Model, Awaitable[model.Model]]
    ]:
        raise NotImplementedError()

    @property
    def list_models(
        self,
    ) -> Callable[
        [model_service.ListModelsRequest],
        Union[
            model_service.ListModelsResponse,
            Awaitable[model_service.ListModelsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_model_versions(
        self,
    ) -> Callable[
        [model_service.ListModelVersionsRequest],
        Union[
            model_service.ListModelVersionsResponse,
            Awaitable[model_service.ListModelVersionsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_model(
        self,
    ) -> Callable[
        [model_service.UpdateModelRequest],
        Union[gca_model.Model, Awaitable[gca_model.Model]],
    ]:
        raise NotImplementedError()

    @property
    def update_explanation_dataset(
        self,
    ) -> Callable[
        [model_service.UpdateExplanationDatasetRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_model(
        self,
    ) -> Callable[
        [model_service.DeleteModelRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_model_version(
        self,
    ) -> Callable[
        [model_service.DeleteModelVersionRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def merge_version_aliases(
        self,
    ) -> Callable[
        [model_service.MergeVersionAliasesRequest],
        Union[model.Model, Awaitable[model.Model]],
    ]:
        raise NotImplementedError()

    @property
    def export_model(
        self,
    ) -> Callable[
        [model_service.ExportModelRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def copy_model(
        self,
    ) -> Callable[
        [model_service.CopyModelRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def import_model_evaluation(
        self,
    ) -> Callable[
        [model_service.ImportModelEvaluationRequest],
        Union[
            gca_model_evaluation.ModelEvaluation,
            Awaitable[gca_model_evaluation.ModelEvaluation],
        ],
    ]:
        raise NotImplementedError()

    @property
    def batch_import_model_evaluation_slices(
        self,
    ) -> Callable[
        [model_service.BatchImportModelEvaluationSlicesRequest],
        Union[
            model_service.BatchImportModelEvaluationSlicesResponse,
            Awaitable[model_service.BatchImportModelEvaluationSlicesResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def batch_import_evaluated_annotations(
        self,
    ) -> Callable[
        [model_service.BatchImportEvaluatedAnnotationsRequest],
        Union[
            model_service.BatchImportEvaluatedAnnotationsResponse,
            Awaitable[model_service.BatchImportEvaluatedAnnotationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_model_evaluation(
        self,
    ) -> Callable[
        [model_service.GetModelEvaluationRequest],
        Union[
            model_evaluation.ModelEvaluation,
            Awaitable[model_evaluation.ModelEvaluation],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_model_evaluations(
        self,
    ) -> Callable[
        [model_service.ListModelEvaluationsRequest],
        Union[
            model_service.ListModelEvaluationsResponse,
            Awaitable[model_service.ListModelEvaluationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_model_evaluation_slice(
        self,
    ) -> Callable[
        [model_service.GetModelEvaluationSliceRequest],
        Union[
            model_evaluation_slice.ModelEvaluationSlice,
            Awaitable[model_evaluation_slice.ModelEvaluationSlice],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_model_evaluation_slices(
        self,
    ) -> Callable[
        [model_service.ListModelEvaluationSlicesRequest],
        Union[
            model_service.ListModelEvaluationSlicesResponse,
            Awaitable[model_service.ListModelEvaluationSlicesResponse],
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


__all__ = ("ModelServiceTransport",)
