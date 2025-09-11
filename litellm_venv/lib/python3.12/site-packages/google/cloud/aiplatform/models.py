# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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
import json
import pathlib
import re
import shutil
import tempfile
import requests
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from google.api_core import operation
from google.api_core import exceptions as api_exceptions
from google.auth import credentials as auth_credentials
from google.auth.transport import requests as google_auth_requests
from google.protobuf import duration_pb2
import proto

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import constants
from google.cloud.aiplatform import explain
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import jobs
from google.cloud.aiplatform import models
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.utils import gcs_utils
from google.cloud.aiplatform.utils import _explanation_utils
from google.cloud.aiplatform.utils import _ipython_utils
from google.cloud.aiplatform import model_evaluation
from google.cloud.aiplatform.compat.services import endpoint_service_client

from google.cloud.aiplatform.compat.types import (
    encryption_spec as gca_encryption_spec,
    endpoint as gca_endpoint_compat,
    explanation as gca_explanation_compat,
    io as gca_io_compat,
    machine_resources as gca_machine_resources_compat,
    model as gca_model_compat,
    model_service as gca_model_service_compat,
    env_var as gca_env_var_compat,
)

from google.cloud.aiplatform.constants import (
    prediction as prediction_constants,
)

from google.cloud.aiplatform_v1.types import model as model_v1

from google.protobuf import field_mask_pb2, timestamp_pb2
from google.protobuf import json_format

if TYPE_CHECKING:
    from google.cloud.aiplatform.prediction import LocalModel

_DEFAULT_MACHINE_TYPE = "n1-standard-2"
_DEPLOYING_MODEL_TRAFFIC_SPLIT_KEY = "0"
_SUCCESSFUL_HTTP_RESPONSE = 300
_RAW_PREDICT_DEPLOYED_MODEL_ID_KEY = "X-Vertex-AI-Deployed-Model-Id"
_RAW_PREDICT_MODEL_RESOURCE_KEY = "X-Vertex-AI-Model"
_RAW_PREDICT_MODEL_VERSION_ID_KEY = "X-Vertex-AI-Model-Version-Id"

_LOGGER = base.Logger(__name__)


_SUPPORTED_MODEL_FILE_NAMES = [
    "model.pkl",
    "model.joblib",
    "model.bst",
    "model.mar",
    "saved_model.pb",
    "saved_model.pbtxt",
]

_SUPPORTED_EVAL_PREDICTION_TYPES = [
    "classification",
    "regression",
]


class VersionInfo(NamedTuple):
    """VersionInfo class envelopes returned Model version information.

    Attributes:
        version_id:
            The version ID of the model.
        create_time:
            Timestamp when this Model version was uploaded into Vertex AI.
        update_time:
            Timestamp when this Model version was most recently updated.
        model_display_name:
            The user-defined name of the model this version belongs to.
        model_resource_name:
            The fully-qualified model resource name.
            e.g. projects/{project}/locations/{location}/models/{model_display_name}
        version_aliases:
            User provided version aliases so that a model version can be referenced via
            alias (i.e. projects/{project}/locations/{location}/models/{model_display_name}@{version_alias}).
            Default is None.
        version_description:
            The description of this version.
            Default is None.
    """

    version_id: str
    version_create_time: timestamp_pb2.Timestamp
    version_update_time: timestamp_pb2.Timestamp
    model_display_name: str
    model_resource_name: str
    version_aliases: Optional[Sequence[str]] = None
    version_description: Optional[str] = None


class Prediction(NamedTuple):
    """Prediction class envelopes returned Model predictions and the Model id.

    Attributes:
        predictions:
            The predictions that are the output of the predictions
            call. The schema of any single prediction may be specified via
            Endpoint's DeployedModels' [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
            [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
        deployed_model_id:
            ID of the Endpoint's DeployedModel that served this prediction.
        metadata:
            The metadata that is the output of the predictions call.
        model_version_id:
            ID of the DeployedModel's version that served this prediction.
        model_resource_name:
            The fully-qualified resource name of the model that served this prediction.
        explanations:
            The explanations of the Model's predictions. It has the same number
            of elements as instances to be explained. Default is None.
    """

    predictions: List[Any]
    deployed_model_id: str
    metadata: Optional[Any] = None
    model_version_id: Optional[str] = None
    model_resource_name: Optional[str] = None
    explanations: Optional[Sequence[gca_explanation_compat.Explanation]] = None


class Endpoint(base.VertexAiResourceNounWithFutureManager, base.PreviewMixin):
    client_class = utils.EndpointClientWithOverride
    _resource_noun = "endpoints"
    _getter_method = "get_endpoint"
    _list_method = "list_endpoints"
    _delete_method = "delete_endpoint"
    _parse_resource_name_method = "parse_endpoint_path"
    _format_resource_name_method = "endpoint_path"
    _preview_class = "google.cloud.aiplatform.aiplatform.preview.models.Endpoint"

    @property
    def preview(self):
        """Return an Endpoint instance with preview features enabled."""
        from google.cloud.aiplatform.preview import models as preview_models

        if not hasattr(self, "_preview_instance"):
            self._preview_instance = preview_models.Endpoint(
                self.resource_name, credentials=self.credentials
            )

        return self._preview_instance

    def __init__(
        self,
        endpoint_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an endpoint resource.

        Args:
            endpoint_name (str):
                Required. A fully-qualified endpoint resource name or endpoint ID.
                Example: "projects/123/locations/us-central1/endpoints/456" or
                "456" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve endpoint from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve endpoint from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=endpoint_name,
        )

        endpoint_name = utils.full_resource_name(
            resource_name=endpoint_name,
            resource_noun="endpoints",
            parse_resource_name_method=self._parse_resource_name,
            format_resource_name_method=self._format_resource_name,
            project=project,
            location=location,
        )

        # Lazy load the Endpoint gca_resource until needed
        self._gca_resource = gca_endpoint_compat.Endpoint(name=endpoint_name)

        self.authorized_session = None
        self.raw_predict_request_url = None

    @property
    def _prediction_client(self) -> utils.PredictionClientWithOverride:
        # The attribute might not exist due to issues in
        # `VertexAiResourceNounWithFutureManager._sync_object_with_future_result`
        # We should switch to @functools.cached_property once its available.
        if not getattr(self, "_prediction_client_value", None):
            self._prediction_client_value = initializer.global_config.create_client(
                client_class=utils.PredictionClientWithOverride,
                credentials=self.credentials,
                location_override=self.location,
                prediction_client=True,
            )
        return self._prediction_client_value

    @property
    def _prediction_async_client(self) -> utils.PredictionAsyncClientWithOverride:
        # The attribute might not exist due to issues in
        # `VertexAiResourceNounWithFutureManager._sync_object_with_future_result`
        # We should switch to @functools.cached_property once its available.
        if not getattr(self, "_prediction_async_client_value", None):
            self._prediction_async_client_value = (
                initializer.global_config.create_client(
                    client_class=utils.PredictionAsyncClientWithOverride,
                    credentials=self.credentials,
                    location_override=self.location,
                    prediction_client=True,
                )
            )
        return self._prediction_async_client_value

    def _skipped_getter_call(self) -> bool:
        """Check if GAPIC resource was populated by call to get/list API methods

        Returns False if `_gca_resource` is None or fully populated. Returns True
        if `_gca_resource` is partially populated
        """
        return self._gca_resource and not self._gca_resource.create_time

    def _sync_gca_resource_if_skipped(self) -> None:
        """Sync GAPIC service representation of Endpoint class resource only if
        get_endpoint() was never called."""
        if self._skipped_getter_call():
            self._gca_resource = self._get_gca_resource(
                resource_name=self._gca_resource.name
            )

    def _assert_gca_resource_is_available(self) -> None:
        """Ensures Endpoint getter was called at least once before
        asserting on gca_resource's availability."""
        super()._assert_gca_resource_is_available()
        self._sync_gca_resource_if_skipped()

    @property
    def traffic_split(self) -> Dict[str, int]:
        """A map from a DeployedModel's ID to the percentage of this Endpoint's
        traffic that should be forwarded to that DeployedModel.

        If a DeployedModel's ID is not listed in this map, then it receives no traffic.

        The traffic percentage values must add up to 100, or map must be empty if
        the Endpoint is to not accept any traffic at a moment.
        """
        self._sync_gca_resource()
        return dict(self._gca_resource.traffic_split)

    @property
    def network(self) -> Optional[str]:
        """The full name of the Google Compute Engine
        [network](https://cloud.google.com/vpc/docs/vpc#networks) to which this
        Endpoint should be peered.

        Takes the format `projects/{project}/global/networks/{network}`. Where
        {project} is a project number, as in `12345`, and {network} is a network name.

        Private services access must already be configured for the network. If left
        unspecified, the Endpoint is not peered with any network.
        """
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "network", None)

    @classmethod
    def create(
        cls,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
        sync=True,
        create_request_timeout: Optional[float] = None,
        endpoint_id: Optional[str] = None,
        enable_request_response_logging=False,
        request_response_logging_sampling_rate: Optional[float] = None,
        request_response_logging_bq_destination_table: Optional[str] = None,
    ) -> "Endpoint":
        """Creates a new endpoint.

        Args:
            display_name (str):
                Optional. The user-defined name of the Endpoint.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            description (str):
                Optional. The description of the Endpoint.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Endpoints.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            project (str):
                Optional. Project to retrieve endpoint from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve endpoint from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Endpoint and all sub-resources of this Endpoint will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            endpoint_id (str):
                Optional. The ID to use for endpoint, which will become
                the final component of the endpoint resource name. If
                not provided, Vertex AI will generate a value for this
                ID.

                This value should be 1-10 characters, and valid
                characters are /[0-9]/. When using HTTP/JSON, this field
                is populated based on a query string argument, such as
                ``?endpoint_id=12345``. This is the fallback for fields
                that are not included in either the URI or the body.
            enable_request_response_logging (bool):
                Optional. Whether to enable request & response logging for this endpoint.
            request_response_logging_sampling_rate (float):
                Optional. The request response logging sampling rate. If not set, default is 0.0.
            request_response_logging_bq_destination_table (str):
                Optional. The request response logging bigquery destination. If not set, will create a table with name:
                ``bq://{project_id}.logging_{endpoint_display_name}_{endpoint_id}.request_response_logging``.

        Returns:
            endpoint (aiplatform.Endpoint):
                Created endpoint.
        """
        api_client = cls._instantiate_client(location=location, credentials=credentials)

        if not display_name:
            display_name = cls._generate_display_name()

        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)

        project = project or initializer.global_config.project
        location = location or initializer.global_config.location

        predict_request_response_logging_config = None
        if enable_request_response_logging:
            predict_request_response_logging_config = (
                gca_endpoint_compat.PredictRequestResponseLoggingConfig(
                    enabled=True,
                    sampling_rate=request_response_logging_sampling_rate,
                    bigquery_destination=gca_io_compat.BigQueryDestination(
                        output_uri=request_response_logging_bq_destination_table
                    ),
                )
            )
        return cls._create(
            api_client=api_client,
            display_name=display_name,
            project=project,
            location=location,
            description=description,
            labels=labels,
            metadata=metadata,
            credentials=credentials,
            encryption_spec=initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            ),
            sync=sync,
            create_request_timeout=create_request_timeout,
            endpoint_id=endpoint_id,
            predict_request_response_logging_config=predict_request_response_logging_config,
        )

    @classmethod
    @base.optional_sync()
    def _create(
        cls,
        api_client: endpoint_service_client.EndpointServiceClient,
        display_name: str,
        project: str,
        location: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec: Optional[gca_encryption_spec.EncryptionSpec] = None,
        network: Optional[str] = None,
        sync=True,
        create_request_timeout: Optional[float] = None,
        endpoint_id: Optional[str] = None,
        predict_request_response_logging_config: Optional[
            gca_endpoint_compat.PredictRequestResponseLoggingConfig
        ] = None,
    ) -> "Endpoint":
        """Creates a new endpoint by calling the API client.

        Args:
            api_client (EndpointServiceClient):
                Required. An instance of EndpointServiceClient with the correct
                api_endpoint already set based on user's preferences.
            display_name (str):
                Required. The user-defined name of the Endpoint.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            project (str):
                Required. Project to retrieve endpoint from.
            location (str):
                Required. Location to retrieve endpoint from.
            description (str):
                Optional. The description of the Endpoint.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Endpoints.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
            encryption_spec (gca_encryption_spec.EncryptionSpec):
                Optional. The Cloud KMS customer managed encryption key used to protect the dataset.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Dataset and all sub-resources of this Dataset will be secured by this key.
            network (str):
                Optional. The full name of the Compute Engine network to which
                this Endpoint will be peered. E.g. "projects/12345/global/networks/myVPC".
                Private services access must already be configured for the network.
                Read more about PrivateEndpoints
                [in the documentation](https://cloud.google.com/vertex-ai/docs/predictions/using-private-endpoints).
            sync (bool):
                Whether to create this endpoint synchronously.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            endpoint_id (str):
                Optional. The ID to use for endpoint, which will become
                the final component of the endpoint resource name. If
                not provided, Vertex AI will generate a value for this
                ID.

                This value should be 1-10 characters, and valid
                characters are /[0-9]/. When using HTTP/JSON, this field
                is populated based on a query string argument, such as
                ``?endpoint_id=12345``. This is the fallback for fields
                that are not included in either the URI or the body.
            predict_request_response_logging_config (aiplatform.endpoint.PredictRequestResponseLoggingConfig):
                Optional. The request response logging configuration for online prediction.

        Returns:
            endpoint (aiplatform.Endpoint):
                Created endpoint.
        """

        parent = initializer.global_config.common_location_path(
            project=project, location=location
        )

        gapic_endpoint = gca_endpoint_compat.Endpoint(
            display_name=display_name,
            description=description,
            labels=labels,
            encryption_spec=encryption_spec,
            network=network,
            predict_request_response_logging_config=predict_request_response_logging_config,
        )

        operation_future = api_client.create_endpoint(
            parent=parent,
            endpoint=gapic_endpoint,
            endpoint_id=endpoint_id,
            metadata=metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, operation_future)

        created_endpoint = operation_future.result()

        _LOGGER.log_create_complete(cls, created_endpoint, "endpoint")

        return cls._construct_sdk_resource_from_gapic(
            gapic_resource=created_endpoint,
            project=project,
            location=location,
            credentials=credentials,
        )

    @classmethod
    def _construct_sdk_resource_from_gapic(
        cls,
        gapic_resource: proto.Message,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "Endpoint":
        """Given a GAPIC Endpoint object, return the SDK representation.

        Args:
            gapic_resource (proto.Message):
                A GAPIC representation of a Endpoint resource, usually
                retrieved by a get_* or in a list_* API call.
            project (str):
                Optional. Project to construct Endpoint object from. If not set,
                project set in aiplatform.init will be used.
            location (str):
                Optional. Location to construct Endpoint object from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to construct Endpoint.
                Overrides credentials set in aiplatform.init.

        Returns:
            Endpoint (aiplatform.Endpoint):
                An initialized Endpoint resource.
        """
        endpoint = super()._construct_sdk_resource_from_gapic(
            gapic_resource=gapic_resource,
            project=project,
            location=location,
            credentials=credentials,
        )
        endpoint.authorized_session = None
        endpoint.raw_predict_request_url = None

        return endpoint

    @staticmethod
    def _allocate_traffic(
        traffic_split: Dict[str, int],
        traffic_percentage: int,
    ) -> Dict[str, int]:
        """Allocates desired traffic to new deployed model and scales traffic
        of older deployed models.

        Args:
            traffic_split (Dict[str, int]):
                Required. Current traffic split of deployed models in endpoint.
            traffic_percentage (int):
                Required. Desired traffic to new deployed model.

        Returns:
            new_traffic_split (Dict[str, int]):
                Traffic split to use.
        """
        new_traffic_split = {}
        old_models_traffic = 100 - traffic_percentage
        if old_models_traffic:
            unallocated_traffic = old_models_traffic
            for deployed_model in traffic_split:
                current_traffic = traffic_split[deployed_model]
                new_traffic = int(current_traffic / 100 * old_models_traffic)
                new_traffic_split[deployed_model] = new_traffic
                unallocated_traffic -= new_traffic
            # will likely under-allocate. make total 100.
            for deployed_model in new_traffic_split:
                if unallocated_traffic == 0:
                    break
                new_traffic_split[deployed_model] += 1
                unallocated_traffic -= 1

        new_traffic_split[_DEPLOYING_MODEL_TRAFFIC_SPLIT_KEY] = traffic_percentage

        return new_traffic_split

    @staticmethod
    def _unallocate_traffic(
        traffic_split: Dict[str, int],
        deployed_model_id: str,
    ) -> Dict[str, int]:
        """Sets deployed model id's traffic to 0 and scales the traffic of
        other deployed models.

        Args:
            traffic_split (Dict[str, int]):
                Required. Current traffic split of deployed models in endpoint.
            deployed_model_id (str):
                Required. Desired traffic to new deployed model.

        Returns:
            new_traffic_split (Dict[str, int]):
                Traffic split to use.
        """
        new_traffic_split = traffic_split.copy()
        del new_traffic_split[deployed_model_id]
        deployed_model_id_traffic = traffic_split[deployed_model_id]
        traffic_percent_left = 100 - deployed_model_id_traffic

        if traffic_percent_left:
            unallocated_traffic = 100
            for deployed_model in new_traffic_split:
                current_traffic = traffic_split[deployed_model]
                new_traffic = int(current_traffic / traffic_percent_left * 100)
                new_traffic_split[deployed_model] = new_traffic
                unallocated_traffic -= new_traffic
            # will likely under-allocate. make total 100.
            for deployed_model in new_traffic_split:
                if unallocated_traffic == 0:
                    break
                new_traffic_split[deployed_model] += 1
                unallocated_traffic -= 1

        new_traffic_split[deployed_model_id] = 0

        return new_traffic_split

    @staticmethod
    def _validate_deploy_args(
        min_replica_count: int,
        max_replica_count: int,
        accelerator_type: Optional[str],
        deployed_model_display_name: Optional[str],
        traffic_split: Optional[Dict[str, int]],
        traffic_percentage: Optional[int],
    ):
        """Helper method to validate deploy arguments.

        Args:
            min_replica_count (int):
                Required. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.
            max_replica_count (int):
                Required. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the larger value of min_replica_count or 1 will
                be used. If value provided is smaller than min_replica_count, it
                will automatically be increased to be min_replica_count.
            accelerator_type (str):
                Required. Hardware accelerator type. One of ACCELERATOR_TYPE_UNSPECIFIED,
                NVIDIA_TESLA_K80, NVIDIA_TESLA_P100, NVIDIA_TESLA_V100, NVIDIA_TESLA_P4,
                NVIDIA_TESLA_T4
            deployed_model_display_name (str):
                Required. The display name of the DeployedModel. If not provided
                upon creation, the Model's display_name is used.
            traffic_split (Dict[str, int]):
                Optional. A map from a DeployedModel's ID to the percentage of
                this Endpoint's traffic that should be forwarded to that DeployedModel.
                If a DeployedModel's ID is not listed in this map, then it receives
                no traffic. The traffic percentage values must add up to 100, or
                map must be empty if the Endpoint is to not accept any traffic at
                the moment. Key for model being deployed is "0". Should not be
                provided if traffic_percentage is provided.
            traffic_percentage (int):
                Optional. Desired traffic to newly deployed model. Defaults to
                0 if there are pre-existing deployed models. Defaults to 100 if
                there are no pre-existing deployed models. Negative values should
                not be provided. Traffic of previously deployed models at the endpoint
                will be scaled down to accommodate new deployed model's traffic.
                Should not be provided if traffic_split is provided.

        Raises:
            ValueError: if Min or Max replica is negative. Traffic percentage > 100 or
                < 0. Or if traffic_split does not sum to 100.
        """
        if min_replica_count < 0:
            raise ValueError("Min replica cannot be negative.")
        if max_replica_count < 0:
            raise ValueError("Max replica cannot be negative.")
        if deployed_model_display_name is not None:
            utils.validate_display_name(deployed_model_display_name)

        if traffic_split is None:
            if traffic_percentage > 100:
                raise ValueError("Traffic percentage cannot be greater than 100.")
            if traffic_percentage < 0:
                raise ValueError("Traffic percentage cannot be negative.")

        elif traffic_split:
            if sum(traffic_split.values()) != 100:
                raise ValueError(
                    "Sum of all traffic within traffic split needs to be 100."
                )

        # Raises ValueError if invalid accelerator
        if accelerator_type:
            utils.validate_accelerator_type(accelerator_type)

    def deploy(
        self,
        model: "Model",
        deployed_model_display_name: Optional[str] = None,
        traffic_percentage: int = 0,
        traffic_split: Optional[Dict[str, int]] = None,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        service_account: Optional[str] = None,
        explanation_metadata: Optional[aiplatform.explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[
            aiplatform.explain.ExplanationParameters
        ] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync=True,
        deploy_request_timeout: Optional[float] = None,
        autoscaling_target_cpu_utilization: Optional[int] = None,
        autoscaling_target_accelerator_duty_cycle: Optional[int] = None,
        enable_access_logging=False,
        disable_container_logging: bool = False,
    ) -> None:
        """Deploys a Model to the Endpoint.

        Args:
            model (aiplatform.Model):
                Required. Model to be deployed.
            deployed_model_display_name (str):
                Optional. The display name of the DeployedModel. If not provided
                upon creation, the Model's display_name is used.
            traffic_percentage (int):
                Optional. Desired traffic to newly deployed model. Defaults to
                0 if there are pre-existing deployed models. Defaults to 100 if
                there are no pre-existing deployed models. Negative values should
                not be provided. Traffic of previously deployed models at the endpoint
                will be scaled down to accommodate new deployed model's traffic.
                Should not be provided if traffic_split is provided.
            traffic_split (Dict[str, int]):
                Optional. A map from a DeployedModel's ID to the percentage of
                this Endpoint's traffic that should be forwarded to that DeployedModel.
                If a DeployedModel's ID is not listed in this map, then it receives
                no traffic. The traffic percentage values must add up to 100, or
                map must be empty if the Endpoint is to not accept any traffic at
                the moment. Key for model being deployed is "0". Should not be
                provided if traffic_percentage is provided.
            machine_type (str):
                Optional. The type of machine. Not specifying machine type will
                result in model to be deployed with automatic resources.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the larger value of min_replica_count or 1 will
                be used. If value provided is smaller than min_replica_count, it
                will automatically be increased to be min_replica_count.
            accelerator_type (str):
                Optional. Hardware accelerator type. Must also set accelerator_count if used.
                One of ACCELERATOR_TYPE_UNSPECIFIED, NVIDIA_TESLA_K80, NVIDIA_TESLA_P100,
                NVIDIA_TESLA_V100, NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int):
                Optional. The number of accelerators to attach to a worker replica.
            service_account (str):
                The service account that the DeployedModel's container runs as. Specify the
                email address of the service account. If this service account is not
                specified, the container runs as a service account that doesn't have access
                to the resource project.
                Users deploying the Model must have the `iam.serviceAccounts.actAs`
                permission on this service account.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Metadata describing the Model's input and output for explanation.
                `explanation_metadata` is optional while `explanation_parameters` must be
                specified when used.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            deploy_request_timeout (float):
                Optional. The timeout for the deploy request in seconds.
            autoscaling_target_cpu_utilization (int):
                Target CPU Utilization to use for Autoscaling Replicas.
                A default value of 60 will be used if not specified.
            autoscaling_target_accelerator_duty_cycle (int):
                Target Accelerator Duty Cycle.
                Must also set accelerator_type and accelerator_count if specified.
                A default value of 60 will be used if not specified.
            enable_access_logging (bool):
                Whether to enable endpoint access logging. Defaults to False.
            disable_container_logging (bool):
                If True, container logs from the deployed model will not be
                written to Cloud Logging. Defaults to False.
        """
        self._sync_gca_resource_if_skipped()

        self._validate_deploy_args(
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            deployed_model_display_name=deployed_model_display_name,
            traffic_split=traffic_split,
            traffic_percentage=traffic_percentage,
        )

        explanation_spec = _explanation_utils.create_and_validate_explanation_spec(
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
        )

        self._deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            traffic_percentage=traffic_percentage,
            traffic_split=traffic_split,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            service_account=service_account,
            explanation_spec=explanation_spec,
            metadata=metadata,
            sync=sync,
            deploy_request_timeout=deploy_request_timeout,
            autoscaling_target_cpu_utilization=autoscaling_target_cpu_utilization,
            autoscaling_target_accelerator_duty_cycle=autoscaling_target_accelerator_duty_cycle,
            enable_access_logging=enable_access_logging,
            disable_container_logging=disable_container_logging,
        )

    @base.optional_sync()
    def _deploy(
        self,
        model: "Model",
        deployed_model_display_name: Optional[str] = None,
        traffic_percentage: Optional[int] = 0,
        traffic_split: Optional[Dict[str, int]] = None,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        service_account: Optional[str] = None,
        explanation_spec: Optional[aiplatform.explain.ExplanationSpec] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync=True,
        deploy_request_timeout: Optional[float] = None,
        autoscaling_target_cpu_utilization: Optional[int] = None,
        autoscaling_target_accelerator_duty_cycle: Optional[int] = None,
        enable_access_logging=False,
        disable_container_logging: bool = False,
    ) -> None:
        """Deploys a Model to the Endpoint.

        Args:
            model (aiplatform.Model):
                Required. Model to be deployed.
            deployed_model_display_name (str):
                Optional. The display name of the DeployedModel. If not provided
                upon creation, the Model's display_name is used.
            traffic_percentage (int):
                Optional. Desired traffic to newly deployed model. Defaults to
                0 if there are pre-existing deployed models. Defaults to 100 if
                there are no pre-existing deployed models. Negative values should
                not be provided. Traffic of previously deployed models at the endpoint
                will be scaled down to accommodate new deployed model's traffic.
                Should not be provided if traffic_split is provided.
            traffic_split (Dict[str, int]):
                Optional. A map from a DeployedModel's ID to the percentage of
                this Endpoint's traffic that should be forwarded to that DeployedModel.
                If a DeployedModel's ID is not listed in this map, then it receives
                no traffic. The traffic percentage values must add up to 100, or
                map must be empty if the Endpoint is to not accept any traffic at
                the moment. Key for model being deployed is "0". Should not be
                provided if traffic_percentage is provided.
            machine_type (str):
                Optional. The type of machine. Not specifying machine type will
                result in model to be deployed with automatic resources.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the larger value of min_replica_count or 1 will
                be used. If value provided is smaller than min_replica_count, it
                will automatically be increased to be min_replica_count.
            accelerator_type (str):
                Optional. Hardware accelerator type. Must also set accelerator_count if used.
                One of ACCELERATOR_TYPE_UNSPECIFIED, NVIDIA_TESLA_K80, NVIDIA_TESLA_P100,
                NVIDIA_TESLA_V100, NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int):
                Optional. The number of accelerators to attach to a worker replica.
            service_account (str):
                The service account that the DeployedModel's container runs as. Specify the
                email address of the service account. If this service account is not
                specified, the container runs as a service account that doesn't have access
                to the resource project.
                Users deploying the Model must have the `iam.serviceAccounts.actAs`
                permission on this service account.
            explanation_spec (aiplatform.explain.ExplanationSpec):
                Optional. Specification of Model explanation.
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            deploy_request_timeout (float):
                Optional. The timeout for the deploy request in seconds.
            autoscaling_target_cpu_utilization (int):
                Target CPU Utilization to use for Autoscaling Replicas.
                A default value of 60 will be used if not specified.
            autoscaling_target_accelerator_duty_cycle (int):
                Target Accelerator Duty Cycle.
                Must also set accelerator_type and accelerator_count if specified.
                A default value of 60 will be used if not specified.
            enable_access_logging (bool):
                Whether to enable endpoint access logging. Defaults to False.
            disable_container_logging (bool):
                If True, container logs from the deployed model will not be
                written to Cloud Logging. Defaults to False.
        """
        _LOGGER.log_action_start_against_resource(
            f"Deploying Model {model.resource_name} to", "", self
        )

        self._deploy_call(
            api_client=self.api_client,
            endpoint_resource_name=self.resource_name,
            model=model,
            endpoint_resource_traffic_split=self._gca_resource.traffic_split,
            network=self.network,
            deployed_model_display_name=deployed_model_display_name,
            traffic_percentage=traffic_percentage,
            traffic_split=traffic_split,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            service_account=service_account,
            explanation_spec=explanation_spec,
            metadata=metadata,
            deploy_request_timeout=deploy_request_timeout,
            autoscaling_target_cpu_utilization=autoscaling_target_cpu_utilization,
            autoscaling_target_accelerator_duty_cycle=autoscaling_target_accelerator_duty_cycle,
            enable_access_logging=enable_access_logging,
            disable_container_logging=disable_container_logging,
        )

        _LOGGER.log_action_completed_against_resource("model", "deployed", self)

        self._sync_gca_resource()

    @classmethod
    def _deploy_call(
        cls,
        api_client: endpoint_service_client.EndpointServiceClient,
        endpoint_resource_name: str,
        model: "Model",
        endpoint_resource_traffic_split: Optional[proto.MapField] = None,
        network: Optional[str] = None,
        deployed_model_display_name: Optional[str] = None,
        traffic_percentage: Optional[int] = 0,
        traffic_split: Optional[Dict[str, int]] = None,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        service_account: Optional[str] = None,
        explanation_spec: Optional[aiplatform.explain.ExplanationSpec] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        deploy_request_timeout: Optional[float] = None,
        autoscaling_target_cpu_utilization: Optional[int] = None,
        autoscaling_target_accelerator_duty_cycle: Optional[int] = None,
        enable_access_logging=False,
        disable_container_logging: bool = False,
    ) -> None:
        """Helper method to deploy model to endpoint.

        Args:
            api_client (endpoint_service_client.EndpointServiceClient):
                Required. endpoint_service_client.EndpointServiceClient to make call.
            endpoint_resource_name (str):
                Required. Endpoint resource name to deploy model to.
            model (aiplatform.Model):
                Required. Model to be deployed.
            endpoint_resource_traffic_split (proto.MapField):
                Optional. Endpoint current resource traffic split.
            network (str):
                Optional. The full name of the Compute Engine network to which
                this Endpoint will be peered. E.g. "projects/123/global/networks/my_vpc".
                Private services access must already be configured for the network.
            deployed_model_display_name (str):
                Optional. The display name of the DeployedModel. If not provided
                upon creation, the Model's display_name is used.
            traffic_percentage (int):
                Optional. Desired traffic to newly deployed model. Defaults to
                0 if there are pre-existing deployed models. Defaults to 100 if
                there are no pre-existing deployed models. Negative values should
                not be provided. Traffic of previously deployed models at the endpoint
                will be scaled down to accommodate new deployed model's traffic.
                Should not be provided if traffic_split is provided.
            traffic_split (Dict[str, int]):
                Optional. A map from a DeployedModel's ID to the percentage of
                this Endpoint's traffic that should be forwarded to that DeployedModel.
                If a DeployedModel's ID is not listed in this map, then it receives
                no traffic. The traffic percentage values must add up to 100, or
                map must be empty if the Endpoint is to not accept any traffic at
                the moment. Key for model being deployed is "0". Should not be
                provided if traffic_percentage is provided.
            machine_type (str):
                Optional. The type of machine. Not specifying machine type will
                result in model to be deployed with automatic resources.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the larger value of min_replica_count or 1 will
                be used. If value provided is smaller than min_replica_count, it
                will automatically be increased to be min_replica_count.
            accelerator_type (str):
                Optional. Hardware accelerator type. Must also set accelerator_count if used.
                One of ACCELERATOR_TYPE_UNSPECIFIED, NVIDIA_TESLA_K80, NVIDIA_TESLA_P100,
                NVIDIA_TESLA_V100, NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int):
                Optional. The number of accelerators to attach to a worker replica.
            service_account (str):
                The service account that the DeployedModel's container runs as. Specify the
                email address of the service account. If this service account is not
                specified, the container runs as a service account that doesn't have access
                to the resource project.
                Users deploying the Model must have the `iam.serviceAccounts.actAs`
                permission on this service account.
                If not specified, uses the service account set in aiplatform.init.
            explanation_spec (aiplatform.explain.ExplanationSpec):
                Optional. Specification of Model explanation.
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            deploy_request_timeout (float):
                Optional. The timeout for the deploy request in seconds.
            autoscaling_target_cpu_utilization (int):
                Optional. Target CPU Utilization to use for Autoscaling Replicas.
                A default value of 60 will be used if not specified.
            autoscaling_target_accelerator_duty_cycle (int):
                Optional. Target Accelerator Duty Cycle.
                Must also set accelerator_type and accelerator_count if specified.
                A default value of 60 will be used if not specified.
            enable_access_logging (bool):
                Whether to enable endpoint access logging. Defaults to False.
            disable_container_logging (bool):
                If True, container logs from the deployed model will not be
                written to Cloud Logging. Defaults to False.

        Raises:
            ValueError: If only `accelerator_type` or `accelerator_count` is specified.
            ValueError: If model does not support deployment.
            ValueError: If there is not current traffic split and traffic percentage
                is not 0 or 100.
        """

        service_account = service_account or initializer.global_config.service_account

        max_replica_count = max(min_replica_count, max_replica_count)

        if bool(accelerator_type) != bool(accelerator_count):
            raise ValueError(
                "Both `accelerator_type` and `accelerator_count` should be specified or None."
            )

        if autoscaling_target_accelerator_duty_cycle is not None and (
            not accelerator_type or not accelerator_count
        ):
            raise ValueError(
                "Both `accelerator_type` and `accelerator_count` should be set "
                "when specifying autoscaling_target_accelerator_duty_cycle`"
            )

        deployed_model = gca_endpoint_compat.DeployedModel(
            model=model.versioned_resource_name,
            display_name=deployed_model_display_name,
            service_account=service_account,
            enable_access_logging=enable_access_logging,
            disable_container_logging=disable_container_logging,
        )

        supports_automatic_resources = (
            gca_model_compat.Model.DeploymentResourcesType.AUTOMATIC_RESOURCES
            in model.supported_deployment_resources_types
        )
        supports_dedicated_resources = (
            gca_model_compat.Model.DeploymentResourcesType.DEDICATED_RESOURCES
            in model.supported_deployment_resources_types
        )
        provided_custom_machine_spec = (
            machine_type
            or accelerator_type
            or accelerator_count
            or autoscaling_target_accelerator_duty_cycle
            or autoscaling_target_cpu_utilization
        )

        # If the model supports both automatic and dedicated deployment resources,
        # decide based on the presence of machine spec customizations
        use_dedicated_resources = supports_dedicated_resources and (
            not supports_automatic_resources or provided_custom_machine_spec
        )

        if provided_custom_machine_spec and not use_dedicated_resources:
            _LOGGER.info(
                "Model does not support dedicated deployment resources. "
                "The machine_type, accelerator_type and accelerator_count,"
                "autoscaling_target_accelerator_duty_cycle,"
                "autoscaling_target_cpu_utilization parameters are ignored."
            )

        if use_dedicated_resources and not machine_type:
            machine_type = _DEFAULT_MACHINE_TYPE
            _LOGGER.info(f"Using default machine_type: {machine_type}")

        if use_dedicated_resources:
            dedicated_resources = gca_machine_resources_compat.DedicatedResources(
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )

            machine_spec = gca_machine_resources_compat.MachineSpec(
                machine_type=machine_type
            )

            if autoscaling_target_cpu_utilization:
                autoscaling_metric_spec = gca_machine_resources_compat.AutoscalingMetricSpec(
                    metric_name="aiplatform.googleapis.com/prediction/online/cpu/utilization",
                    target=autoscaling_target_cpu_utilization,
                )
                dedicated_resources.autoscaling_metric_specs.extend(
                    [autoscaling_metric_spec]
                )

            if accelerator_type and accelerator_count:
                utils.validate_accelerator_type(accelerator_type)
                machine_spec.accelerator_type = accelerator_type
                machine_spec.accelerator_count = accelerator_count

                if autoscaling_target_accelerator_duty_cycle:
                    autoscaling_metric_spec = gca_machine_resources_compat.AutoscalingMetricSpec(
                        metric_name="aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle",
                        target=autoscaling_target_accelerator_duty_cycle,
                    )
                    dedicated_resources.autoscaling_metric_specs.extend(
                        [autoscaling_metric_spec]
                    )

            dedicated_resources.machine_spec = machine_spec
            deployed_model.dedicated_resources = dedicated_resources

        elif supports_automatic_resources:
            deployed_model.automatic_resources = (
                gca_machine_resources_compat.AutomaticResources(
                    min_replica_count=min_replica_count,
                    max_replica_count=max_replica_count,
                )
            )
        else:
            _LOGGER.warning(
                "Model does not support deployment. "
                "See https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1#google.cloud.aiplatform.v1.Model.FIELDS.repeated.google.cloud.aiplatform.v1.Model.DeploymentResourcesType.google.cloud.aiplatform.v1.Model.supported_deployment_resources_types"
            )

        deployed_model.explanation_spec = explanation_spec

        # Checking if traffic percentage is valid
        # TODO(b/221059294) PrivateEndpoint should support traffic split
        if traffic_split is None and not network:
            # new model traffic needs to be 100 if no pre-existing models
            if not endpoint_resource_traffic_split:
                # default scenario
                if traffic_percentage == 0:
                    traffic_percentage = 100
                # verify user specified 100
                elif traffic_percentage < 100:
                    raise ValueError(
                        """There are currently no deployed models so the traffic
                        percentage for this deployed model needs to be 100."""
                    )
            traffic_split = cls._allocate_traffic(
                traffic_split=dict(endpoint_resource_traffic_split),
                traffic_percentage=traffic_percentage,
            )

        operation_future = api_client.deploy_model(
            endpoint=endpoint_resource_name,
            deployed_model=deployed_model,
            traffic_split=traffic_split,
            metadata=metadata,
            timeout=deploy_request_timeout,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Deploy", "model", cls, operation_future
        )

        operation_future.result(timeout=None)

    def undeploy(
        self,
        deployed_model_id: str,
        traffic_split: Optional[Dict[str, int]] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync=True,
    ) -> None:
        """Undeploys a deployed model.

        The model to be undeployed should have no traffic or user must provide
        a new traffic_split with the remaining deployed models. Refer
        to `Endpoint.traffic_split` for the current traffic split mapping.

        Args:
            deployed_model_id (str):
                Required. The ID of the DeployedModel to be undeployed from the
                Endpoint.
            traffic_split (Dict[str, int]):
                Optional. A map of DeployedModel IDs to the percentage of
                this Endpoint's traffic that should be forwarded to that DeployedModel.
                Required if undeploying a model with non-zero traffic from an Endpoint
                with multiple deployed models. The traffic percentage values must add
                up to 100, or map must be empty if the Endpoint is to not accept any traffic
                at the moment. If a DeployedModel's ID is not listed in this map, then it
                receives no traffic.
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
        """
        self._sync_gca_resource_if_skipped()

        if traffic_split is not None:
            if deployed_model_id in traffic_split and traffic_split[deployed_model_id]:
                raise ValueError("Model being undeployed should have 0 traffic.")
            if sum(traffic_split.values()) != 100:
                raise ValueError(
                    "Sum of all traffic within traffic split needs to be 100."
                )

        # Two or more models deployed to Endpoint and remaining traffic will be zero
        elif (
            len(self.traffic_split) > 1
            and deployed_model_id in self._gca_resource.traffic_split
            and self._gca_resource.traffic_split[deployed_model_id] == 100
        ):
            raise ValueError(
                f"Undeploying deployed model '{deployed_model_id}' would leave the remaining "
                "traffic split at 0%. Traffic split must add up to 100% when models are "
                "deployed. Please undeploy the other models first or provide an updated "
                "traffic_split."
            )

        self._undeploy(
            deployed_model_id=deployed_model_id,
            traffic_split=traffic_split,
            metadata=metadata,
            sync=sync,
        )

    @base.optional_sync()
    def _undeploy(
        self,
        deployed_model_id: str,
        traffic_split: Optional[Dict[str, int]] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync=True,
    ) -> None:
        """Undeploys a deployed model.

        Proportionally adjusts the traffic_split among the remaining deployed
        models of the endpoint.

        Args:
            deployed_model_id (str):
                Required. The ID of the DeployedModel to be undeployed from the
                Endpoint.
            traffic_split (Dict[str, int]):
                Optional. A map from a DeployedModel's ID to the percentage of
                this Endpoint's traffic that should be forwarded to that DeployedModel.
                If a DeployedModel's ID is not listed in this map, then it receives
                no traffic. The traffic percentage values must add up to 100, or
                map must be empty if the Endpoint is to not accept any traffic at
                the moment. Key for model being deployed is "0". Should not be
                provided if traffic_percentage is provided.
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
        """
        self._sync_gca_resource_if_skipped()
        current_traffic_split = traffic_split or dict(self._gca_resource.traffic_split)

        if deployed_model_id in current_traffic_split:
            current_traffic_split = self._unallocate_traffic(
                traffic_split=current_traffic_split,
                deployed_model_id=deployed_model_id,
            )
            current_traffic_split.pop(deployed_model_id)

        _LOGGER.log_action_start_against_resource("Undeploying", "model", self)

        operation_future = self.api_client.undeploy_model(
            endpoint=self.resource_name,
            deployed_model_id=deployed_model_id,
            traffic_split=current_traffic_split,
            metadata=metadata,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Undeploy", "model", self.__class__, operation_future
        )

        # block before returning
        operation_future.result()

        _LOGGER.log_action_completed_against_resource("model", "undeployed", self)

        # update local resource
        self._sync_gca_resource()

    def update(
        self,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        traffic_split: Optional[Dict[str, int]] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        update_request_timeout: Optional[float] = None,
    ) -> "Endpoint":
        """Updates an endpoint.

        Example usage:
            my_endpoint = my_endpoint.update(
                display_name='my-updated-endpoint',
                description='my updated description',
                labels={'key': 'value'},
                traffic_split={
                    '123456': 20,
                    '234567': 80,
                },
            )

        Args:
            display_name (str):
                Optional. The display name of the Endpoint.
                The name can be up to 128 characters long and can be consist of any UTF-8
                characters.
            description (str):
                Optional. The description of the Endpoint.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to organize your Endpoints.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                See https://goo.gl/xmQnxf for more information and examples of labels.
            traffic_split (Dict[str, int]):
                Optional. A map from a DeployedModel's ID to the percentage of this Endpoint's
                traffic that should be forwarded to that DeployedModel.
                If a DeployedModel's ID is not listed in this map, then it receives no traffic.
                The traffic percentage values must add up to 100, or map must be empty if
                the Endpoint is to not accept any traffic at a moment.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            update_request_timeout (float):
                Optional. The timeout for the update request in seconds.

        Returns:
            Endpoint (aiplatform.Prediction):
                Updated endpoint resource.

        Raises:
            ValueError: If `labels` is not the correct format.
        """

        self.wait()

        current_endpoint_proto = self.gca_resource
        copied_endpoint_proto = current_endpoint_proto.__class__(current_endpoint_proto)

        update_mask: List[str] = []

        if display_name:
            utils.validate_display_name(display_name)
            copied_endpoint_proto.display_name = display_name
            update_mask.append("display_name")

        if description:
            copied_endpoint_proto.description = description
            update_mask.append("description")

        if labels:
            utils.validate_labels(labels)
            copied_endpoint_proto.labels = labels
            update_mask.append("labels")

        if traffic_split:
            update_mask.append("traffic_split")
            copied_endpoint_proto.traffic_split = traffic_split

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)

        _LOGGER.log_action_start_against_resource(
            "Updating",
            "endpoint",
            self,
        )

        self._gca_resource = self.api_client.update_endpoint(
            endpoint=copied_endpoint_proto,
            update_mask=update_mask,
            metadata=request_metadata,
            timeout=update_request_timeout,
        )

        _LOGGER.log_action_completed_against_resource("endpoint", "updated", self)

        return self

    def predict(
        self,
        instances: List,
        parameters: Optional[Dict] = None,
        timeout: Optional[float] = None,
        use_raw_predict: Optional[bool] = False,
    ) -> Prediction:
        """Make a prediction against this Endpoint.

        Args:
            instances (List):
                Required. The instances that are the input to the
                prediction call. A DeployedModel may have an upper limit
                on the number of instances it supports per request, and
                when it is exceeded the prediction call errors in case
                of AutoML Models, or, in case of customer created
                Models, the behaviour is as documented by that Model.
                The schema of any single instance may be specified via
                Endpoint's DeployedModels'
                [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``instance_schema_uri``.
            parameters (Dict):
                The parameters that govern the prediction. The schema of
                the parameters may be specified via Endpoint's
                DeployedModels' [Model's
                ][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``parameters_schema_uri``.
            timeout (float): Optional. The timeout for this request in seconds.
            use_raw_predict (bool):
                Optional. Default value is False. If set to True, the underlying prediction call will be made
                against Endpoint.raw_predict().

        Returns:
            prediction (aiplatform.Prediction):
                Prediction with returned predictions and Model ID.
        """
        self.wait()
        if use_raw_predict:
            raw_predict_response = self.raw_predict(
                body=json.dumps({"instances": instances, "parameters": parameters}),
                headers={"Content-Type": "application/json"},
            )
            json_response = raw_predict_response.json()
            return Prediction(
                predictions=json_response["predictions"],
                metadata=json_response.get("metadata"),
                deployed_model_id=raw_predict_response.headers[
                    _RAW_PREDICT_DEPLOYED_MODEL_ID_KEY
                ],
                model_resource_name=raw_predict_response.headers[
                    _RAW_PREDICT_MODEL_RESOURCE_KEY
                ],
                model_version_id=raw_predict_response.headers.get(
                    _RAW_PREDICT_MODEL_VERSION_ID_KEY, None
                ),
            )
        else:
            prediction_response = self._prediction_client.predict(
                endpoint=self._gca_resource.name,
                instances=instances,
                parameters=parameters,
                timeout=timeout,
            )
            if prediction_response._pb.metadata:
                metadata = json_format.MessageToDict(prediction_response._pb.metadata)
            else:
                metadata = None

            return Prediction(
                predictions=[
                    json_format.MessageToDict(item)
                    for item in prediction_response.predictions.pb
                ],
                metadata=metadata,
                deployed_model_id=prediction_response.deployed_model_id,
                model_version_id=prediction_response.model_version_id,
                model_resource_name=prediction_response.model,
            )

    async def predict_async(
        self,
        instances: List,
        *,
        parameters: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> Prediction:
        """Make an asynchronous prediction against this Endpoint.
        Example usage:
            ```
            response = await my_endpoint.predict_async(instances=[...])
            my_predictions = response.predictions
            ```

        Args:
            instances (List):
                Required. The instances that are the input to the
                prediction call. A DeployedModel may have an upper limit
                on the number of instances it supports per request, and
                when it is exceeded the prediction call errors in case
                of AutoML Models, or, in case of customer created
                Models, the behaviour is as documented by that Model.
                The schema of any single instance may be specified via
                Endpoint's DeployedModels'
                [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``instance_schema_uri``.
            parameters (Dict):
                Optional. The parameters that govern the prediction. The schema of
                the parameters may be specified via Endpoint's
                DeployedModels' [Model's
                ][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``parameters_schema_uri``.
            timeout (float): Optional. The timeout for this request in seconds.

        Returns:
            prediction (aiplatform.Prediction):
                Prediction with returned predictions and Model ID.
        """
        self.wait()

        prediction_response = await self._prediction_async_client.predict(
            endpoint=self._gca_resource.name,
            instances=instances,
            parameters=parameters,
            timeout=timeout,
        )
        if prediction_response._pb.metadata:
            metadata = json_format.MessageToDict(prediction_response._pb.metadata)
        else:
            metadata = None

        return Prediction(
            predictions=[
                json_format.MessageToDict(item)
                for item in prediction_response.predictions.pb
            ],
            metadata=metadata,
            deployed_model_id=prediction_response.deployed_model_id,
            model_version_id=prediction_response.model_version_id,
            model_resource_name=prediction_response.model,
        )

    def raw_predict(
        self, body: bytes, headers: Dict[str, str]
    ) -> requests.models.Response:
        """Makes a prediction request using arbitrary headers.

        Example usage:
            my_endpoint = aiplatform.Endpoint(ENDPOINT_ID)
            response = my_endpoint.raw_predict(
                body = b'{"instances":[{"feat_1":val_1, "feat_2":val_2}]}'
                headers = {'Content-Type':'application/json'}
            )
            status_code = response.status_code
            results = json.dumps(response.text)

        Args:
            body (bytes):
                The body of the prediction request in bytes. This must not exceed 1.5 mb per request.
            headers (Dict[str, str]):
                The header of the request as a dictionary. There are no restrictions on the header.

        Returns:
            A requests.models.Response object containing the status code and prediction results.
        """
        if not self.authorized_session:
            self.credentials._scopes = constants.base.DEFAULT_AUTHED_SCOPES
            self.raw_predict_request_url = f"https://{self.location}-{constants.base.API_BASE_PATH}/v1/projects/{self.project}/locations/{self.location}/endpoints/{self.name}:rawPredict"
            self.authorized_session = google_auth_requests.AuthorizedSession(
                self.credentials
            )

        return self.authorized_session.post(
            url=self.raw_predict_request_url, data=body, headers=headers
        )

    def explain(
        self,
        instances: List[Dict],
        parameters: Optional[Dict] = None,
        deployed_model_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Prediction:
        """Make a prediction with explanations against this Endpoint.

        Example usage:
            response = my_endpoint.explain(instances=[...])
            my_explanations = response.explanations

        Args:
            instances (List):
                Required. The instances that are the input to the
                prediction call. A DeployedModel may have an upper limit
                on the number of instances it supports per request, and
                when it is exceeded the prediction call errors in case
                of AutoML Models, or, in case of customer created
                Models, the behaviour is as documented by that Model.
                The schema of any single instance may be specified via
                Endpoint's DeployedModels'
                [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``instance_schema_uri``.
            parameters (Dict):
                The parameters that govern the prediction. The schema of
                the parameters may be specified via Endpoint's
                DeployedModels' [Model's
                ][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``parameters_schema_uri``.
            deployed_model_id (str):
                Optional. If specified, this ExplainRequest will be served by the
                chosen DeployedModel, overriding this Endpoint's traffic split.
            timeout (float): Optional. The timeout for this request in seconds.

        Returns:
            prediction (aiplatform.Prediction):
                Prediction with returned predictions, explanations, and Model ID.
        """
        self.wait()

        explain_response = self._prediction_client.explain(
            endpoint=self.resource_name,
            instances=instances,
            parameters=parameters,
            deployed_model_id=deployed_model_id,
            timeout=timeout,
        )

        return Prediction(
            predictions=[
                json_format.MessageToDict(item)
                for item in explain_response.predictions.pb
            ],
            deployed_model_id=explain_response.deployed_model_id,
            explanations=explain_response.explanations,
        )

    async def explain_async(
        self,
        instances: List[Dict],
        *,
        parameters: Optional[Dict] = None,
        deployed_model_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Prediction:
        """Make a prediction with explanations against this Endpoint.

        Example usage:
            ```
            response = await my_endpoint.explain_async(instances=[...])
            my_explanations = response.explanations
            ```

        Args:
            instances (List):
                Required. The instances that are the input to the
                prediction call. A DeployedModel may have an upper limit
                on the number of instances it supports per request, and
                when it is exceeded the prediction call errors in case
                of AutoML Models, or, in case of customer created
                Models, the behaviour is as documented by that Model.
                The schema of any single instance may be specified via
                Endpoint's DeployedModels'
                [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``instance_schema_uri``.
            parameters (Dict):
                The parameters that govern the prediction. The schema of
                the parameters may be specified via Endpoint's
                DeployedModels' [Model's
                ][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``parameters_schema_uri``.
            deployed_model_id (str):
                Optional. If specified, this ExplainRequest will be served by the
                chosen DeployedModel, overriding this Endpoint's traffic split.
            timeout (float): Optional. The timeout for this request in seconds.

        Returns:
            prediction (aiplatform.Prediction):
                Prediction with returned predictions, explanations, and Model ID.
        """
        self.wait()

        explain_response = await self._prediction_async_client.explain(
            endpoint=self.resource_name,
            instances=instances,
            parameters=parameters,
            deployed_model_id=deployed_model_id,
            timeout=timeout,
        )

        return Prediction(
            predictions=[
                json_format.MessageToDict(item)
                for item in explain_response.predictions.pb
            ],
            deployed_model_id=explain_response.deployed_model_id,
            explanations=explain_response.explanations,
        )

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["models.Endpoint"]:
        """List all Endpoint resource instances.

        Example Usage:
            aiplatform.Endpoint.list(
                filter='labels.my_label="my_label_value" OR display_name=!"old_endpoint"',
            )

        Args:
            filter (str):
                Optional. An expression for filtering the results of the request.
                For field names both snake_case and camelCase are supported.
            order_by (str):
                Optional. A comma-separated list of fields to order by, sorted in
                ascending order. Use "desc" after a field name for descending.
                Supported fields: `display_name`, `create_time`, `update_time`
            project (str):
                Optional. Project to retrieve list from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve list from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve list. Overrides
                credentials set in aiplatform.init.

        Returns:
            List[models.Endpoint]:
                A list of Endpoint resource objects
        """

        return cls._list_with_local_order(
            cls_filter=lambda ep: not bool(
                ep.network
            ),  # `network` is empty for public Endpoints
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
        )

    def list_models(self) -> List[gca_endpoint_compat.DeployedModel]:
        """Returns a list of the models deployed to this Endpoint.

        Returns:
            deployed_models (List[aiplatform.gapic.DeployedModel]):
                A list of the models deployed in this Endpoint.
        """
        self._sync_gca_resource()
        return list(self._gca_resource.deployed_models)

    def undeploy_all(self, sync: bool = True) -> "Endpoint":
        """Undeploys every model deployed to this Endpoint.

        Args:
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
        """
        self._sync_gca_resource()

        models_in_traffic_split = sorted(  # Undeploy zero traffic models first
            self._gca_resource.traffic_split.keys(),
            key=lambda id: self._gca_resource.traffic_split[id],
        )

        # Some deployed models may not in the traffic_split dict.
        # These models have 0% traffic and should be undeployed first.
        models_not_in_traffic_split = [
            deployed_model.id
            for deployed_model in self._gca_resource.deployed_models
            if deployed_model.id not in models_in_traffic_split
        ]

        models_to_undeploy = models_not_in_traffic_split + models_in_traffic_split

        for deployed_model in models_to_undeploy:
            self._undeploy(deployed_model_id=deployed_model, sync=sync)

        return self

    def delete(self, force: bool = False, sync: bool = True) -> None:
        """Deletes this Vertex AI Endpoint resource. If force is set to True,
        all models on this Endpoint will be undeployed prior to deletion.

        Args:
            force (bool):
                Required. If force is set to True, all deployed models on this
                Endpoint will be undeployed first. Default is False.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.

        Raises:
            FailedPrecondition: If models are deployed on this Endpoint and force = False.
        """
        if force:
            self.undeploy_all(sync=sync)

        super().delete(sync=sync)


class PrivateEndpoint(Endpoint):
    """
    Represents a Vertex AI PrivateEndpoint resource.

    Read more [about private endpoints in the documentation.](https://cloud.google.com/vertex-ai/docs/predictions/using-private-endpoints)
    """

    def __init__(
        self,
        endpoint_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a PrivateEndpoint resource.

        Example usage:
            my_private_endpoint = aiplatform.PrivateEndpoint(
                endpoint_name="projects/123/locations/us-central1/endpoints/1234567891234567890"
            )

            or (when project and location are initialized)

            my_private_endpoint = aiplatform.PrivateEndpoint(
                endpoint_name="1234567891234567890"
            )

        Args:
            endpoint_name (str):
                Required. A fully-qualified endpoint resource name or endpoint ID.
                Example: "projects/123/locations/us-central1/endpoints/my_endpoint_id" or
                "my_endpoint_id" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve endpoint from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve endpoint from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.

        Raises:
            ValueError: If the Endpoint being retrieved is not a PrivateEndpoint.
            ImportError: If there is an issue importing the `urllib3` package.
        """
        try:
            import urllib3
        except ImportError:
            raise ImportError(
                "Cannot import the urllib3 HTTP client. Please install google-cloud-aiplatform[private_endpoints]."
            )

        super().__init__(
            endpoint_name=endpoint_name,
            project=project,
            location=location,
            credentials=credentials,
        )

        if not self.network:
            raise ValueError(
                "Please ensure the Endpoint being retrieved is a PrivateEndpoint."
            )

        self._http_client = urllib3.PoolManager()

    @property
    def predict_http_uri(self) -> Optional[str]:
        """HTTP path to send prediction requests to, used when calling `PrivateEndpoint.predict()`"""
        if not self._gca_resource.deployed_models:
            return None
        return self._gca_resource.deployed_models[0].private_endpoints.predict_http_uri

    @property
    def explain_http_uri(self) -> Optional[str]:
        """HTTP path to send explain requests to, used when calling `PrivateEndpoint.explain()`"""
        if not self._gca_resource.deployed_models:
            return None
        return self._gca_resource.deployed_models[0].private_endpoints.explain_http_uri

    @property
    def health_http_uri(self) -> Optional[str]:
        """HTTP path to send health check requests to, used when calling `PrivateEndpoint.health_check()`"""
        if not self._gca_resource.deployed_models:
            return None
        return self._gca_resource.deployed_models[0].private_endpoints.health_http_uri

    @classmethod
    def create(
        cls,
        display_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        network: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
        sync=True,
    ) -> "PrivateEndpoint":
        """Creates a new PrivateEndpoint.

        Example usage:
            my_private_endpoint = aiplatform.PrivateEndpoint.create(
                display_name="my_endpoint_name",
                project="my_project_id",
                location="us-central1",
                network="projects/123456789123/global/networks/my_vpc"
            )

            or (when project and location are initialized)

            my_private_endpoint = aiplatform.PrivateEndpoint.create(
                display_name="my_endpoint_name",
                network="projects/123456789123/global/networks/my_vpc"
            )

        Args:
            display_name (str):
                Required. The user-defined name of the Endpoint.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            project (str):
                Optional. Project to retrieve endpoint from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve endpoint from. If not set, location
                set in aiplatform.init will be used.
            network (str):
                Optional. The full name of the Compute Engine network to which
                this Endpoint will be peered. E.g. "projects/123456789123/global/networks/my_vpc".
                Private services access must already be configured for the network.
                If left unspecified, the network set with aiplatform.init will be used.
            description (str):
                Optional. The description of the Endpoint.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Endpoints.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.

        Returns:
            endpoint (aiplatform.PrivateEndpoint):
                Created endpoint.

        Raises:
            ValueError: A network must be instantiated when creating a PrivateEndpoint.
        """
        api_client = cls._instantiate_client(location=location, credentials=credentials)

        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)

        project = project or initializer.global_config.project
        location = location or initializer.global_config.location
        network = network or initializer.global_config.network

        if not network:
            raise ValueError(
                "Please provide required argument `network` or set network"
                "using aiplatform.init(network=...)"
            )

        return cls._create(
            api_client=api_client,
            display_name=display_name,
            project=project,
            location=location,
            description=description,
            labels=labels,
            credentials=credentials,
            encryption_spec=initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            ),
            network=network,
            sync=sync,
        )

    @classmethod
    def _construct_sdk_resource_from_gapic(
        cls,
        gapic_resource: proto.Message,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "PrivateEndpoint":
        """Given a GAPIC PrivateEndpoint object, return the SDK representation.

        Args:
            gapic_resource (proto.Message):
                A GAPIC representation of a PrivateEndpoint resource, usually
                retrieved by a get_* or in a list_* API call.
            project (str):
                Optional. Project to construct Endpoint object from. If not set,
                project set in aiplatform.init will be used.
            location (str):
                Optional. Location to construct Endpoint object from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to construct Endpoint.
                Overrides credentials set in aiplatform.init.

        Returns:
            endpoint (aiplatform.PrivateEndpoint):
                An initialized PrivateEndpoint resource.

        Raises:
            ImportError: If there is an issue importing the `urllib3` package.
        """
        try:
            import urllib3
        except ImportError:
            raise ImportError(
                "Cannot import the urllib3 HTTP client. Please install google-cloud-aiplatform[private_endpoints]."
            )

        endpoint = super()._construct_sdk_resource_from_gapic(
            gapic_resource=gapic_resource,
            project=project,
            location=location,
            credentials=credentials,
        )

        endpoint._http_client = urllib3.PoolManager()

        return endpoint

    def _http_request(
        self,
        method: str,
        url: str,
        body: Optional[Dict[Any, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> "urllib3.response.HTTPResponse":  # type: ignore # noqa: F821
        """Helper function used to perform HTTP requests for PrivateEndpoint.

        Args:
            method (str):
                Required. The HTTP request method to use. Example: "POST" or "GET"
            url (str):
                Required. The url used to send requests and get responses from.
            body (Dict[Any, Any]):
                Optional. Data sent to the url in the HTTP request. For a PrivateEndpoint,
                an instance is sent and a prediction response is expected.
            headers (Dict[str, str]):
                Optional. Header in the HTTP request.

        Returns:
            urllib3.response.HTTPResponse:
                A HTTP Response container.

        Raises:
            ImportError: If there is an issue importing the `urllib3` package.
            RuntimeError: If a HTTP request could not be made.
            RuntimeError: A connection could not be established with the PrivateEndpoint and
                a HTTP request could not be made.
        """
        try:
            import urllib3
        except ImportError:
            raise ImportError(
                "Cannot import the urllib3 HTTP client. Please install google-cloud-aiplatform[private_endpoints]."
            )

        try:
            response = self._http_client.request(
                method=method, url=url, body=body, headers=headers
            )

            if response.status < _SUCCESSFUL_HTTP_RESPONSE:
                return response
            else:
                raise RuntimeError(
                    f"{response.status} - Failed to make request, see response: "
                    + response.data.decode("utf-8")
                )

        except urllib3.exceptions.MaxRetryError as exc:
            raise RuntimeError(
                f"Failed to make a {method} request to this URI, make sure: "
                " this call is being made inside the network this PrivateEndpoint is peered to "
                f"({self._gca_resource.network}), calling health_check() returns True, "
                f"and that {url} is a valid URL."
            ) from exc

    def predict(self, instances: List, parameters: Optional[Dict] = None) -> Prediction:
        """Make a prediction against this PrivateEndpoint using a HTTP request.
        This method must be called within the network the PrivateEndpoint is peered to.
        Otherwise, the predict() call will fail with error code 404. To check, use `PrivateEndpoint.network`.

        Example usage:
            response = my_private_endpoint.predict(instances=[...])
            my_predictions = response.predictions

        Args:
            instances (List):
                Required. The instances that are the input to the
                prediction call. Instance types mut be JSON serializable.
                A DeployedModel may have an upper limit
                on the number of instances it supports per request, and
                when it is exceeded the prediction call errors in case
                of AutoML Models, or, in case of customer created
                Models, the behaviour is as documented by that Model.
                The schema of any single instance may be specified via
                Endpoint's DeployedModels'
                [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``instance_schema_uri``.
            parameters (Dict):
                The parameters that govern the prediction. The schema of
                the parameters may be specified via Endpoint's
                DeployedModels' [Model's
                ][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                ``parameters_schema_uri``.

        Returns:
            prediction (aiplatform.Prediction):
                Prediction object with returned predictions and Model ID.

        Raises:
            RuntimeError: If a model has not been deployed a request cannot be made.
        """
        self.wait()
        self._sync_gca_resource_if_skipped()

        if not self._gca_resource.deployed_models:
            raise RuntimeError(
                "Cannot make a predict request because a model has not been deployed on this Private"
                "Endpoint. Please ensure a model has been deployed."
            )

        response = self._http_request(
            method="POST",
            url=self.predict_http_uri,
            body=json.dumps({"instances": instances}),
            headers={"Content-Type": "application/json"},
        )

        prediction_response = json.loads(response.data)

        return Prediction(
            predictions=prediction_response.get("predictions"),
            metadata=prediction_response.get("metadata"),
            deployed_model_id=self._gca_resource.deployed_models[0].id,
        )

    def raw_predict(
        self, body: bytes, headers: Dict[str, str]
    ) -> requests.models.Response:
        """Make a prediction request using arbitrary headers.
        This method must be called within the network the PrivateEndpoint is peered to.
        Otherwise, the predict() call will fail with error code 404. To check, use `PrivateEndpoint.network`.

        Example usage:
            my_endpoint = aiplatform.PrivateEndpoint(ENDPOINT_ID)
            response = my_endpoint.raw_predict(
                body = b'{"instances":[{"feat_1":val_1, "feat_2":val_2}]}'
                headers = {'Content-Type':'application/json'}
            )
            status_code = response.status_code
            results = json.dumps(response.text)

        Args:
            body (bytes):
                The body of the prediction request in bytes. This must not exceed 1.5 mb per request.
            headers (Dict[str, str]):
                The header of the request as a dictionary. There are no restrictions on the header.

        Returns:
            A requests.models.Response object containing the status code and prediction results.
        """
        self.wait()
        return self._http_request(
            method="POST",
            url=self.predict_http_uri,
            body=body,
            headers=headers,
        )

    def explain(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not support 'explain' as of now."
        )

    def health_check(self) -> bool:
        """
        Makes a request to this PrivateEndpoint's health check URI. Must be within network
        that this PrivateEndpoint is in.

        Example Usage:
            if my_private_endpoint.health_check():
                print("PrivateEndpoint is healthy!")

        Returns:
            bool:
                Checks if calls can be made to this PrivateEndpoint.

        Raises:
            RuntimeError: If a model has not been deployed a request cannot be made.
        """
        self.wait()
        self._sync_gca_resource_if_skipped()

        if not self._gca_resource.deployed_models:
            raise RuntimeError(
                "Cannot make a health check request because a model has not been deployed on this Private"
                "Endpoint. Please ensure a model has been deployed."
            )

        response = self._http_request(
            method="GET",
            url=self.health_http_uri,
        )

        return response.status < _SUCCESSFUL_HTTP_RESPONSE

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["models.PrivateEndpoint"]:
        """List all PrivateEndpoint resource instances.

        Example Usage:
            my_private_endpoints = aiplatform.PrivateEndpoint.list()

            or

            my_private_endpoints = aiplatform.PrivateEndpoint.list(
                filter='labels.my_label="my_label_value" OR display_name=!"old_endpoint"',
            )

        Args:
            filter (str):
                Optional. An expression for filtering the results of the request.
                For field names both snake_case and camelCase are supported.
            order_by (str):
                Optional. A comma-separated list of fields to order by, sorted in
                ascending order. Use "desc" after a field name for descending.
                Supported fields: `display_name`, `create_time`, `update_time`
            project (str):
                Optional. Project to retrieve list from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve list from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve list. Overrides
                credentials set in aiplatform.init.

        Returns:
            List[models.PrivateEndpoint]:
                A list of PrivateEndpoint resource objects.
        """

        return cls._list_with_local_order(
            cls_filter=lambda ep: bool(
                ep.network
            ),  # Only PrivateEndpoints have a network set
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
        )

    def deploy(
        self,
        model: "Model",
        deployed_model_display_name: Optional[str] = None,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        service_account: Optional[str] = None,
        explanation_metadata: Optional[aiplatform.explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[
            aiplatform.explain.ExplanationParameters
        ] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync=True,
        disable_container_logging: bool = False,
    ) -> None:
        """Deploys a Model to the PrivateEndpoint.

        Example Usage:
            my_private_endpoint.deploy(
                model=my_model
            )

        Args:
            model (aiplatform.Model):
                Required. Model to be deployed.
            deployed_model_display_name (str):
                Optional. The display name of the DeployedModel. If not provided
                upon creation, the Model's display_name is used.
            machine_type (str):
                Optional. The type of machine. Not specifying machine type will
                result in model to be deployed with automatic resources.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the larger value of min_replica_count or 1 will
                be used. If value provided is smaller than min_replica_count, it
                will automatically be increased to be min_replica_count.
            accelerator_type (str):
                Optional. Hardware accelerator type. Must also set accelerator_count if used.
                One of ACCELERATOR_TYPE_UNSPECIFIED, NVIDIA_TESLA_K80, NVIDIA_TESLA_P100,
                NVIDIA_TESLA_V100, NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int):
                Optional. The number of accelerators to attach to a worker replica.
            service_account (str):
                The service account that the DeployedModel's container runs as. Specify the
                email address of the service account. If this service account is not
                specified, the container runs as a service account that doesn't have access
                to the resource project.
                Users deploying the Model must have the `iam.serviceAccounts.actAs`
                permission on this service account.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Metadata describing the Model's input and output for explanation.
                `explanation_metadata` is optional while `explanation_parameters` must be
                specified when used.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
        """
        self._validate_deploy_args(
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            deployed_model_display_name=deployed_model_display_name,
            traffic_split=None,
            traffic_percentage=100,
        )

        explanation_spec = _explanation_utils.create_and_validate_explanation_spec(
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
        )

        self._deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            traffic_percentage=100,
            traffic_split=None,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            service_account=service_account,
            explanation_spec=explanation_spec,
            metadata=metadata,
            sync=sync,
            disable_container_logging=disable_container_logging,
        )

    def undeploy(
        self,
        deployed_model_id: str,
        sync=True,
    ) -> None:
        """Undeploys a deployed model from the PrivateEndpoint.

        Example Usage:
            my_private_endpoint.undeploy(
                deployed_model_id="1234567891232567891"
            )

            or

            my_deployed_model_id = my_private_endpoint.list_models()[0].id
            my_private_endpoint.undeploy(
                deployed_model_id=my_deployed_model_id
            )

        Args:
            deployed_model_id (str):
                Required. The ID of the DeployedModel to be undeployed from the
                PrivateEndpoint. Use PrivateEndpoint.list_models() to get the
                deployed model ID.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
        """
        self._sync_gca_resource_if_skipped()

        # TODO(b/211351292): Add traffic splitting for PrivateEndpoint
        self._undeploy(
            deployed_model_id=deployed_model_id,
            traffic_split=None,
            sync=sync,
        )

    def delete(self, force: bool = False, sync: bool = True) -> None:
        """Deletes this Vertex AI PrivateEndpoint resource. If force is set to True,
        all models on this PrivateEndpoint will be undeployed prior to deletion.

        Args:
            force (bool):
                Required. If force is set to True, all deployed models on this
                Endpoint will be undeployed first. Default is False.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.

        Raises:
            FailedPrecondition: If models are deployed on this Endpoint and force = False.
        """
        if force and self._gca_resource.deployed_models:
            self.undeploy(
                deployed_model_id=self._gca_resource.deployed_models[0].id,
                sync=sync,
            )

        super().delete(force=False, sync=sync)


class Model(base.VertexAiResourceNounWithFutureManager, base.PreviewMixin):
    client_class = utils.ModelClientWithOverride
    _resource_noun = "models"
    _getter_method = "get_model"
    _list_method = "list_models"
    _delete_method = "delete_model"
    _parse_resource_name_method = "parse_model_path"
    _format_resource_name_method = "model_path"
    _preview_class = "google.cloud.aiplatform.aiplatform.preview.models.Model"

    @property
    def preview(self):
        """Return a Model instance with preview features enabled."""
        from google.cloud.aiplatform.preview import models as preview_models

        if not hasattr(self, "_preview_instance"):
            self._preview_instance = preview_models.Model(
                self.resource_name, credentials=self.credentials
            )

        return self._preview_instance

    @property
    def uri(self) -> Optional[str]:
        """Path to the directory containing the Model artifact and any of its
        supporting files. Not present for AutoML Models."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.artifact_uri or None

    @property
    def description(self) -> str:
        """Description of the model."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.description

    @property
    def supported_export_formats(
        self,
    ) -> Dict[str, List[gca_model_compat.Model.ExportFormat.ExportableContent]]:
        """The formats and content types in which this Model may be exported.
        If empty, this Model is not available for export.

        For example, if this model can be exported as a Tensorflow SavedModel and
        have the artifacts written to Cloud Storage, the expected value would be:

            {'tf-saved-model': [<ExportableContent.ARTIFACT: 1>]}
        """
        self._assert_gca_resource_is_available()
        return {
            export_format.id: [
                gca_model_compat.Model.ExportFormat.ExportableContent(content)
                for content in export_format.exportable_contents
            ]
            for export_format in self._gca_resource.supported_export_formats
        }

    @property
    def supported_deployment_resources_types(
        self,
    ) -> List[model_v1.Model.DeploymentResourcesType]:
        """List of deployment resource types accepted for this Model.

        When this Model is deployed, its prediction resources are described by
        the `prediction_resources` field of the objects returned by
        `Endpoint.list_models()`. Because not all Models support all resource
        configuration types, the configuration types this Model supports are
        listed here.

        If no configuration types are listed, the Model cannot be
        deployed to an `Endpoint` and does not support online predictions
        (`Endpoint.predict()` or `Endpoint.explain()`). Such a Model can serve
        predictions by using a `BatchPredictionJob`, if it has at least one entry
        each in `Model.supported_input_storage_formats` and
        `Model.supported_output_storage_formats`."""
        self._assert_gca_resource_is_available()
        return list(self._gca_resource.supported_deployment_resources_types)

    @property
    def supported_input_storage_formats(self) -> List[str]:
        """The formats this Model supports in the `input_config` field of a
        `BatchPredictionJob`. If `Model.predict_schemata.instance_schema_uri`
        exists, the instances should be given as per that schema.

        [Read the docs for more on batch prediction formats](https://cloud.google.com/vertex-ai/docs/predictions/batch-predictions#batch_request_input)

        If this Model doesn't support any of these formats it means it cannot be
        used with a `BatchPredictionJob`. However, if it has
        `supported_deployment_resources_types`, it could serve online predictions
        by using `Endpoint.predict()` or `Endpoint.explain()`.
        """
        self._assert_gca_resource_is_available()
        return list(self._gca_resource.supported_input_storage_formats)

    @property
    def supported_output_storage_formats(self) -> List[str]:
        """The formats this Model supports in the `output_config` field of a
        `BatchPredictionJob`.

        If both `Model.predict_schemata.instance_schema_uri` and
        `Model.predict_schemata.prediction_schema_uri` exist, the predictions
        are returned together with their instances. In other words, the
        prediction has the original instance data first, followed by the actual
        prediction content (as per the schema).

        [Read the docs for more on batch prediction formats](https://cloud.google.com/vertex-ai/docs/predictions/batch-predictions)

        If this Model doesn't support any of these formats it means it cannot be
        used with a `BatchPredictionJob`. However, if it has
        `supported_deployment_resources_types`, it could serve online predictions
        by using `Endpoint.predict()` or `Endpoint.explain()`.
        """
        self._assert_gca_resource_is_available()
        return list(self._gca_resource.supported_output_storage_formats)

    @property
    def predict_schemata(self) -> Optional[model_v1.PredictSchemata]:
        """The schemata that describe formats of the Model's predictions and
        explanations, if available."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "predict_schemata")

    @property
    def training_job(self) -> Optional["aiplatform.training_jobs._TrainingJob"]:
        """The TrainingJob that uploaded this Model, if any.

        Raises:
            api_core.exceptions.NotFound: If the Model's training job resource
                cannot be found on the Vertex service.
        """
        self._assert_gca_resource_is_available()
        job_name = getattr(self._gca_resource, "training_pipeline")

        if not job_name:
            return None

        try:
            return aiplatform.training_jobs._TrainingJob._get_and_return_subclass(
                resource_name=job_name,
                project=self.project,
                location=self.location,
                credentials=self.credentials,
            )
        except api_exceptions.NotFound as exc:
            raise api_exceptions.NotFound(
                f"The training job used to create this model could not be found: {job_name}"
            ) from exc

    @property
    def container_spec(self) -> Optional[model_v1.ModelContainerSpec]:
        """The specification of the container that is to be used when deploying
        this Model. Not present for AutoML Models."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "container_spec")

    @property
    def version_id(self) -> str:
        """The version ID of the model.
        A new version is committed when a new model version is uploaded or
        trained under an existing model id. It is an auto-incrementing decimal
        number in string representation."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "version_id")

    @property
    def version_aliases(self) -> Sequence[str]:
        """User provided version aliases so that a model version can be referenced via
        alias (i.e. projects/{project}/locations/{location}/models/{model_id}@{version_alias}
        instead of auto-generated version id (i.e.
        projects/{project}/locations/{location}/models/{model_id}@{version_id}).
        The format is [a-z][a-zA-Z0-9-]{0,126}[a-z0-9] to distinguish from
        version_id. A default version alias will be created for the first version
        of the model, and there must be exactly one default version alias for a model.
        """
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "version_aliases")

    @property
    def version_create_time(self) -> timestamp_pb2.Timestamp:
        """Timestamp when this version was created."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "version_create_time")

    @property
    def version_update_time(self) -> timestamp_pb2.Timestamp:
        """Timestamp when this version was updated."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "version_update_time")

    @property
    def version_description(self) -> str:
        """The description of this version."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "version_description")

    @property
    def resource_name(self) -> str:
        """Full qualified resource name, without any version ID."""
        self._assert_gca_resource_is_available()
        return ModelRegistry._parse_versioned_name(self._gca_resource.name)[0]

    @property
    def name(self) -> str:
        """Name of this resource."""
        self._assert_gca_resource_is_available()
        return ModelRegistry._parse_versioned_name(super().name)[0]

    @property
    def versioned_resource_name(self) -> str:
        """The fully-qualified resource name, including the version ID. For example,
        projects/{project}/locations/{location}/models/{model_id}@{version_id}
        """
        self._assert_gca_resource_is_available()
        return ModelRegistry._get_versioned_name(
            self.resource_name,
            self.version_id,
        )

    @property
    def versioning_registry(self) -> "ModelRegistry":
        """The registry of model versions associated with this
        Model instance."""
        return self._registry

    def __init__(
        self,
        model_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        version: Optional[str] = None,
    ):
        """Retrieves the model resource and instantiates its representation.

        Args:
            model_name (str):
                Required. A fully-qualified model resource name or model ID.
                Example: "projects/123/locations/us-central1/models/456" or
                "456" when project and location are initialized or passed.
                May optionally contain a version ID or version alias in
                {model_name}@{version} form. See version arg.
            project (str):
                Optional project to retrieve model from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional location to retrieve model from. If not set, location
                set in aiplatform.init will be used.
            credentials: Optional[auth_credentials.Credentials]=None,
                Custom credentials to use to upload this model. If not set,
                credentials set in aiplatform.init will be used.
            version (str):
                Optional. Version ID or version alias.
                When set, the specified model version will be targeted
                unless overridden in method calls.
                When not set, the model with the "default" alias will
                be targeted unless overridden in method calls.
                No behavior change if only one version of a model exists.
        Raises:
            ValueError: If `version` is passed alongside a model_name referencing a different version.
        """
        # If the version was passed in model_name, parse it
        model_name, parsed_version = ModelRegistry._parse_versioned_name(model_name)
        if parsed_version:
            if version and version != parsed_version:
                raise ValueError(
                    f"A version of {version} was passed that conflicts with the version of {parsed_version} in the model_name."
                )
            version = parsed_version

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=model_name,
        )

        # Model versions can include @{version} in the resource name.
        self._resource_id_validator = super()._revisioned_resource_id_validator

        # Create a versioned model_name, if it exists, for getting the GCA model
        versioned_model_name = ModelRegistry._get_versioned_name(model_name, version)
        self._gca_resource = self._get_gca_resource(resource_name=versioned_model_name)

        # Create ModelRegistry with the unversioned resource name
        self._registry = ModelRegistry(
            self.resource_name,
            location=location,
            project=project,
            credentials=credentials,
        )

    def update(
        self,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> "Model":
        """Updates a model.

        Example usage:
            my_model = my_model.update(
                display_name="my-model",
                description="my description",
                labels={'key': 'value'},
            )

        Args:
            display_name (str):
                The display name of the Model. The name can be up to 128
                characters long and can be consist of any UTF-8 characters.
            description (str):
                The description of the model.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Models.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.

        Returns:
            model (aiplatform.Model):
                Updated model resource.

        Raises:
            ValueError: If `labels` is not the correct format.
        """

        self.wait()

        current_model_proto = self.gca_resource
        copied_model_proto = current_model_proto.__class__(current_model_proto)

        update_mask: List[str] = []

        # Updates to base model properties cannot occur if a versioned model is passed.
        # Use the unversioned model resource name.
        copied_model_proto.name = self.resource_name

        if display_name:
            utils.validate_display_name(display_name)

            copied_model_proto.display_name = display_name
            update_mask.append("display_name")

        if description:
            copied_model_proto.description = description
            update_mask.append("description")

        if labels:
            utils.validate_labels(labels)

            copied_model_proto.labels = labels
            update_mask.append("labels")

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)

        self.api_client.update_model(model=copied_model_proto, update_mask=update_mask)

        self._sync_gca_resource()

        return self

    # TODO(b/170979926) Add support for metadata and metadata schema
    @classmethod
    @base.optional_sync()
    def upload(
        cls,
        serving_container_image_uri: Optional[str] = None,
        *,
        artifact_uri: Optional[str] = None,
        model_id: Optional[str] = None,
        parent_model: Optional[str] = None,
        is_default_version: bool = True,
        version_aliases: Optional[Sequence[str]] = None,
        version_description: Optional[str] = None,
        serving_container_predict_route: Optional[str] = None,
        serving_container_health_route: Optional[str] = None,
        description: Optional[str] = None,
        serving_container_command: Optional[Sequence[str]] = None,
        serving_container_args: Optional[Sequence[str]] = None,
        serving_container_environment_variables: Optional[Dict[str, str]] = None,
        serving_container_ports: Optional[Sequence[int]] = None,
        serving_container_grpc_ports: Optional[Sequence[int]] = None,
        local_model: Optional["LocalModel"] = None,
        instance_schema_uri: Optional[str] = None,
        parameters_schema_uri: Optional[str] = None,
        prediction_schema_uri: Optional[str] = None,
        explanation_metadata: Optional[explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[explain.ExplanationParameters] = None,
        display_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        sync=True,
        upload_request_timeout: Optional[float] = None,
        serving_container_deployment_timeout: Optional[int] = None,
        serving_container_shared_memory_size_mb: Optional[int] = None,
        serving_container_startup_probe_exec: Optional[Sequence[str]] = None,
        serving_container_startup_probe_period_seconds: Optional[int] = None,
        serving_container_startup_probe_timeout_seconds: Optional[int] = None,
        serving_container_health_probe_exec: Optional[Sequence[str]] = None,
        serving_container_health_probe_period_seconds: Optional[int] = None,
        serving_container_health_probe_timeout_seconds: Optional[int] = None,
    ) -> "Model":
        """Uploads a model and returns a Model representing the uploaded Model
        resource.

        Example usage:
            my_model = Model.upload(
                display_name="my-model",
                artifact_uri="gs://my-model/saved-model",
                serving_container_image_uri="tensorflow/serving"
            )

        Args:
            serving_container_image_uri (str):
                Optional. The URI of the Model serving container. This parameter is required
                if the parameter `local_model` is not specified.
            artifact_uri (str):
                Optional. The path to the directory containing the Model artifact and
                any of its supporting files. Leave blank for custom container prediction.
                Not present for AutoML Models.
            model_id (str):
                Optional. The ID to use for the uploaded Model, which will
                become the final component of the model resource name.
                This value may be up to 63 characters, and valid characters
                are `[a-z0-9_-]`. The first character cannot be a number or hyphen.
            parent_model (str):
                Optional. The resource name or model ID of an existing model that the
                newly-uploaded model will be a version of.

                Only set this field when uploading a new version of an existing model.
            is_default_version (bool):
                Optional. When set to True, the newly uploaded model version will
                automatically have alias "default" included. Subsequent uses of
                this model without a version specified will use this "default" version.

                When set to False, the "default" alias will not be moved.
                Actions targeting the newly-uploaded model version will need
                to specifically reference this version by ID or alias.

                New model uploads, i.e. version 1, will always be "default" aliased.
            version_aliases (Sequence[str]):
                Optional. User provided version aliases so that a model version
                can be referenced via alias instead of auto-generated version ID.
                A default version alias will be created for the first version of the model.

                The format is [a-z][a-zA-Z0-9-]{0,126}[a-z0-9]
            version_description (str):
                Optional. The description of the model version being uploaded.
            serving_container_predict_route (str):
                Optional. An HTTP path to send prediction requests to the container, and
                which must be supported by it. If not specified a default HTTP path will
                be used by Vertex AI.
            serving_container_health_route (str):
                Optional. An HTTP path to send health check requests to the container, and which
                must be supported by it. If not specified a standard HTTP path will be
                used by Vertex AI.
            description (str):
                The description of the model.
            serving_container_command: Optional[Sequence[str]]=None,
                The command with which the container is run. Not executed within a
                shell. The Docker image's ENTRYPOINT is used if this is not provided.
                Variable references $(VAR_NAME) are expanded using the container's
                environment. If a variable cannot be resolved, the reference in the
                input string will be unchanged. The $(VAR_NAME) syntax can be escaped
                with a double $$, ie: $$(VAR_NAME). Escaped references will never be
                expanded, regardless of whether the variable exists or not.
            serving_container_args: Optional[Sequence[str]]=None,
                The arguments to the command. The Docker image's CMD is used if this is
                not provided. Variable references $(VAR_NAME) are expanded using the
                container's environment. If a variable cannot be resolved, the reference
                in the input string will be unchanged. The $(VAR_NAME) syntax can be
                escaped with a double $$, ie: $$(VAR_NAME). Escaped references will
                never be expanded, regardless of whether the variable exists or not.
            serving_container_environment_variables: Optional[Dict[str, str]]=None,
                The environment variables that are to be present in the container.
                Should be a dictionary where keys are environment variable names
                and values are environment variable values for those names.
            serving_container_ports: Optional[Sequence[int]]=None,
                Declaration of ports that are exposed by the container. This field is
                primarily informational, it gives Vertex AI information about the
                network connections the container uses. Listing or not a port here has
                no impact on whether the port is actually exposed, any port listening on
                the default "0.0.0.0" address inside a container will be accessible from
                the network.
            serving_container_grpc_ports: Optional[Sequence[int]]=None,
                Declaration of ports that are exposed by the container. Vertex AI sends gRPC
                prediction requests that it receives to the first port on this list. Vertex
                AI also sends liveness and health checks to this port.
                If you do not specify this field, gRPC requests to the container will be
                disabled.
                Vertex AI does not use ports other than the first one listed. This field
                corresponds to the `ports` field of the Kubernetes Containers v1 core API.
            local_model (Optional[LocalModel]):
                Optional. A LocalModel instance that includes a `serving_container_spec`.
                If provided, the `serving_container_spec` of the LocalModel instance
                will overwrite the values of all other serving container parameters.
            instance_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single instance, which
                are used in
                ``PredictRequest.instances``,
                ``ExplainRequest.instances``
                and
                ``BatchPredictionJob.input_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            parameters_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the parameters of prediction and
                explanation via
                ``PredictRequest.parameters``,
                ``ExplainRequest.parameters``
                and
                ``BatchPredictionJob.model_parameters``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform, if no parameters are supported it is set to an
                empty string. Note: The URI given on output will be
                immutable and probably different, including the URI scheme,
                than the one given on input. The output URI will point to a
                location where the user only has a read access.
            prediction_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single prediction
                produced by this Model, which are returned via
                ``PredictResponse.predictions``,
                ``ExplainResponse.explanations``,
                and
                ``BatchPredictionJob.output_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Metadata describing the Model's input and output for explanation.
                `explanation_metadata` is optional while `explanation_parameters` must be
                specified when used.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            display_name (str):
                Optional. The display name of the Model. The name can be up to 128
                characters long and can be consist of any UTF-8 characters.
            project: Optional[str]=None,
                Project to upload this model to. Overrides project set in
                aiplatform.init.
            location: Optional[str]=None,
                Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials: Optional[auth_credentials.Credentials]=None,
                Custom credentials to use to upload this model. Overrides credentials
                set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Models.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            staging_bucket (str):
                Optional. Bucket to stage local model artifacts. Overrides
                staging_bucket set in aiplatform.init.
            upload_request_timeout (float):
                Optional. The timeout for the upload request in seconds.
            serving_container_deployment_timeout (int):
                Optional. Deployment timeout in seconds.
            serving_container_shared_memory_size_mb (int):
                Optional. The amount of the VM memory to reserve as the shared
                memory for the model in megabytes.
            serving_container_startup_probe_exec (Sequence[str]):
                Optional. Exec specifies the action to take. Used by startup
                probe. An example of this argument would be
                ["cat", "/tmp/healthy"]
            serving_container_startup_probe_period_seconds (int):
                Optional. How often (in seconds) to perform the startup probe.
                Default to 10 seconds. Minimum value is 1.
            serving_container_startup_probe_timeout_seconds (int):
                Optional. Number of seconds after which the startup probe times
                out. Defaults to 1 second. Minimum value is 1.
            serving_container_health_probe_exec (Sequence[str]):
                Optional. Exec specifies the action to take. Used by health
                probe. An example of this argument would be
                ["cat", "/tmp/healthy"]
            serving_container_health_probe_period_seconds (int):
                Optional. How often (in seconds) to perform the health probe.
                Default to 10 seconds. Minimum value is 1.
            serving_container_health_probe_timeout_seconds (int):
                Optional. Number of seconds after which the health probe times
                out. Defaults to 1 second. Minimum value is 1.

        Returns:
            model (aiplatform.Model):
                Instantiated representation of the uploaded model resource.

        Raises:
            ValueError: If explanation_metadata is specified while explanation_parameters
                is not.

                Also if model directory does not contain a supported model file.
                If `local_model` is specified but `serving_container_spec.image_uri`
                in the `local_model` is None.
                If `local_model` is not specified and `serving_container_image_uri`
                is None.
        """
        if not display_name:
            display_name = cls._generate_display_name()
        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)

        appended_user_agent = None
        if local_model:
            container_spec = local_model.get_serving_container_spec()
            appended_user_agent = [prediction_constants.CUSTOM_PREDICTION_ROUTINES]
        else:
            if not serving_container_image_uri:
                raise ValueError(
                    "The parameter `serving_container_image_uri` is required "
                    "if no `local_model` is provided."
                )

            env = None
            ports = None
            grpc_ports = None
            deployment_timeout = (
                duration_pb2.Duration(seconds=serving_container_deployment_timeout)
                if serving_container_deployment_timeout
                else None
            )
            startup_probe = None
            health_probe = None

            if serving_container_environment_variables:
                env = [
                    gca_env_var_compat.EnvVar(name=str(key), value=str(value))
                    for key, value in serving_container_environment_variables.items()
                ]
            if serving_container_ports:
                ports = [
                    gca_model_compat.Port(container_port=port)
                    for port in serving_container_ports
                ]
            if serving_container_grpc_ports:
                grpc_ports = [
                    gca_model_compat.Port(container_port=port)
                    for port in serving_container_grpc_ports
                ]
            if (
                serving_container_startup_probe_exec
                or serving_container_startup_probe_period_seconds
                or serving_container_startup_probe_timeout_seconds
            ):
                startup_probe_exec = None
                if serving_container_startup_probe_exec:
                    startup_probe_exec = gca_model_compat.Probe.ExecAction(
                        command=serving_container_startup_probe_exec
                    )
                startup_probe = gca_model_compat.Probe(
                    exec=startup_probe_exec,
                    period_seconds=serving_container_startup_probe_period_seconds,
                    timeout_seconds=serving_container_startup_probe_timeout_seconds,
                )
            if (
                serving_container_health_probe_exec
                or serving_container_health_probe_period_seconds
                or serving_container_health_probe_timeout_seconds
            ):
                health_probe_exec = None
                if serving_container_health_probe_exec:
                    health_probe_exec = gca_model_compat.Probe.ExecAction(
                        command=serving_container_health_probe_exec
                    )
                health_probe = gca_model_compat.Probe(
                    exec=health_probe_exec,
                    period_seconds=serving_container_health_probe_period_seconds,
                    timeout_seconds=serving_container_health_probe_timeout_seconds,
                )

            container_spec = gca_model_compat.ModelContainerSpec(
                image_uri=serving_container_image_uri,
                command=serving_container_command,
                args=serving_container_args,
                env=env,
                ports=ports,
                grpc_ports=grpc_ports,
                predict_route=serving_container_predict_route,
                health_route=serving_container_health_route,
                deployment_timeout=deployment_timeout,
                shared_memory_size_mb=serving_container_shared_memory_size_mb,
                startup_probe=startup_probe,
                health_probe=health_probe,
            )

        model_predict_schemata = None
        if any([instance_schema_uri, parameters_schema_uri, prediction_schema_uri]):
            model_predict_schemata = gca_model_compat.PredictSchemata(
                instance_schema_uri=instance_schema_uri,
                parameters_schema_uri=parameters_schema_uri,
                prediction_schema_uri=prediction_schema_uri,
            )

        # TODO(b/182388545) initializer.global_config.get_encryption_spec from a sync function
        encryption_spec = initializer.global_config.get_encryption_spec(
            encryption_spec_key_name=encryption_spec_key_name,
        )

        parent_model = ModelRegistry._get_true_version_parent(
            location=location, project=project, parent_model=parent_model
        )

        version_aliases = ModelRegistry._get_true_alias_list(
            version_aliases=version_aliases, is_default_version=is_default_version
        )

        managed_model = gca_model_compat.Model(
            display_name=display_name,
            description=description,
            version_aliases=version_aliases,
            version_description=version_description,
            container_spec=container_spec,
            predict_schemata=model_predict_schemata,
            labels=labels,
            encryption_spec=encryption_spec,
        )

        if artifact_uri and not artifact_uri.startswith("gs://"):
            model_dir = pathlib.Path(artifact_uri)
            # Validating the model directory
            if not model_dir.exists():
                raise ValueError(f"artifact_uri path does not exist: '{artifact_uri}'")
            PREBUILT_IMAGE_RE = "(us|europe|asia)-docker.pkg.dev/vertex-ai/prediction/"
            if re.match(PREBUILT_IMAGE_RE, serving_container_image_uri):
                if not model_dir.is_dir():
                    raise ValueError(
                        f"artifact_uri path must be a directory: '{artifact_uri}' when using prebuilt image '{serving_container_image_uri}'"
                    )
                if not any(
                    (model_dir / file_name).exists()
                    for file_name in _SUPPORTED_MODEL_FILE_NAMES
                ):
                    raise ValueError(
                        "artifact_uri directory does not contain any supported model files. "
                        f"When using a prebuilt serving image, the upload method only supports the following model files: '{_SUPPORTED_MODEL_FILE_NAMES}'"
                    )

            # Uploading the model
            staged_data_uri = gcs_utils.stage_local_data_in_gcs(
                data_path=str(model_dir),
                staging_gcs_dir=staging_bucket,
                project=project,
                location=location,
                credentials=credentials,
            )
            artifact_uri = staged_data_uri

        if artifact_uri:
            managed_model.artifact_uri = artifact_uri

        managed_model.explanation_spec = (
            _explanation_utils.create_and_validate_explanation_spec(
                explanation_metadata=explanation_metadata,
                explanation_parameters=explanation_parameters,
            )
        )

        request = gca_model_service_compat.UploadModelRequest(
            parent=initializer.global_config.common_location_path(project, location),
            model=managed_model,
            parent_model=parent_model,
            model_id=model_id,
        )

        api_client = cls._instantiate_client(
            location, credentials, appended_user_agent=appended_user_agent
        )

        lro = api_client.upload_model(
            request=request,
            timeout=upload_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, lro)

        model_upload_response = lro.result()

        this_model = cls(
            model_upload_response.model, version=model_upload_response.model_version_id
        )

        _LOGGER.log_create_complete(cls, this_model._gca_resource, "model")

        return this_model

    def deploy(
        self,
        endpoint: Optional[Union["Endpoint", "PrivateEndpoint"]] = None,
        deployed_model_display_name: Optional[str] = None,
        traffic_percentage: Optional[int] = 0,
        traffic_split: Optional[Dict[str, int]] = None,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        service_account: Optional[str] = None,
        explanation_metadata: Optional[aiplatform.explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[
            aiplatform.explain.ExplanationParameters
        ] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        encryption_spec_key_name: Optional[str] = None,
        network: Optional[str] = None,
        sync=True,
        deploy_request_timeout: Optional[float] = None,
        autoscaling_target_cpu_utilization: Optional[int] = None,
        autoscaling_target_accelerator_duty_cycle: Optional[int] = None,
        enable_access_logging=False,
        disable_container_logging: bool = False,
    ) -> Union[Endpoint, PrivateEndpoint]:
        """Deploys model to endpoint. Endpoint will be created if unspecified.

        Args:
            endpoint (Union[Endpoint, PrivateEndpoint]):
                Optional. Public or private Endpoint to deploy model to. If not specified,
                endpoint display name will be model display name+'_endpoint'.
            deployed_model_display_name (str):
                Optional. The display name of the DeployedModel. If not provided
                upon creation, the Model's display_name is used.
            traffic_percentage (int):
                Optional. Desired traffic to newly deployed model. Defaults to
                0 if there are pre-existing deployed models. Defaults to 100 if
                there are no pre-existing deployed models. Negative values should
                not be provided. Traffic of previously deployed models at the endpoint
                will be scaled down to accommodate new deployed model's traffic.
                Should not be provided if traffic_split is provided.
            traffic_split (Dict[str, int]):
                Optional. A map from a DeployedModel's ID to the percentage of
                this Endpoint's traffic that should be forwarded to that DeployedModel.
                If a DeployedModel's ID is not listed in this map, then it receives
                no traffic. The traffic percentage values must add up to 100, or
                map must be empty if the Endpoint is to not accept any traffic at
                the moment. Key for model being deployed is "0". Should not be
                provided if traffic_percentage is provided.
            machine_type (str):
                Optional. The type of machine. Not specifying machine type will
                result in model to be deployed with automatic resources.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the smaller value of min_replica_count or 1 will
                be used.
            accelerator_type (str):
                Optional. Hardware accelerator type. Must also set accelerator_count if used.
                One of ACCELERATOR_TYPE_UNSPECIFIED, NVIDIA_TESLA_K80, NVIDIA_TESLA_P100,
                NVIDIA_TESLA_V100, NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int):
                Optional. The number of accelerators to attach to a worker replica.
            service_account (str):
                The service account that the DeployedModel's container runs as. Specify the
                email address of the service account. If this service account is not
                specified, the container runs as a service account that doesn't have access
                to the resource project.
                Users deploying the Model must have the `iam.serviceAccounts.actAs`
                permission on this service account.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Metadata describing the Model's input and output for explanation.
                `explanation_metadata` is optional while `explanation_parameters` must be
                specified when used.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Endpoint and all sub-resources of this Endpoint will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            network (str):
                Optional. The full name of the Compute Engine network to which
                the Endpoint, if created, will be peered to. E.g. "projects/12345/global/networks/myVPC".
                Private services access must already be configured for the network.
                If set or aiplatform.init(network=...) has been set, a PrivateEndpoint will be created.
                If left unspecified, an Endpoint will be created. Read more about PrivateEndpoints
                [in the documentation](https://cloud.google.com/vertex-ai/docs/predictions/using-private-endpoints).
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            deploy_request_timeout (float):
                Optional. The timeout for the deploy request in seconds.
            autoscaling_target_cpu_utilization (int):
                Optional. Target CPU Utilization to use for Autoscaling Replicas.
                A default value of 60 will be used if not specified.
            autoscaling_target_accelerator_duty_cycle (int):
                Optional. Target Accelerator Duty Cycle.
                Must also set accelerator_type and accelerator_count if specified.
                A default value of 60 will be used if not specified.
            enable_access_logging (bool):
                Whether to enable endpoint access logging. Defaults to False.
            disable_container_logging (bool):
                If True, container logs from the deployed model will not be
                written to Cloud Logging. Defaults to False.

        Returns:
            endpoint (Union[Endpoint, PrivateEndpoint]):
                Endpoint with the deployed model.

        Raises:
            ValueError: If `traffic_split` is set for PrivateEndpoint.
        """
        network = network or initializer.global_config.network

        Endpoint._validate_deploy_args(
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            deployed_model_display_name=deployed_model_display_name,
            traffic_split=traffic_split,
            traffic_percentage=traffic_percentage,
        )

        if isinstance(endpoint, PrivateEndpoint):
            if traffic_split:
                raise ValueError(
                    "Traffic splitting is not yet supported for PrivateEndpoint. "
                    "Try calling deploy() without providing `traffic_split`. "
                    "A maximum of one model can be deployed to each private Endpoint."
                )

        explanation_spec = _explanation_utils.create_and_validate_explanation_spec(
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
        )

        return self._deploy(
            endpoint=endpoint,
            deployed_model_display_name=deployed_model_display_name,
            traffic_percentage=traffic_percentage,
            traffic_split=traffic_split,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            service_account=service_account,
            explanation_spec=explanation_spec,
            metadata=metadata,
            encryption_spec_key_name=encryption_spec_key_name
            or initializer.global_config.encryption_spec_key_name,
            network=network,
            sync=sync,
            deploy_request_timeout=deploy_request_timeout,
            autoscaling_target_cpu_utilization=autoscaling_target_cpu_utilization,
            autoscaling_target_accelerator_duty_cycle=autoscaling_target_accelerator_duty_cycle,
            enable_access_logging=enable_access_logging,
            disable_container_logging=disable_container_logging,
        )

    @base.optional_sync(return_input_arg="endpoint", bind_future_to_self=False)
    def _deploy(
        self,
        endpoint: Optional[Union["Endpoint", "PrivateEndpoint"]] = None,
        deployed_model_display_name: Optional[str] = None,
        traffic_percentage: Optional[int] = 0,
        traffic_split: Optional[Dict[str, int]] = None,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        service_account: Optional[str] = None,
        explanation_spec: Optional[aiplatform.explain.ExplanationSpec] = None,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        encryption_spec_key_name: Optional[str] = None,
        network: Optional[str] = None,
        sync: bool = True,
        deploy_request_timeout: Optional[float] = None,
        autoscaling_target_cpu_utilization: Optional[int] = None,
        autoscaling_target_accelerator_duty_cycle: Optional[int] = None,
        enable_access_logging=False,
        disable_container_logging: bool = False,
    ) -> Union[Endpoint, PrivateEndpoint]:
        """Deploys model to endpoint. Endpoint will be created if unspecified.

        Args:
            endpoint (Union[Endpoint, PrivateEndpoint]):
                Optional. Public or private Endpoint to deploy model to. If not specified,
                endpoint display name will be model display name+'_endpoint'.
            deployed_model_display_name (str):
                Optional. The display name of the DeployedModel. If not provided
                upon creation, the Model's display_name is used.
            traffic_percentage (int):
                Optional. Desired traffic to newly deployed model. Defaults to
                0 if there are pre-existing deployed models. Defaults to 100 if
                there are no pre-existing deployed models. Negative values should
                not be provided. Traffic of previously deployed models at the endpoint
                will be scaled down to accommodate new deployed model's traffic.
                Should not be provided if traffic_split is provided.
            traffic_split (Dict[str, int]):
                Optional. A map from a DeployedModel's ID to the percentage of
                this Endpoint's traffic that should be forwarded to that DeployedModel.
                If a DeployedModel's ID is not listed in this map, then it receives
                no traffic. The traffic percentage values must add up to 100, or
                map must be empty if the Endpoint is to not accept any traffic at
                the moment. Key for model being deployed is "0". Should not be
                provided if traffic_percentage is provided.
            machine_type (str):
                Optional. The type of machine. Not specifying machine type will
                result in model to be deployed with automatic resources.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the smaller value of min_replica_count or 1 will
                be used.
            accelerator_type (str):
                Optional. Hardware accelerator type. Must also set accelerator_count if used.
                One of ACCELERATOR_TYPE_UNSPECIFIED, NVIDIA_TESLA_K80, NVIDIA_TESLA_P100,
                NVIDIA_TESLA_V100, NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int):
                Optional. The number of accelerators to attach to a worker replica.
            service_account (str):
                The service account that the DeployedModel's container runs as. Specify the
                email address of the service account. If this service account is not
                specified, the container runs as a service account that doesn't have access
                to the resource project.
                Users deploying the Model must have the `iam.serviceAccounts.actAs`
                permission on this service account.
            explanation_spec (aiplatform.explain.ExplanationSpec):
                Optional. Specification of Model explanation.
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init
            network (str):
                Optional. The full name of the Compute Engine network to which
                the Endpoint, if created, will be peered to. E.g. "projects/12345/global/networks/myVPC".
                Private services access must already be configured for the network.
                Read more about PrivateEndpoints
                [in the documentation](https://cloud.google.com/vertex-ai/docs/predictions/using-private-endpoints).
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            deploy_request_timeout (float):
                Optional. The timeout for the deploy request in seconds.
            autoscaling_target_cpu_utilization (int):
                Optional. Target CPU Utilization to use for Autoscaling Replicas.
                A default value of 60 will be used if not specified.
            autoscaling_target_accelerator_duty_cycle (int):
                Optional. Target Accelerator Duty Cycle.
                Must also set accelerator_type and accelerator_count if specified.
                A default value of 60 will be used if not specified.
            enable_access_logging (bool):
                Whether to enable endpoint access logging. Defaults to False.
            disable_container_logging (bool):
                If True, container logs from the deployed model will not be
                written to Cloud Logging. Defaults to False.

        Returns:
            endpoint (Union[Endpoint, PrivateEndpoint]):
                Endpoint with the deployed model.
        """

        if endpoint is None:
            display_name = self.display_name[:118] + "_endpoint"

            if not network:
                endpoint = Endpoint.create(
                    display_name=display_name,
                    project=self.project,
                    location=self.location,
                    credentials=self.credentials,
                    encryption_spec_key_name=encryption_spec_key_name,
                )
            else:
                endpoint = PrivateEndpoint.create(
                    display_name=display_name,
                    network=network,
                    project=self.project,
                    location=self.location,
                    credentials=self.credentials,
                    encryption_spec_key_name=encryption_spec_key_name,
                )

        _LOGGER.log_action_start_against_resource("Deploying model to", "", endpoint)

        endpoint._deploy_call(
            endpoint.api_client,
            endpoint.resource_name,
            self,
            endpoint._gca_resource.traffic_split,
            network=network or endpoint.network,
            deployed_model_display_name=deployed_model_display_name,
            traffic_percentage=traffic_percentage,
            traffic_split=traffic_split,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            service_account=service_account,
            explanation_spec=explanation_spec,
            metadata=metadata,
            deploy_request_timeout=deploy_request_timeout,
            autoscaling_target_cpu_utilization=autoscaling_target_cpu_utilization,
            autoscaling_target_accelerator_duty_cycle=autoscaling_target_accelerator_duty_cycle,
            enable_access_logging=enable_access_logging,
            disable_container_logging=disable_container_logging,
        )

        _LOGGER.log_action_completed_against_resource("model", "deployed", endpoint)

        endpoint._sync_gca_resource()

        return endpoint

    def batch_predict(
        self,
        job_display_name: Optional[str] = None,
        gcs_source: Optional[Union[str, Sequence[str]]] = None,
        bigquery_source: Optional[str] = None,
        instances_format: str = "jsonl",
        gcs_destination_prefix: Optional[str] = None,
        bigquery_destination_prefix: Optional[str] = None,
        predictions_format: str = "jsonl",
        model_parameters: Optional[Dict] = None,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        starting_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
        generate_explanation: Optional[bool] = False,
        explanation_metadata: Optional[aiplatform.explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[
            aiplatform.explain.ExplanationParameters
        ] = None,
        labels: Optional[Dict[str, str]] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
        batch_size: Optional[int] = None,
        service_account: Optional[str] = None,
    ) -> jobs.BatchPredictionJob:
        """Creates a batch prediction job using this Model and outputs
        prediction results to the provided destination prefix in the specified
        `predictions_format`. One source and one destination prefix are
        required.

        Example usage:
            my_model.batch_predict(
                job_display_name="prediction-123",
                gcs_source="gs://example-bucket/instances.csv",
                instances_format="csv",
                bigquery_destination_prefix="projectId.bqDatasetId.bqTableId"
            )

        Args:
            job_display_name (str):
                Optional. The user-defined name of the BatchPredictionJob.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            gcs_source: Optional[Sequence[str]] = None
                Google Cloud Storage URI(-s) to your instances to run
                batch prediction on. They must match `instances_format`.
            bigquery_source: Optional[str] = None
                BigQuery URI to a table, up to 2000 characters long. For example:
                `bq://projectId.bqDatasetId.bqTableId`
            instances_format: str = "jsonl"
                The format in which instances are provided. Must be one
                of the formats listed in `Model.supported_input_storage_formats`.
                Default is "jsonl" when using `gcs_source`. If a `bigquery_source`
                is provided, this is overridden to "bigquery".
            gcs_destination_prefix: Optional[str] = None
                The Google Cloud Storage location of the directory where the
                output is to be written to. In the given directory a new
                directory is created. Its name is
                ``prediction-<model-display-name>-<job-create-time>``, where
                timestamp is in YYYY-MM-DDThh:mm:ss.sssZ ISO-8601 format.
                Inside of it files ``predictions_0001.<extension>``,
                ``predictions_0002.<extension>``, ...,
                ``predictions_N.<extension>`` are created where
                ``<extension>`` depends on chosen ``predictions_format``,
                and N may equal 0001 and depends on the total number of
                successfully predicted instances. If the Model has both
                ``instance`` and ``prediction`` schemata defined then each such
                file contains predictions as per the ``predictions_format``.
                If prediction for any instance failed (partially or
                completely), then an additional ``errors_0001.<extension>``,
                ``errors_0002.<extension>``,..., ``errors_N.<extension>``
                files are created (N depends on total number of failed
                predictions). These files contain the failed instances, as
                per their schema, followed by an additional ``error`` field
                which as value has ```google.rpc.Status`` <Status>`__
                containing only ``code`` and ``message`` fields.
            bigquery_destination_prefix: Optional[str] = None
                The BigQuery URI to a project or table, up to 2000 characters long.
                When only the project is specified, the Dataset and Table is created.
                When the full table reference is specified, the Dataset must exist and
                table must not exist. Accepted forms: ``bq://projectId`` or
                ``bq://projectId.bqDatasetId``. If no Dataset is specified,
                a new one is created with the name
                ``prediction_<model-display-name>_<job-create-time>``
                where the table name is made BigQuery-dataset-name compatible
                (for example, most special characters become underscores), and
                timestamp is in YYYY_MM_DDThh_mm_ss_sssZ "based on ISO-8601"
                format. In the dataset two tables will be created, ``predictions``,
                and ``errors``. If the Model has both ``instance`` and
                ``prediction`` schemata defined then the tables have columns as
                follows: The ``predictions`` table contains instances for which
                the prediction succeeded, it has columns as per a concatenation
                of the Model's instance and prediction schemata. The ``errors``
                table contains rows for which the prediction has failed, it has
                instance columns, as per the instance schema, followed by a single
                "errors" column, which as values has ```google.rpc.Status`` <Status>`__
                represented as a STRUCT, and containing only ``code`` and ``message``.
            predictions_format: str = "jsonl"
                Required. The format in which Vertex AI outputs the
                predictions, must be one of the formats specified in
                `Model.supported_output_storage_formats`.
                Default is "jsonl" when using `gcs_destination_prefix`. If a
                `bigquery_destination_prefix` is provided, this is overridden to
                "bigquery".
            model_parameters: Optional[Dict] = None
                Optional. The parameters that govern the predictions. The schema of
                the parameters may be specified via the Model's `parameters_schema_uri`.
            machine_type: Optional[str] = None
                Optional. The type of machine for running batch prediction on
                dedicated resources. Not specifying machine type will result in
                batch prediction job being run with automatic resources.
            accelerator_type: Optional[str] = None
                Optional. The type of accelerator(s) that may be attached
                to the machine as per `accelerator_count`. Only used if
                `machine_type` is set.
            accelerator_count: Optional[int] = None
                Optional. The number of accelerators to attach to the
                `machine_type`. Only used if `machine_type` is set.
            starting_replica_count: Optional[int] = None
                The number of machine replicas used at the start of the batch
                operation. If not set, Vertex AI decides starting number, not
                greater than `max_replica_count`. Only used if `machine_type` is
                set.
            max_replica_count: Optional[int] = None
                The maximum number of machine replicas the batch operation may
                be scaled to. Only used if `machine_type` is set.
                Default is 10.
            generate_explanation (bool):
                Optional. Generate explanation along with the batch prediction
                results. This will cause the batch prediction output to include
                explanations based on the `prediction_format`:
                    - `bigquery`: output includes a column named `explanation`. The value
                        is a struct that conforms to the [aiplatform.gapic.Explanation] object.
                    - `jsonl`: The JSON objects on each line include an additional entry
                        keyed `explanation`. The value of the entry is a JSON object that
                        conforms to the [aiplatform.gapic.Explanation] object.
                    - `csv`: Generating explanations for CSV format is not supported.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Explanation metadata configuration for this BatchPredictionJob.
                Can be specified only if `generate_explanation` is set to `True`.

                This value overrides the value of `Model.explanation_metadata`.
                All fields of `explanation_metadata` are optional in the request. If
                a field of the `explanation_metadata` object is not populated, the
                corresponding field of the `Model.explanation_metadata` object is inherited.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                Can be specified only if `generate_explanation` is set to `True`.

                This value overrides the value of `Model.explanation_parameters`.
                All fields of `explanation_parameters` are optional in the request. If
                a field of the `explanation_parameters` object is not populated, the
                corresponding field of the `Model.explanation_parameters` object is inherited.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            labels: Optional[Dict[str, str]] = None
                Optional. The labels with user-defined metadata to organize your
                BatchPredictionJobs. Label keys and values can be no longer than
                64 characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. See https://goo.gl/xmQnxf
                for more information and examples of labels.
            credentials: Optional[auth_credentials.Credentials] = None
                Optional. Custom credentials to use to create this batch prediction
                job. Overrides credentials set in aiplatform.init.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            batch_size (int):
                Optional. The number of the records (e.g. instances) of the operation given in each batch
                to a machine replica. Machine type, and size of a single record should be considered
                when setting this parameter, higher value speeds up the batch operation's execution,
                but too high value will result in a whole batch not fitting in a machine's memory,
                and the whole operation will fail.
                The default value is 64.
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.

        Returns:
            job (jobs.BatchPredictionJob):
                Instantiated representation of the created batch prediction job.
        """

        return jobs.BatchPredictionJob.create(
            job_display_name=job_display_name,
            model_name=self,
            instances_format=instances_format,
            predictions_format=predictions_format,
            gcs_source=gcs_source,
            bigquery_source=bigquery_source,
            gcs_destination_prefix=gcs_destination_prefix,
            bigquery_destination_prefix=bigquery_destination_prefix,
            model_parameters=model_parameters,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            starting_replica_count=starting_replica_count,
            max_replica_count=max_replica_count,
            batch_size=batch_size,
            generate_explanation=generate_explanation,
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
            labels=labels,
            project=self.project,
            location=self.location,
            credentials=credentials or self.credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            sync=sync,
            create_request_timeout=create_request_timeout,
            service_account=service_account,
        )

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["models.Model"]:
        """List all Model resource instances.

        Example Usage:
            aiplatform.Model.list(
                filter='labels.my_label="my_label_value" AND display_name="my_model"',
            )

        Args:
            filter (str):
                Optional. An expression for filtering the results of the request.
                For field names both snake_case and camelCase are supported.
            order_by (str):
                Optional. A comma-separated list of fields to order by, sorted in
                ascending order. Use "desc" after a field name for descending.
                Supported fields: `display_name`, `create_time`, `update_time`
            project (str):
                Optional. Project to retrieve list from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve list from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve list. Overrides
                credentials set in aiplatform.init.

        Returns:
            List[models.Model]:
                A list of Model resource objects
        """

        return cls._list(
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
        )

    @classmethod
    def _construct_sdk_resource_from_gapic(
        cls,
        gapic_resource: gca_model_compat.Model,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "Model":
        """Override base._construct_sdk_resource_from_gapic to allow for setting
        a ModelRegistry and resource_id_validator.

        Args:
            gapic_resource (gca_model_compat.Model):
                A GAPIC representation of a Model resource.
            project (str):
                Optional. Project to construct SDK object from. If not set,
                project set in aiplatform.init will be used.
            location (str):
                Optional. Location to construct SDK object from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to construct SDK object.
                Overrides credentials set in aiplatform.init.

        Returns:
            Model:
                An initialized SDK Model object that represents the Model GAPIC type.
        """
        sdk_resource = super()._construct_sdk_resource_from_gapic(
            gapic_resource=gapic_resource,
            project=project,
            location=location,
            credentials=credentials,
        )
        sdk_resource._resource_id_validator = super()._revisioned_resource_id_validator

        sdk_resource._registry = ModelRegistry(
            sdk_resource.resource_name,
            location=location,
            project=project,
            credentials=credentials,
        )

        return sdk_resource

    @base.optional_sync()
    def _wait_on_export(self, operation_future: operation.Operation, sync=True) -> None:
        operation_future.result()

    def export_model(
        self,
        export_format_id: str,
        artifact_destination: Optional[str] = None,
        image_destination: Optional[str] = None,
        sync: bool = True,
    ) -> Dict[str, str]:
        """Exports a trained, exportable Model to a location specified by the user.
        A Model is considered to be exportable if it has at least one `supported_export_formats`.
        Either `artifact_destination` or `image_destination` must be provided.

        Example Usage:
            my_model.export(
                export_format_id="tf-saved-model",
                artifact_destination="gs://my-bucket/models/"
            )

            or

            my_model.export(
                export_format_id="custom-model",
                image_destination="us-central1-docker.pkg.dev/projectId/repo/image"
            )

        Args:
            export_format_id (str):
                Required. The ID of the format in which the Model must be exported.
                The list of export formats that this Model supports can be found
                by calling `Model.supported_export_formats`.
            artifact_destination (str):
                The Cloud Storage location where the Model artifact is to be
                written to. Under the directory given as the destination a
                new one with name
                "``model-export-<model-display-name>-<timestamp-of-export-call>``",
                where timestamp is in YYYY-MM-DDThh:mm:ss.sssZ ISO-8601
                format, will be created. Inside, the Model and any of its
                supporting files will be written.

                This field should only be set when, in [Model.supported_export_formats],
                the value for the key given in `export_format_id` contains ``ARTIFACT``.
            image_destination (str):
                The Google Container Registry or Artifact Registry URI where
                the Model container image will be copied to. Accepted forms:

                -  Google Container Registry path. For example:
                ``gcr.io/projectId/imageName:tag``.

                -  Artifact Registry path. For example:
                ``us-central1-docker.pkg.dev/projectId/repoName/imageName:tag``.

                This field should only be set when, in [Model.supported_export_formats],
                the value for the key given in `export_format_id` contains ``IMAGE``.
            sync (bool):
                Whether to execute this export synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.

        Returns:
            output_info (Dict[str, str]):
                Details of the completed export with output destination paths to
                the artifacts or container image.

        Raises:
            ValueError: If model does not support exporting.
            ValueError: If invalid arguments or export formats are provided.
        """

        self.wait()

        # Model does not support exporting
        if not self.supported_export_formats:
            raise ValueError(f"The model `{self.resource_name}` is not exportable.")

        # No destination provided
        if not any((artifact_destination, image_destination)):
            raise ValueError(
                "Please provide an `artifact_destination` or `image_destination`."
            )

        export_format_id = export_format_id.lower()

        # Unsupported export type
        if export_format_id not in self.supported_export_formats:
            raise ValueError(
                f"'{export_format_id}' is not a supported export format for this model. "
                f"Choose one of the following: {self.supported_export_formats}"
            )

        content_types = gca_model_compat.Model.ExportFormat.ExportableContent
        supported_content_types = self.supported_export_formats[export_format_id]

        if (
            artifact_destination
            and content_types.ARTIFACT not in supported_content_types
        ):
            raise ValueError(
                "This model can not be exported as an artifact in '{export_format_id}' format. "
                "Try exporting as a container image by passing the `image_destination` argument."
            )

        if image_destination and content_types.IMAGE not in supported_content_types:
            raise ValueError(
                "This model can not be exported as a container image in '{export_format_id}' format. "
                "Try exporting the model artifacts by passing a `artifact_destination` argument."
            )

        # Construct request payload
        output_config = gca_model_service_compat.ExportModelRequest.OutputConfig(
            export_format_id=export_format_id
        )

        if artifact_destination:
            output_config.artifact_destination = gca_io_compat.GcsDestination(
                output_uri_prefix=artifact_destination
            )

        if image_destination:
            output_config.image_destination = (
                gca_io_compat.ContainerRegistryDestination(output_uri=image_destination)
            )

        _LOGGER.log_action_start_against_resource("Exporting", "model", self)

        model_name = self.versioned_resource_name

        operation_future = self.api_client.export_model(
            name=model_name, output_config=output_config
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Export", "model", self.__class__, operation_future
        )

        # Block before returning
        self._wait_on_export(operation_future=operation_future, sync=sync)

        _LOGGER.log_action_completed_against_resource("model", "exported", self)

        return json_format.MessageToDict(operation_future.metadata.output_info._pb)

    @classmethod
    @base.optional_sync()
    def upload_xgboost_model_file(
        cls,
        model_file_path: str,
        xgboost_version: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        model_id: Optional[str] = None,
        parent_model: Optional[str] = None,
        is_default_version: Optional[bool] = True,
        version_aliases: Optional[Sequence[str]] = None,
        version_description: Optional[str] = None,
        instance_schema_uri: Optional[str] = None,
        parameters_schema_uri: Optional[str] = None,
        prediction_schema_uri: Optional[str] = None,
        explanation_metadata: Optional[aiplatform.explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[
            aiplatform.explain.ExplanationParameters
        ] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        sync=True,
        upload_request_timeout: Optional[float] = None,
    ) -> "Model":
        """Uploads a model and returns a Model representing the uploaded Model
        resource.

        Example usage:
            my_model = Model.upload_xgboost_model_file(
                model_file_path="iris.xgboost_model.bst"
            )

        Args:
            model_file_path (str): Required. Local file path of the model.
            xgboost_version (str): Optional. The version of the XGBoost serving container.
                Supported versions: ["0.82", "0.90", "1.1", "1.2", "1.3", "1.4"].
                If the version is not specified, the latest version is used.
            display_name (str):
                Optional. The display name of the Model. The name can be up to 128
                characters long and can be consist of any UTF-8 characters.
            description (str):
                The description of the model.
            model_id (str):
                Optional. The ID to use for the uploaded Model, which will
                become the final component of the model resource name.
                This value may be up to 63 characters, and valid characters
                are `[a-z0-9_-]`. The first character cannot be a number or hyphen.
            parent_model (str):
                Optional. The resource name or model ID of an existing model that the
                newly-uploaded model will be a version of.

                Only set this field when uploading a new version of an existing model.
            is_default_version (bool):
                Optional. When set to True, the newly uploaded model version will
                automatically have alias "default" included. Subsequent uses of
                this model without a version specified will use this "default" version.

                When set to False, the "default" alias will not be moved.
                Actions targeting the newly-uploaded model version will need
                to specifically reference this version by ID or alias.

                New model uploads, i.e. version 1, will always be "default" aliased.
            version_aliases (Sequence[str]):
                Optional. User provided version aliases so that a model version
                can be referenced via alias instead of auto-generated version ID.
                A default version alias will be created for the first version of the model.

                The format is [a-z][a-zA-Z0-9-]{0,126}[a-z0-9]
            version_description (str):
                Optional. The description of the model version being uploaded.
            instance_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single instance, which
                are used in
                ``PredictRequest.instances``,
                ``ExplainRequest.instances``
                and
                ``BatchPredictionJob.input_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            parameters_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the parameters of prediction and
                explanation via
                ``PredictRequest.parameters``,
                ``ExplainRequest.parameters``
                and
                ``BatchPredictionJob.model_parameters``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform, if no parameters are supported it is set to an
                empty string. Note: The URI given on output will be
                immutable and probably different, including the URI scheme,
                than the one given on input. The output URI will point to a
                location where the user only has a read access.
            prediction_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single prediction
                produced by this Model, which are returned via
                ``PredictResponse.predictions``,
                ``ExplainResponse.explanations``,
                and
                ``BatchPredictionJob.output_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Metadata describing the Model's input and output for explanation.
                `explanation_metadata` is optional while `explanation_parameters` must be
                specified when used.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            project: Optional[str]=None,
                Project to upload this model to. Overrides project set in
                aiplatform.init.
            location: Optional[str]=None,
                Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials: Optional[auth_credentials.Credentials]=None,
                Custom credentials to use to upload this model. Overrides credentials
                set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Models.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            staging_bucket (str):
                Optional. Bucket to stage local model artifacts. Overrides
                staging_bucket set in aiplatform.init.
            upload_request_timeout (float):
                Optional. The timeout for the upload request in seconds.

        Returns:
            model (aiplatform.Model):
                Instantiated representation of the uploaded model resource.

        Raises:
            ValueError: If model directory does not contain a supported model file.
        """
        if not display_name:
            display_name = cls._generate_display_name("XGBoost model")

        XGBOOST_SUPPORTED_MODEL_FILE_EXTENSIONS = [
            ".pkl",
            ".joblib",
            ".bst",
        ]

        container_image_uri = aiplatform.helpers.get_prebuilt_prediction_container_uri(
            region=location,
            framework="xgboost",
            framework_version=xgboost_version or "1.4",
            accelerator="cpu",
        )

        model_file_path_obj = pathlib.Path(model_file_path)
        if not model_file_path_obj.is_file():
            raise ValueError(
                f"model_file_path path must point to a file: '{model_file_path}'"
            )

        model_file_extension = model_file_path_obj.suffix
        if model_file_extension not in XGBOOST_SUPPORTED_MODEL_FILE_EXTENSIONS:
            _LOGGER.warning(
                f"Only the following XGBoost model file extensions are currently supported: '{XGBOOST_SUPPORTED_MODEL_FILE_EXTENSIONS}'"
            )
            _LOGGER.warning(
                "Treating the model file as a binary serialized XGBoost Booster."
            )
            model_file_extension = ".bst"

        # Preparing model directory
        # We cannot clean up the directory immediately after calling Model.upload since
        # that call may be asynchronous and return before the model file has been read.
        # To work around this, we make this method asynchronous (decorate with @base.optional_sync)
        # but call Model.upload with sync=True.
        with tempfile.TemporaryDirectory() as prepared_model_dir:
            prepared_model_file_path = pathlib.Path(prepared_model_dir) / (
                "model" + model_file_extension
            )
            shutil.copy(model_file_path_obj, prepared_model_file_path)

            return cls.upload(
                serving_container_image_uri=container_image_uri,
                artifact_uri=prepared_model_dir,
                display_name=display_name,
                description=description,
                model_id=model_id,
                parent_model=parent_model,
                is_default_version=is_default_version,
                version_aliases=version_aliases,
                version_description=version_description,
                instance_schema_uri=instance_schema_uri,
                parameters_schema_uri=parameters_schema_uri,
                prediction_schema_uri=prediction_schema_uri,
                explanation_metadata=explanation_metadata,
                explanation_parameters=explanation_parameters,
                project=project,
                location=location,
                credentials=credentials,
                labels=labels,
                encryption_spec_key_name=encryption_spec_key_name,
                staging_bucket=staging_bucket,
                sync=True,
                upload_request_timeout=upload_request_timeout,
            )

    @classmethod
    @base.optional_sync()
    def upload_scikit_learn_model_file(
        cls,
        model_file_path: str,
        sklearn_version: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        model_id: Optional[str] = None,
        parent_model: Optional[str] = None,
        is_default_version: Optional[bool] = True,
        version_aliases: Optional[Sequence[str]] = None,
        version_description: Optional[str] = None,
        instance_schema_uri: Optional[str] = None,
        parameters_schema_uri: Optional[str] = None,
        prediction_schema_uri: Optional[str] = None,
        explanation_metadata: Optional[aiplatform.explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[
            aiplatform.explain.ExplanationParameters
        ] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        sync=True,
        upload_request_timeout: Optional[float] = None,
    ) -> "Model":
        """Uploads a model and returns a Model representing the uploaded Model
        resource.

        Example usage:
            my_model = Model.upload_scikit_learn_model_file(
                model_file_path="iris.sklearn_model.joblib"
            )

        Args:
            model_file_path (str): Required. Local file path of the model.
            sklearn_version (str):
                Optional. The version of the Scikit-learn serving container.
                Supported versions: ["0.20", "0.22", "0.23", "0.24", "1.0"].
                If the version is not specified, the latest version is used.
            display_name (str):
                Optional. The display name of the Model. The name can be up to 128
                characters long and can be consist of any UTF-8 characters.
            description (str):
                The description of the model.
            model_id (str):
                Optional. The ID to use for the uploaded Model, which will
                become the final component of the model resource name.
                This value may be up to 63 characters, and valid characters
                are `[a-z0-9_-]`. The first character cannot be a number or hyphen.
            parent_model (str):
                Optional. The resource name or model ID of an existing model that the
                newly-uploaded model will be a version of.

                Only set this field when uploading a new version of an existing model.
            is_default_version (bool):
                Optional. When set to True, the newly uploaded model version will
                automatically have alias "default" included. Subsequent uses of
                this model without a version specified will use this "default" version.

                When set to False, the "default" alias will not be moved.
                Actions targeting the newly-uploaded model version will need
                to specifically reference this version by ID or alias.

                New model uploads, i.e. version 1, will always be "default" aliased.
            version_aliases (Sequence[str]):
                Optional. User provided version aliases so that a model version
                can be referenced via alias instead of auto-generated version ID.
                A default version alias will be created for the first version of the model.

                The format is [a-z][a-zA-Z0-9-]{0,126}[a-z0-9]
            version_description (str):
                Optional. The description of the model version being uploaded.
            instance_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single instance, which
                are used in
                ``PredictRequest.instances``,
                ``ExplainRequest.instances``
                and
                ``BatchPredictionJob.input_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            parameters_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the parameters of prediction and
                explanation via
                ``PredictRequest.parameters``,
                ``ExplainRequest.parameters``
                and
                ``BatchPredictionJob.model_parameters``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform, if no parameters are supported it is set to an
                empty string. Note: The URI given on output will be
                immutable and probably different, including the URI scheme,
                than the one given on input. The output URI will point to a
                location where the user only has a read access.
            prediction_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single prediction
                produced by this Model, which are returned via
                ``PredictResponse.predictions``,
                ``ExplainResponse.explanations``,
                and
                ``BatchPredictionJob.output_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Metadata describing the Model's input and output for explanation.
                `explanation_metadata` is optional while `explanation_parameters` must be
                specified when used.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            project: Optional[str]=None,
                Project to upload this model to. Overrides project set in
                aiplatform.init.
            location: Optional[str]=None,
                Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials: Optional[auth_credentials.Credentials]=None,
                Custom credentials to use to upload this model. Overrides credentials
                set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Models.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            staging_bucket (str):
                Optional. Bucket to stage local model artifacts. Overrides
                staging_bucket set in aiplatform.init.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            upload_request_timeout (float):
                Optional. The timeout for the upload request in seconds.

        Returns:
            model (aiplatform.Model):
                Instantiated representation of the uploaded model resource.

        Raises:
            ValueError: If explanation_metadata is specified while explanation_parameters
                is not. Also if model directory does not contain a supported model file.
        """
        if not display_name:
            display_name = cls._generate_display_name("Scikit-Learn model")

        SKLEARN_SUPPORTED_MODEL_FILE_EXTENSIONS = [
            ".pkl",
            ".joblib",
        ]

        container_image_uri = aiplatform.helpers.get_prebuilt_prediction_container_uri(
            region=location,
            framework="sklearn",
            framework_version=sklearn_version or "1.0",
            accelerator="cpu",
        )

        model_file_path_obj = pathlib.Path(model_file_path)
        if not model_file_path_obj.is_file():
            raise ValueError(
                f"model_file_path path must point to a file: '{model_file_path}'"
            )

        model_file_extension = model_file_path_obj.suffix
        if model_file_extension not in SKLEARN_SUPPORTED_MODEL_FILE_EXTENSIONS:
            _LOGGER.warning(
                f"Only the following Scikit-learn model file extensions are currently supported: '{SKLEARN_SUPPORTED_MODEL_FILE_EXTENSIONS}'"
            )
            _LOGGER.warning(
                "Treating the model file as a pickle serialized Scikit-learn model."
            )
            model_file_extension = ".pkl"

        # Preparing model directory
        # We cannot clean up the directory immediately after calling Model.upload since
        # that call may be asynchronous and return before the model file has been read.
        # To work around this, we make this method asynchronous (decorate with @base.optional_sync)
        # but call Model.upload with sync=True.
        with tempfile.TemporaryDirectory() as prepared_model_dir:
            prepared_model_file_path = pathlib.Path(prepared_model_dir) / (
                "model" + model_file_extension
            )
            shutil.copy(model_file_path_obj, prepared_model_file_path)

            return cls.upload(
                serving_container_image_uri=container_image_uri,
                artifact_uri=prepared_model_dir,
                display_name=display_name,
                description=description,
                model_id=model_id,
                parent_model=parent_model,
                is_default_version=is_default_version,
                version_aliases=version_aliases,
                version_description=version_description,
                instance_schema_uri=instance_schema_uri,
                parameters_schema_uri=parameters_schema_uri,
                prediction_schema_uri=prediction_schema_uri,
                explanation_metadata=explanation_metadata,
                explanation_parameters=explanation_parameters,
                project=project,
                location=location,
                credentials=credentials,
                labels=labels,
                encryption_spec_key_name=encryption_spec_key_name,
                staging_bucket=staging_bucket,
                sync=True,
                upload_request_timeout=upload_request_timeout,
            )

    @classmethod
    def upload_tensorflow_saved_model(
        cls,
        saved_model_dir: str,
        tensorflow_version: Optional[str] = None,
        use_gpu: bool = False,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        model_id: Optional[str] = None,
        parent_model: Optional[str] = None,
        is_default_version: Optional[bool] = True,
        version_aliases: Optional[Sequence[str]] = None,
        version_description: Optional[str] = None,
        instance_schema_uri: Optional[str] = None,
        parameters_schema_uri: Optional[str] = None,
        prediction_schema_uri: Optional[str] = None,
        explanation_metadata: Optional[aiplatform.explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[
            aiplatform.explain.ExplanationParameters
        ] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        sync=True,
        upload_request_timeout: Optional[str] = None,
    ) -> "Model":
        """Uploads a model and returns a Model representing the uploaded Model
        resource.

        Example usage:
            my_model = Model.upload_scikit_learn_model_file(
                model_file_path="iris.tensorflow_model.SavedModel"
            )

        Args:
            saved_model_dir (str): Required.
                Local directory of the Tensorflow SavedModel.
            tensorflow_version (str):
                Optional. The version of the Tensorflow serving container.
                Supported versions: ["0.15", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"].
                If the version is not specified, the latest version is used.
            use_gpu (bool): Whether to use GPU for model serving.
            display_name (str):
                Optional. The display name of the Model. The name can be up to 128
                characters long and can be consist of any UTF-8 characters.
            description (str):
                The description of the model.
            model_id (str):
                Optional. The ID to use for the uploaded Model, which will
                become the final component of the model resource name.
                This value may be up to 63 characters, and valid characters
                are `[a-z0-9_-]`. The first character cannot be a number or hyphen.
            parent_model (str):
                Optional. The resource name or model ID of an existing model that the
                newly-uploaded model will be a version of.

                Only set this field when uploading a new version of an existing model.
            is_default_version (bool):
                Optional. When set to True, the newly uploaded model version will
                automatically have alias "default" included. Subsequent uses of
                this model without a version specified will use this "default" version.

                When set to False, the "default" alias will not be moved.
                Actions targeting the newly-uploaded model version will need
                to specifically reference this version by ID or alias.

                New model uploads, i.e. version 1, will always be "default" aliased.
            version_aliases (Sequence[str]):
                Optional. User provided version aliases so that a model version
                can be referenced via alias instead of auto-generated version ID.
                A default version alias will be created for the first version of the model.

                The format is [a-z][a-zA-Z0-9-]{0,126}[a-z0-9]
            version_description (str):
                Optional. The description of the model version being uploaded.
            instance_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single instance, which
                are used in
                ``PredictRequest.instances``,
                ``ExplainRequest.instances``
                and
                ``BatchPredictionJob.input_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            parameters_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the parameters of prediction and
                explanation via
                ``PredictRequest.parameters``,
                ``ExplainRequest.parameters``
                and
                ``BatchPredictionJob.model_parameters``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform, if no parameters are supported it is set to an
                empty string. Note: The URI given on output will be
                immutable and probably different, including the URI scheme,
                than the one given on input. The output URI will point to a
                location where the user only has a read access.
            prediction_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single prediction
                produced by this Model, which are returned via
                ``PredictResponse.predictions``,
                ``ExplainResponse.explanations``,
                and
                ``BatchPredictionJob.output_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Metadata describing the Model's input and output for explanation.
                `explanation_metadata` is optional while `explanation_parameters` must be
                specified when used.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            project: Optional[str]=None,
                Project to upload this model to. Overrides project set in
                aiplatform.init.
            location: Optional[str]=None,
                Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials: Optional[auth_credentials.Credentials]=None,
                Custom credentials to use to upload this model. Overrides credentials
                set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Models.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            staging_bucket (str):
                Optional. Bucket to stage local model artifacts. Overrides
                staging_bucket set in aiplatform.init.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            upload_request_timeout (float):
                Optional. The timeout for the upload request in seconds.

        Returns:
            model (aiplatform.Model):
                Instantiated representation of the uploaded model resource.

        Raises:
            ValueError: If explanation_metadata is specified while explanation_parameters
                is not. Also if model directory does not contain a supported model file.
        """
        if not display_name:
            display_name = cls._generate_display_name("Tensorflow model")

        container_image_uri = aiplatform.helpers.get_prebuilt_prediction_container_uri(
            region=location,
            framework="tensorflow",
            framework_version=tensorflow_version or "2.7",
            accelerator="gpu" if use_gpu else "cpu",
        )

        return cls.upload(
            serving_container_image_uri=container_image_uri,
            artifact_uri=saved_model_dir,
            display_name=display_name,
            description=description,
            model_id=model_id,
            parent_model=parent_model,
            is_default_version=is_default_version,
            version_aliases=version_aliases,
            version_description=version_description,
            instance_schema_uri=instance_schema_uri,
            parameters_schema_uri=parameters_schema_uri,
            prediction_schema_uri=prediction_schema_uri,
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
            project=project,
            location=location,
            credentials=credentials,
            labels=labels,
            encryption_spec_key_name=encryption_spec_key_name,
            staging_bucket=staging_bucket,
            sync=sync,
            upload_request_timeout=upload_request_timeout,
        )

    # TODO(b/273499620): Add async support.
    def copy(
        self,
        destination_location: str,
        destination_model_id: Optional[str] = None,
        destination_parent_model: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
        copy_request_timeout: Optional[float] = None,
    ) -> "Model":
        """Copys a model and returns a Model representing the copied Model
        resource. This method is a blocking call.

        Example usage:
            copied_model = my_model.copy(
                destination_location="us-central1"
            )

        Args:
            destination_location (str):
                The destination location to copy the model to.
            destination_model_id (str):
                Optional. The ID to use for the copied Model, which will
                become the final component of the model resource name.
                This value may be up to 63 characters, and valid characters
                are `[a-z0-9_-]`. The first character cannot be a number or hyphen.

                Only set this field when copying as a new model. If this field is not set,
                a numeric model id will be generated.
            destination_parent_model (str):
                Optional. The resource name or model ID of an existing model that the
                newly-copied model will be a version of.

                Only set this field when copying as a new version of an existing model.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            copy_request_timeout (float):
                Optional. The timeout for the copy request in seconds.

        Returns:
            model (aiplatform.Model):
                Instantiated representation of the copied model resource.

        Raises:
            ValueError: If both `destination_model_id` and `destination_parent_model` are set.
        """
        if destination_model_id is not None and destination_parent_model is not None:
            raise ValueError(
                "`destination_model_id` and `destination_parent_model` can not be set together."
            )

        parent = initializer.global_config.common_location_path(
            initializer.global_config.project, destination_location
        )

        source_model = self.versioned_resource_name

        destination_parent_model = ModelRegistry._get_true_version_parent(
            parent_model=destination_parent_model,
            project=initializer.global_config.project,
            location=destination_location,
        )

        encryption_spec = initializer.global_config.get_encryption_spec(
            encryption_spec_key_name=encryption_spec_key_name,
        )

        if destination_model_id is not None:
            request = gca_model_service_compat.CopyModelRequest(
                parent=parent,
                source_model=source_model,
                model_id=destination_model_id,
                encryption_spec=encryption_spec,
            )
        else:
            request = gca_model_service_compat.CopyModelRequest(
                parent=parent,
                source_model=source_model,
                parent_model=destination_parent_model,
                encryption_spec=encryption_spec,
            )

        api_client = initializer.global_config.create_client(
            client_class=utils.ModelClientWithOverride,
            location_override=destination_location,
            credentials=initializer.global_config.credentials,
        )

        _LOGGER.log_action_start_against_resource("Copying", "", self)

        lro = api_client.copy_model(
            request=request,
            timeout=copy_request_timeout,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Copy", "", self.__class__, lro
        )

        model_copy_response = lro.result(timeout=None)

        this_model = models.Model(
            model_copy_response.model,
            version=model_copy_response.model_version_id,
            location=destination_location,
        )

        _LOGGER.log_action_completed_against_resource("", "copied", this_model)

        return this_model

    def list_model_evaluations(
        self,
    ) -> List["model_evaluation.ModelEvaluation"]:
        """List all Model Evaluation resources associated with this model.
        If this Model resource was instantiated with a version, the Model
        Evaluation resources for that version will be returned. If no version
        was provided when the Model resource was instantiated, Model Evaluation
        resources will be returned for the default version.

        Example Usage:
            my_model = Model(
                model_name="projects/123/locations/us-central1/models/456@1"
            )

            my_evaluations = my_model.list_model_evaluations()

        Returns:
            List[model_evaluation.ModelEvaluation]:
                List of ModelEvaluation resources for the model.
        """

        return model_evaluation.ModelEvaluation._list(
            parent=self.versioned_resource_name,
            credentials=self.credentials,
        )

    def get_model_evaluation(
        self,
        evaluation_id: Optional[str] = None,
    ) -> Optional[model_evaluation.ModelEvaluation]:
        """Returns a ModelEvaluation resource and instantiates its representation.
        If no evaluation_id is passed, it will return the first evaluation associated
        with this model. If the aiplatform.Model resource was instantiated with a
        version, this will return a Model Evaluation from that version. If no version
        was specified when instantiating the Model resource, this will return an
        Evaluation from the default version.

        Example usage:
            my_model = Model(
                model_name="projects/123/locations/us-central1/models/456"
            )

            my_evaluation = my_model.get_model_evaluation(
                evaluation_id="789"
            )

            # If no arguments are passed, this method returns the first evaluation for the model
            my_evaluation = my_model.get_model_evaluation()

        Args:
            evaluation_id (str):
                Optional. The ID of the model evaluation to retrieve.

        Returns:
            model_evaluation.ModelEvaluation:
                Instantiated representation of the ModelEvaluation resource.
        """

        evaluations = self.list_model_evaluations()

        if not evaluation_id:
            if len(evaluations) > 1:
                _LOGGER.warning(
                    f"Your model has more than one model evaluation, this is returning only one evaluation resource: {evaluations[0].resource_name}"
                )
            _ipython_utils.display_model_evaluation_button(evaluations[0])
            return evaluations[0]
        else:
            resource_uri_parts = self._parse_resource_name(self.resource_name)
            evaluation_resource_name = (
                model_evaluation.ModelEvaluation._format_resource_name(
                    **resource_uri_parts,
                    evaluation=evaluation_id,
                )
            )

            evaluation = model_evaluation.ModelEvaluation(
                evaluation_name=evaluation_resource_name,
                credentials=self.credentials,
            )
            _ipython_utils.display_model_evaluation_button(evaluation)
            return evaluation

    def evaluate(
        self,
        prediction_type: str,
        target_field_name: str,
        gcs_source_uris: Optional[List[str]] = None,
        bigquery_source_uri: Optional[str] = None,
        bigquery_destination_output_uri: Optional[str] = None,
        class_labels: Optional[List[str]] = None,
        prediction_label_column: Optional[str] = None,
        prediction_score_column: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        service_account: Optional[str] = None,
        generate_feature_attributions: bool = False,
        evaluation_pipeline_display_name: Optional[str] = None,
        evaluation_metrics_display_name: Optional[str] = None,
        network: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
        experiment: Optional[Union[str, "aiplatform.Experiment"]] = None,
    ) -> "model_evaluation._ModelEvaluationJob":
        """Creates a model evaluation job running on Vertex Pipelines and returns the resulting
        ModelEvaluationJob resource.

        Example usage:

            ```
            my_model = Model(
                model_name="projects/123/locations/us-central1/models/456"
            )
            my_evaluation_job = my_model.evaluate(
                prediction_type="classification",
                target_field_name="type",
                data_source_uris=["gs://sdk-model-eval/my-prediction-data.csv"],
                staging_bucket="gs://my-staging-bucket/eval_pipeline_root",
            )
            my_evaluation_job.wait()
            my_evaluation = my_evaluation_job.get_model_evaluation()
            my_evaluation.metrics
            ```

        Args:
            prediction_type (str):
                Required. The problem type being addressed by this evaluation run. 'classification' and 'regression'
                are the currently supported problem types.
            target_field_name (str):
                Required. The column name of the field containing the label for this prediction task.
            gcs_source_uris (List[str]):
                Optional. A list of Cloud Storage data files containing the ground truth data to use for this
                evaluation job. These files should contain your model's prediction column. Currently only Google Cloud Storage
                urls are supported, for example: "gs://path/to/your/data.csv". The provided data files must be
                either CSV or JSONL. One of `gcs_source_uris` or `bigquery_source_uri` is required.
            bigquery_source_uri (str):
                Optional. A bigquery table URI containing the ground truth data to use for this evaluation job. This uri should
                be in the format 'bq://my-project-id.dataset.table'. One of `gcs_source_uris` or `bigquery_source_uri` is
                required.
            bigquery_destination_output_uri (str):
                Optional. A bigquery table URI where the Batch Prediction job associated with your Model Evaluation will write
                prediction output. This can be a BigQuery URI to a project ('bq://my-project'), a dataset
                ('bq://my-project.my-dataset'), or a table ('bq://my-project.my-dataset.my-table'). Required if `bigquery_source_uri`
                is provided.
            class_labels (List[str]):
                Optional. For custom (non-AutoML) classification models, a list of possible class names, in the
                same order that predictions are generated. This argument is required when prediction_type is 'classification'.
                For example, in a classification model with 3 possible classes that are outputted in the format: [0.97, 0.02, 0.01]
                with the class names "cat", "dog", and "fish", the value of `class_labels` should be `["cat", "dog", "fish"]` where
                the class "cat" corresponds with 0.97 in the example above.
            prediction_label_column (str):
                Optional. The column name of the field containing classes the model is scoring. Formatted to be able to find nested
                columns, delimeted by `.`. If not set, defaulted to `prediction.classes` for classification.
            prediction_score_column (str):
                Optional. The column name of the field containing batch prediction scores. Formatted to be able to find nested columns,
                delimeted by `.`. If not set, defaulted to `prediction.scores` for a `classification` problem_type, `prediction.value`
                for a `regression` problem_type.
            staging_bucket (str):
                Optional. The GCS directory to use for staging files from this evaluation job. Defaults to the value set in
                aiplatform.init(staging_bucket=...) if not provided. Required if staging_bucket is not set in aiplatform.init().
            service_account (str):
                Specifies the service account for workload run-as account for this Model Evaluation PipelineJob.
                Users submitting jobs must have act-as permission on this run-as account. The service account running
                this Model Evaluation job needs the following permissions: Dataflow Worker, Storage Admin,
                Vertex AI Administrator, and Vertex AI Service Agent.
            generate_feature_attributions (boolean):
                Optional. Whether the model evaluation job should generate feature attributions. Defaults to False if not specified.
            evaluation_pipeline_display_name (str):
                Optional. The display name of your model evaluation job. This is the display name that will be applied to the
                Vertex Pipeline run for your evaluation job. If not set, a display name will be generated automatically.
            evaluation_metrics_display_name (str):
                Optional. The display name of the model evaluation resource uploaded to Vertex from your Model Evaluation pipeline.
            network (str):
                The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the job is not peered with any network.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer managed encryption key used to protect the job. Has the
                form: ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``. The key needs to be in the same
                region as where the compute resource is created. If this is set, then all
                resources created by the PipelineJob for this Model Evaluation will be encrypted with the provided encryption key.
                If not specified, encryption_spec of original PipelineJob will be used.
            experiment (Union[str, experiments_resource.Experiment]):
                Optional. The Vertex AI experiment name or instance to associate to the PipelineJob executing
                this model evaluation job. Metrics produced by the PipelineJob as system.Metric Artifacts
                will be associated as metrics to the provided experiment, and parameters from this PipelineJob
                will be associated as parameters to the provided experiment.
        Returns:
            model_evaluation.ModelEvaluationJob: Instantiated representation of the
            _ModelEvaluationJob.
        Raises:
            ValueError:
                If staging_bucket was not set in aiplatform.init() and staging_bucket was not provided.
                If the provided `prediction_type` is not valid.
                If the provided `data_source_uris` don't start with 'gs://'.
        """

        if (gcs_source_uris is None) == (bigquery_source_uri is None):
            raise ValueError(
                "Exactly one of `gcs_source_uris` or `bigquery_source_uri` must be provided."
            )

        if isinstance(gcs_source_uris, str):
            gcs_source_uris = [gcs_source_uris]

        if bigquery_source_uri and not isinstance(bigquery_source_uri, str):
            raise ValueError("The provided `bigquery_source_uri` must be a string.")

        if bigquery_source_uri and not bigquery_destination_output_uri:
            raise ValueError(
                "`bigquery_destination_output_uri` must be provided if `bigquery_source_uri` is used as the data source."
            )

        if gcs_source_uris is not None and not all(
            uri.startswith("gs://") for uri in gcs_source_uris
        ):
            raise ValueError("`gcs_source_uris` must start with 'gs://'.")

        if bigquery_source_uri is not None and not bigquery_source_uri.startswith(
            "bq://"
        ):
            raise ValueError(
                "`bigquery_source_uri` and `bigquery_destination_output_uri` must start with 'bq://'"
            )

        if (
            bigquery_destination_output_uri is not None
            and not bigquery_destination_output_uri.startswith("bq://")
        ):
            raise ValueError(
                "`bigquery_source_uri` and `bigquery_destination_output_uri` must start with 'bq://'"
            )

        SUPPORTED_INSTANCES_FORMAT_FILE_EXTENSIONS = [".jsonl", ".csv"]

        if not staging_bucket and initializer.global_config.staging_bucket:
            staging_bucket = initializer.global_config.staging_bucket
        elif not staging_bucket and not initializer.global_config.staging_bucket:
            raise ValueError(
                "Please provide `evaluation_staging_bucket` when calling evaluate or set one using aiplatform.init(staging_bucket=...)"
            )

        if prediction_type not in _SUPPORTED_EVAL_PREDICTION_TYPES:
            raise ValueError(
                f"Please provide a supported model prediction type, one of: {_SUPPORTED_EVAL_PREDICTION_TYPES}."
            )

        if generate_feature_attributions:
            if not self._gca_resource.explanation_spec:
                raise ValueError(
                    "To generate feature attributions with your evaluation, call evaluate on a model with an explanation spec. To run evaluation on the current model, call evaluate with `generate_feature_attributions=False`."
                )

        instances_format = None

        if gcs_source_uris:
            data_file_path_obj = pathlib.Path(gcs_source_uris[0])

            data_file_extension = data_file_path_obj.suffix
            if data_file_extension not in SUPPORTED_INSTANCES_FORMAT_FILE_EXTENSIONS:
                _LOGGER.warning(
                    f"Only the following data file extensions are currently supported: '{SUPPORTED_INSTANCES_FORMAT_FILE_EXTENSIONS}'"
                )
            else:
                instances_format = data_file_extension[1:]

        elif bigquery_source_uri:
            instances_format = "bigquery"

        if (
            self._gca_resource.metadata_schema_uri
            == "https://storage.googleapis.com/google-cloud-aiplatform/schema/model/metadata/automl_tabular_1.0.0.yaml"
        ):
            model_type = "automl_tabular"
        else:
            model_type = "other"

        if (
            model_type == "other"
            and prediction_type == "classification"
            and not class_labels
        ):
            raise ValueError(
                "Please provide `class_labels` when running evaluation on a custom classification model."
            )

        return model_evaluation._ModelEvaluationJob.submit(
            model_name=self.versioned_resource_name,
            prediction_type=prediction_type,
            target_field_name=target_field_name,
            gcs_source_uris=gcs_source_uris,
            bigquery_source_uri=bigquery_source_uri,
            batch_predict_bigquery_destination_output_uri=bigquery_destination_output_uri,
            class_labels=class_labels,
            prediction_label_column=prediction_label_column,
            prediction_score_column=prediction_score_column,
            service_account=service_account,
            pipeline_root=staging_bucket,
            instances_format=instances_format,
            model_type=model_type,
            generate_feature_attributions=generate_feature_attributions,
            evaluation_pipeline_display_name=evaluation_pipeline_display_name,
            evaluation_metrics_display_name=evaluation_metrics_display_name,
            network=network,
            encryption_spec_key_name=encryption_spec_key_name,
            credentials=self.credentials,
            experiment=experiment,
        )


# TODO (b/232546878): Async support
class ModelRegistry:
    def __init__(
        self,
        model: Union[Model, str],
        location: Optional[str] = None,
        project: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Creates a ModelRegistry instance for version management of a registered model.

        Args:
            model (Union[Model, str]):
                Required. One of the following:
                    1. A Model instance
                    2. A fully-qualified model resource name
                    3. A model ID. A location and project must be provided.
            location (str):
                Optional. The model location. Used when passing a model name as model.
                If not set, project set in aiplatform.init will be used.
            project (str):
                Optional. The model project. Used when passing a model name as model.
                If not set, project set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use with model access. If not set,
                credentials set in aiplatform.init will be used.
        """

        if isinstance(model, Model):
            self.model_resource_name = model.resource_name
        else:
            self.model_resource_name = utils.full_resource_name(
                resource_name=model,
                resource_noun="models",
                parse_resource_name_method=Model._parse_resource_name,
                format_resource_name_method=Model._format_resource_name,
                project=project,
                location=location,
                resource_id_validator=base.VertexAiResourceNoun._revisioned_resource_id_validator,
            )

        self.credentials = credentials or (
            model.credentials
            if isinstance(model, Model)
            else initializer.global_config.credentials
        )
        self.client = Model._instantiate_client(location, self.credentials)

    def get_model(
        self,
        version: Optional[str] = None,
    ) -> Model:
        """Gets a registered model with optional version.

        Args:
            version (str):
                Optional. A model version ID or alias to target.
                Defaults to the model with the "default" alias.

        Returns:
            Model: An instance of a Model from this ModelRegistry.
        """
        return Model(
            self.model_resource_name, version=version, credentials=self.credentials
        )

    def list_versions(
        self,
        filter: Optional[str] = None,
    ) -> List[VersionInfo]:
        """Lists the versions and version info of a model.

        Args:
            filter (str):
                Optional. An expression for filtering the results of the request.
                For field names both snake_case and camelCase are supported.
                -  `labels` supports general map functions that is:
                    -  `labels.key=value` - key:value equality
                    -  `labels.key:* or labels:key - key existence
                    -  A key including a space must be quoted.
                        `labels."a key"`.
                Some examples:
                -  `labels.myKey="myValue"`

        Returns:
            List[VersionInfo]:
                A list of VersionInfo, each containing
                info about specific model versions.
        """

        _LOGGER.info(f"Getting versions for {self.model_resource_name}")

        request = gca_model_service_compat.ListModelVersionsRequest(
            name=self.model_resource_name,
            filter=filter,
        )

        page_result = self.client.list_model_versions(
            request=request,
        )

        versions = [
            VersionInfo(
                version_id=model.version_id,
                version_create_time=model.version_create_time,
                version_update_time=model.version_update_time,
                model_display_name=model.display_name,
                model_resource_name=self._parse_versioned_name(model.name)[0],
                version_aliases=model.version_aliases,
                version_description=model.version_description,
            )
            for model in page_result
        ]

        return versions

    def get_version_info(
        self,
        version: str,
    ) -> VersionInfo:
        """Gets information about a specific model version.

        Args:
            version (str): Required. The model version to obtain info for.

        Returns:
            VersionInfo: Contains info about the model version.
        """

        _LOGGER.info(f"Getting version {version} info for {self.model_resource_name}")

        model = self.client.get_model(
            name=self._get_versioned_name(self.model_resource_name, version),
        )

        return VersionInfo(
            version_id=model.version_id,
            version_create_time=model.version_create_time,
            version_update_time=model.version_update_time,
            model_display_name=model.display_name,
            model_resource_name=self._parse_versioned_name(model.name)[0],
            version_aliases=model.version_aliases,
            version_description=model.version_description,
        )

    def delete_version(
        self,
        version: str,
    ) -> None:
        """Deletes a model version from the registry.

        Cannot delete a version if it is the last remaining version.
        Use Model.delete() in that case.

        Args:
            version (str): Required. The model version ID or alias to delete.
        """

        lro = self.client.delete_model_version(
            name=self._get_versioned_name(self.model_resource_name, version),
        )

        _LOGGER.info(f"Deleting version {version} for {self.model_resource_name}")

        lro.result()

        _LOGGER.info(f"Deleted version {version} for {self.model_resource_name}")

    def update_version(
        self,
        version: str,
        version_description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Updates a model version.

        Args:
            version (str): Required. The version ID to receive the new alias(es).
            version_description (str):
                The description of the model version.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Model versions.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.

        Raises:
            ValueError: If `labels` is not the correct format.
        """

        current_model_proto = self.get_model(version).gca_resource
        copied_model_proto = current_model_proto.__class__(current_model_proto)

        update_mask: List[str] = []

        if version_description:
            copied_model_proto.version_description = version_description
            update_mask.append("version_description")

        if labels:
            utils.validate_labels(labels)

            copied_model_proto.labels = labels
            update_mask.append("labels")

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)
        versioned_name = self._get_versioned_name(self.model_resource_name, version)

        _LOGGER.info(f"Updating model {versioned_name}")

        self.client.update_model(
            model=copied_model_proto,
            update_mask=update_mask,
        )

        _LOGGER.info(f"Completed updating model {versioned_name}")

    def add_version_aliases(
        self,
        new_aliases: List[str],
        version: str,
    ) -> None:
        """Adds version alias(es) to a model version.

        Args:
            new_aliases (List[str]): Required. The alias(es) to add to a model version.
            version (str): Required. The version ID to receive the new alias(es).
        """

        self._merge_version_aliases(
            version_aliases=new_aliases,
            version=version,
        )

    def remove_version_aliases(
        self,
        target_aliases: List[str],
        version: str,
    ) -> None:
        """Removes version alias(es) from a model version.

        Args:
            target_aliases (List[str]): Required. The alias(es) to remove from a model version.
            version (str): Required. The version ID to be stripped of the target alias(es).
        """

        self._merge_version_aliases(
            version_aliases=[f"-{alias}" for alias in target_aliases],
            version=version,
        )

    def _merge_version_aliases(
        self,
        version_aliases: List[str],
        version: str,
    ) -> None:
        """Merges a list of version aliases with a model's existing alias list.

        Args:
            version_aliases (List[str]): Required. The version alias change list.
            version (str): Required. The version ID to have its alias list changed.
        """

        _LOGGER.info(f"Merging version aliases for {self.model_resource_name}")

        self.client.merge_version_aliases(
            name=self._get_versioned_name(self.model_resource_name, version),
            version_aliases=version_aliases,
        )

        _LOGGER.info(
            f"Completed merging version aliases for {self.model_resource_name}"
        )

    @staticmethod
    def _get_versioned_name(
        resource_name: str,
        version: Optional[str] = None,
    ) -> str:
        """Creates a versioned form of a model resource name.

        Args:
            resource_name (str): Required. A fully-qualified resource name or resource ID.
            version (str): Optional. The version or alias of the resource.

        Returns:
            versioned_name (str): The versioned resource name in revisioned format.
        """
        if version:
            return f"{resource_name}@{version}"
        return resource_name

    @staticmethod
    def _parse_versioned_name(
        model_name: str,
    ) -> Tuple[str, Optional[str]]:
        """Return a model name and, if included in the model name, a model version.

        Args:
            model_name (str): Required. A fully-qualified model name or model ID,
                optionally with an included version.

        Returns:
            parsed_version_name (Tuple[str, Optional[str]]):
                A tuple containing the model name or ID as the first element,
                and the model version as the second element, if present in `model_name`.

        Raises:
            ValueError: If the `model_name` is invalid and contains too many '@' symbols.
        """
        if "@" not in model_name:
            return model_name, None
        elif model_name.count("@") > 1:
            raise ValueError(
                f"Received an invalid model_name with too many `@`s: {model_name}"
            )
        else:
            return model_name.split("@")

    @staticmethod
    def _get_true_version_parent(
        parent_model: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Optional[str]:
        """Gets the true `parent_model` with full resource name.

        Args:
            parent_model (str): Optional. A fully-qualified resource name or resource ID
                of the model that would be the parent of another model.
            project (str): Optional. The project of `parent_model`, if not included in `parent_model`.
            location (str): Optional. The location of `parent_model`, if not included in `parent_model`.

        Returns:
            true_parent_model (str):
                Optional. The true resource name of the parent model, if one should exist.
        """
        if parent_model:
            existing_resource = utils.full_resource_name(
                resource_name=parent_model,
                resource_noun="models",
                parse_resource_name_method=Model._parse_resource_name,
                format_resource_name_method=Model._format_resource_name,
                project=project,
                location=location,
            )
            parent_model = existing_resource
        return parent_model

    @staticmethod
    def _get_true_alias_list(
        version_aliases: Optional[Sequence[str]] = None,
        is_default_version: bool = True,
    ) -> Optional[Sequence[str]]:
        """Gets the true `version_aliases` list based on `is_default_version`.

        Args:
            version_aliases (Sequence[str]): Optional. The user-provided list of model aliases.
            is_default_version (bool):
                Optional. When set, includes the "default" alias in `version_aliases`.
                Defaults to True.

        Returns:
            true_alias_list (Sequence[str]):
                Optional: The true alias list, should one exist,
                containing "default" if specified.
        """
        if is_default_version:
            if version_aliases and "default" not in version_aliases:
                version_aliases.append("default")
            elif not version_aliases:
                version_aliases = ["default"]
        return version_aliases
