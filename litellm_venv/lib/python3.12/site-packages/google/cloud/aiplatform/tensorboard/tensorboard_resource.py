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

from typing import Dict, List, Optional, Sequence, Tuple, Union

from google.auth import credentials as auth_credentials
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    tensorboard as gca_tensorboard,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_data as gca_tensorboard_data,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_experiment as gca_tensorboard_experiment,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_run as gca_tensorboard_run,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_service as gca_tensorboard_service,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_time_series as gca_tensorboard_time_series,
)

_LOGGER = base.Logger(__name__)


class _TensorboardServiceResource(base.VertexAiResourceNounWithFutureManager):
    client_class = utils.TensorboardClientWithOverride


class Tensorboard(_TensorboardServiceResource):
    """Managed tensorboard resource for Vertex AI."""

    _resource_noun = "tensorboards"
    _getter_method = "get_tensorboard"
    _list_method = "list_tensorboards"
    _delete_method = "delete_tensorboard"
    _parse_resource_name_method = "parse_tensorboard_path"
    _format_resource_name_method = "tensorboard_path"

    def __init__(
        self,
        tensorboard_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing managed tensorboard given a tensorboard name or ID.

        Args:
            tensorboard_name (str):
                Required. A fully-qualified tensorboard resource name or tensorboard ID.
                Example: "projects/123/locations/us-central1/tensorboards/456" or
                "456" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve tensorboard from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve tensorboard from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Tensorboard. Overrides
                credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=tensorboard_name,
        )
        self._gca_resource = self._get_gca_resource(resource_name=tensorboard_name)

    @classmethod
    def create(
        cls,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        encryption_spec_key_name: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
        is_default: bool = False,
    ) -> "Tensorboard":
        """Creates a new tensorboard.

        Example Usage:
        ```py
        tb = aiplatform.Tensorboard.create(
            display_name='my display name',
            description='my description',
            labels={
                'key1': 'value1',
                'key2': 'value2'
            }
        )
        ```

        Args:
            display_name (str):
                Optional. The user-defined name of the Tensorboard.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            description (str):
                Optional. Description of this Tensorboard.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your Tensorboards.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                No more than 64 user labels can be associated with one Tensorboard
                (System labels are excluded).
                See https://goo.gl/xmQnxf for more information and examples of labels.
                System reserved label keys are prefixed with "aiplatform.googleapis.com/"
                and are immutable.
            project (str):
                Optional. Project to upload this model to. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            encryption_spec_key_name (str):
                Optional. Cloud KMS resource identifier of the customer
                managed encryption key used to protect the tensorboard. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Tensorboard and all sub-resources of this Tensorboard will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            is_default (bool):
                If the TensorBoard instance is default or not. The default
                TensorBoard instance will be used by Experiment/ExperimentRun
                when needed if no TensorBoard instance is explicitly specified.

        Returns:
            tensorboard (Tensorboard):
                Instantiated representation of the managed tensorboard resource.
        """
        if not display_name:
            display_name = cls._generate_display_name()

        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        parent = initializer.global_config.common_location_path(
            project=project, location=location
        )

        encryption_spec = initializer.global_config.get_encryption_spec(
            encryption_spec_key_name=encryption_spec_key_name
        )

        gapic_tensorboard = gca_tensorboard.Tensorboard(
            display_name=display_name,
            description=description,
            labels=labels,
            is_default=is_default,
            encryption_spec=encryption_spec,
        )

        create_tensorboard_lro = api_client.create_tensorboard(
            parent=parent,
            tensorboard=gapic_tensorboard,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, create_tensorboard_lro)

        created_tensorboard = create_tensorboard_lro.result()

        _LOGGER.log_create_complete(cls, created_tensorboard, "tb")

        return cls(
            tensorboard_name=created_tensorboard.name,
            credentials=credentials,
        )

    def update(
        self,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        encryption_spec_key_name: Optional[str] = None,
        is_default: Optional[bool] = None,
    ) -> "Tensorboard":
        """Updates an existing tensorboard.

        Example Usage:
        ```py
        tb = aiplatform.Tensorboard(tensorboard_name='123456')
        tb.update(
            display_name='update my display name',
            description='update my description',
        )
        ```

        Args:
            display_name (str):
                Optional. User-defined name of the Tensorboard.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            description (str):
                Optional. Description of this Tensorboard.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your Tensorboards.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                No more than 64 user labels can be associated with one Tensorboard
                (System labels are excluded).
                See https://goo.gl/xmQnxf for more information and examples of labels.
                System reserved label keys are prefixed with "aiplatform.googleapis.com/"
                and are immutable.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            encryption_spec_key_name (str):
                Optional. Cloud KMS resource identifier of the customer
                managed encryption key used to protect the tensorboard. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Tensorboard and all sub-resources of this Tensorboard will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            is_default (bool):
                Optional. If the TensorBoard instance is default or not.
                The default TensorBoard instance will be used by
                Experiment/ExperimentRun when needed if no TensorBoard instance
                is explicitly specified.

        Returns:
            Tensorboard: The managed tensorboard resource.
        """
        update_mask = list()

        if display_name:
            utils.validate_display_name(display_name)
            update_mask.append("display_name")

        if description:
            update_mask.append("description")

        if labels:
            utils.validate_labels(labels)
            update_mask.append("labels")

        if is_default is not None:
            update_mask.append("is_default")

        encryption_spec = None
        if encryption_spec_key_name:
            encryption_spec = initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name,
            )
            update_mask.append("encryption_spec")

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)

        gapic_tensorboard = gca_tensorboard.Tensorboard(
            name=self.resource_name,
            display_name=display_name,
            description=description,
            labels=labels,
            is_default=is_default,
            encryption_spec=encryption_spec,
        )

        _LOGGER.log_action_start_against_resource(
            "Updating",
            "tensorboard",
            self,
        )

        update_tensorboard_lro = self.api_client.update_tensorboard(
            tensorboard=gapic_tensorboard,
            update_mask=update_mask,
            metadata=request_metadata,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Update", "tensorboard", self.__class__, update_tensorboard_lro
        )

        update_tensorboard_lro.result()

        _LOGGER.log_action_completed_against_resource("tensorboard", "updated", self)

        return self


class TensorboardExperiment(_TensorboardServiceResource):
    """Managed tensorboard resource for Vertex AI."""

    _resource_noun = "experiments"
    _getter_method = "get_tensorboard_experiment"
    _list_method = "list_tensorboard_experiments"
    _delete_method = "delete_tensorboard_experiment"
    _parse_resource_name_method = "parse_tensorboard_experiment_path"
    _format_resource_name_method = "tensorboard_experiment_path"

    def __init__(
        self,
        tensorboard_experiment_name: str,
        tensorboard_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing tensorboard experiment given a tensorboard experiment name or ID.

        Example Usage:
        ```py
        tb_exp = aiplatform.TensorboardExperiment(
            tensorboard_experiment_name= "projects/123/locations/us-central1/tensorboards/456/experiments/678"
        )

        tb_exp = aiplatform.TensorboardExperiment(
            tensorboard_experiment_name= "678"
            tensorboard_id = "456"
        )
        ```

        Args:
            tensorboard_experiment_name (str):
                Required. A fully-qualified tensorboard experiment resource name or resource ID.
                Example: "projects/123/locations/us-central1/tensorboards/456/experiments/678" or
                "678" when tensorboard_id is passed and project and location are initialized or passed.
            tensorboard_id (str):
                Optional. A tensorboard resource ID.
            project (str):
                Optional. Project to retrieve tensorboard from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve tensorboard from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Tensorboard. Overrides
                credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=tensorboard_experiment_name,
        )
        self._gca_resource = self._get_gca_resource(
            resource_name=tensorboard_experiment_name,
            parent_resource_name_fields={Tensorboard._resource_noun: tensorboard_id}
            if tensorboard_id
            else tensorboard_id,
        )

    @classmethod
    def create(
        cls,
        tensorboard_experiment_id: str,
        tensorboard_name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Sequence[Tuple[str, str]] = (),
        create_request_timeout: Optional[float] = None,
    ) -> "TensorboardExperiment":
        """Creates a new TensorboardExperiment.

        Example Usage:
        ```py
        tb_exp = aiplatform.TensorboardExperiment.create(
            tensorboard_experiment_id='my-experiment'
            tensorboard_id='456'
            display_name='my display name',
            description='my description',
            labels={
                'key1': 'value1',
                'key2': 'value2'
            }
        )
        ```

        Args:
            tensorboard_experiment_id (str):
                Required. The ID to use for the Tensorboard experiment,
                which will become the final component of the Tensorboard
                experiment's resource name.

                This value should be 1-128 characters, and valid
                characters are /[a-z][0-9]-/.

                This corresponds to the ``tensorboard_experiment_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            tensorboard_name (str):
                Required. The resource name or ID of the Tensorboard to create
                the TensorboardExperiment in. Format of resource name:
                ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
            display_name (str):
                Optional. The user-defined name of the Tensorboard Experiment.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            description (str):
                Optional. Description of this Tensorboard Experiment.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your Tensorboards.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                No more than 64 user labels can be associated with one Tensorboard
                (System labels are excluded).
                See https://goo.gl/xmQnxf for more information and examples of labels.
                System reserved label keys are prefixed with "aiplatform.googleapis.com/"
                and are immutable.
            project (str):
                Optional. Project to upload this model to. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
        Returns:
            TensorboardExperiment: The TensorboardExperiment resource.
        """

        if display_name:
            utils.validate_display_name(display_name)

        if labels:
            utils.validate_labels(labels)

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        parent = utils.full_resource_name(
            resource_name=tensorboard_name,
            resource_noun=Tensorboard._resource_noun,
            parse_resource_name_method=Tensorboard._parse_resource_name,
            format_resource_name_method=Tensorboard._format_resource_name,
            project=project,
            location=location,
        )

        gapic_tensorboard_experiment = gca_tensorboard_experiment.TensorboardExperiment(
            display_name=display_name,
            description=description,
            labels=labels,
        )

        _LOGGER.log_create_with_lro(cls)

        tensorboard_experiment = api_client.create_tensorboard_experiment(
            parent=parent,
            tensorboard_experiment=gapic_tensorboard_experiment,
            tensorboard_experiment_id=tensorboard_experiment_id,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_complete(cls, tensorboard_experiment, "tb experiment")

        return cls(
            tensorboard_experiment_name=tensorboard_experiment.name,
            credentials=credentials,
        )

    @classmethod
    def list(
        cls,
        tensorboard_name: str,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["TensorboardExperiment"]:
        """List TensorboardExperiemnts in a Tensorboard resource.

        ```py
        Example Usage:
            aiplatform.TensorboardExperiment.list(
                tensorboard_name='projects/my-project/locations/us-central1/tensorboards/123'
            )
        ```

        Args:
            tensorboard_name(str):
                Required. The resource name or resource ID of the
                Tensorboard to list
                TensorboardExperiments. Format, if resource name:
                'projects/{project}/locations/{location}/tensorboards/{tensorboard}'
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
            List[TensorboardExperiment] - A list of TensorboardExperiments
        """

        parent = utils.full_resource_name(
            resource_name=tensorboard_name,
            resource_noun=Tensorboard._resource_noun,
            parse_resource_name_method=Tensorboard._parse_resource_name,
            format_resource_name_method=Tensorboard._format_resource_name,
            project=project,
            location=location,
        )

        return super()._list(
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
            parent=parent,
        )


class TensorboardRun(_TensorboardServiceResource):
    """Managed tensorboard resource for Vertex AI."""

    _resource_noun = "runs"
    _getter_method = "get_tensorboard_run"
    _list_method = "list_tensorboard_runs"
    _delete_method = "delete_tensorboard_run"
    _parse_resource_name_method = "parse_tensorboard_run_path"
    _format_resource_name_method = "tensorboard_run_path"
    READ_TIME_SERIES_BATCH_SIZE = 20

    def __init__(
        self,
        tensorboard_run_name: str,
        tensorboard_id: Optional[str] = None,
        tensorboard_experiment_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing tensorboard run given a tensorboard run name or ID.

        ```py
        Example Usage:
            tb_run = aiplatform.TensorboardRun(
                tensorboard_run_name= "projects/123/locations/us-central1/tensorboards/456/experiments/678/run/8910"
            )

            tb_run = aiplatform.TensorboardRun(
                tensorboard_run_name= "8910",
                tensorboard_id = "456",
                tensorboard_experiment_id = "678"
            )
        ```

        Args:
            tensorboard_run_name (str):
                Required. A fully-qualified tensorboard run resource name or resource ID.
                Example: "projects/123/locations/us-central1/tensorboards/456/experiments/678/runs/8910" or
                "8910" when tensorboard_id and tensorboard_experiment_id are passed
                and project and location are initialized or passed.
            tensorboard_id (str):
                Optional. A tensorboard resource ID.
            tensorboard_experiment_id (str):
                Optional. A tensorboard experiment resource ID.
            project (str):
                Optional. Project to retrieve tensorboard from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve tensorboard from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Tensorboard. Overrides
                credentials set in aiplatform.init.
        Raises:
            ValueError: if only one of tensorboard_id or tensorboard_experiment_id is provided.
        """
        if bool(tensorboard_id) != bool(tensorboard_experiment_id):
            raise ValueError(
                "Both tensorboard_id and tensorboard_experiment_id must be provided or neither should be provided."
            )

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=tensorboard_run_name,
        )
        self._gca_resource = self._get_gca_resource(
            resource_name=tensorboard_run_name,
            parent_resource_name_fields={
                Tensorboard._resource_noun: tensorboard_id,
                TensorboardExperiment._resource_noun: tensorboard_experiment_id,
            }
            if tensorboard_id
            else tensorboard_id,
        )

        self._time_series_display_name_to_id_mapping = (
            self._get_time_series_display_name_to_id_mapping()
        )

    @classmethod
    def create(
        cls,
        tensorboard_run_id: str,
        tensorboard_experiment_name: str,
        tensorboard_id: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Sequence[Tuple[str, str]] = (),
        create_request_timeout: Optional[float] = None,
    ) -> "TensorboardRun":
        """Creates a new tensorboard run.

        Example Usage:
        ```py
        tb_run = aiplatform.TensorboardRun.create(
            tensorboard_run_id='my-run'
            tensorboard_experiment_name='my-experiment'
            tensorboard_id='456'
            display_name='my display name',
            description='my description',
            labels={
                'key1': 'value1',
                'key2': 'value2'
            }
        )
        ```

        Args:
            tensorboard_run_id (str):
                Required. The ID to use for the Tensorboard run, which
                will become the final component of the Tensorboard run's
                resource name.

                This value should be 1-128 characters, and valid:
                characters are /[a-z][0-9]-/.
            tensorboard_experiment_name (str):
                Required. The resource name or ID of the TensorboardExperiment
                to create the TensorboardRun in. Resource name format:
                ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``

                If resource ID is provided then tensorboard_id must be provided.
            tensorboard_id (str):
                Optional. The resource ID of the Tensorboard to create the TensorboardRun in.
            display_name (str):
                Optional. The user-defined name of the Tensorboard Run.
                This value must be unique among all TensorboardRuns belonging to the
                same parent TensorboardExperiment.

                If not provided tensorboard_run_id will be used.
            description (str):
                Optional. Description of this Tensorboard Run.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your Tensorboards.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                No more than 64 user labels can be associated with one Tensorboard
                (System labels are excluded).
                See https://goo.gl/xmQnxf for more information and examples of labels.
                System reserved label keys are prefixed with "aiplatform.googleapis.com/"
                and are immutable.
            project (str):
                Optional. Project to upload this model to. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
        Returns:
            TensorboardRun: The TensorboardRun resource.
        """
        if display_name:
            utils.validate_display_name(display_name)

        if labels:
            utils.validate_labels(labels)

        display_name = display_name or tensorboard_run_id

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        parent = utils.full_resource_name(
            resource_name=tensorboard_experiment_name,
            resource_noun=TensorboardExperiment._resource_noun,
            parse_resource_name_method=TensorboardExperiment._parse_resource_name,
            format_resource_name_method=TensorboardExperiment._format_resource_name,
            parent_resource_name_fields={Tensorboard._resource_noun: tensorboard_id},
            project=project,
            location=location,
        )

        gapic_tensorboard_run = gca_tensorboard_run.TensorboardRun(
            display_name=display_name,
            description=description,
            labels=labels,
        )

        _LOGGER.log_create_with_lro(cls)

        tensorboard_run = api_client.create_tensorboard_run(
            parent=parent,
            tensorboard_run=gapic_tensorboard_run,
            tensorboard_run_id=tensorboard_run_id,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_complete(cls, tensorboard_run, "tb_run")

        return cls(
            tensorboard_run_name=tensorboard_run.name,
            credentials=credentials,
        )

    @classmethod
    def list(
        cls,
        tensorboard_experiment_name: str,
        tensorboard_id: Optional[str] = None,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["TensorboardRun"]:
        """List all instances of TensorboardRun in TensorboardExperiment.

        Example Usage:
        ```py
        aiplatform.TensorboardRun.list(
            tensorboard_experiment_name='projects/my-project/locations/us-central1/tensorboards/123/experiments/456'
        )
        ```

        Args:
            tensorboard_experiment_name (str):
                Required. The resource name or resource ID of the
                TensorboardExperiment to list
                TensorboardRun. Format, if resource name:
                'projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}'

                If resource ID is provided then tensorboard_id must be provided.
            tensorboard_id (str):
                Optional. The resource ID of the Tensorboard that contains the TensorboardExperiment
                to list TensorboardRun.
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
            List[TensorboardRun] - A list of TensorboardRun
        """

        parent = utils.full_resource_name(
            resource_name=tensorboard_experiment_name,
            resource_noun=TensorboardExperiment._resource_noun,
            parse_resource_name_method=TensorboardExperiment._parse_resource_name,
            format_resource_name_method=TensorboardExperiment._format_resource_name,
            parent_resource_name_fields={Tensorboard._resource_noun: tensorboard_id},
            project=project,
            location=location,
        )

        tensorboard_runs = super()._list(
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
            parent=parent,
        )

        for tensorboard_run in tensorboard_runs:
            tensorboard_run._sync_time_series_display_name_to_id_mapping()

        return tensorboard_runs

    def write_tensorboard_scalar_data(
        self,
        time_series_data: Dict[str, float],
        step: int,
        wall_time: Optional[timestamp_pb2.Timestamp] = None,
    ):
        """Writes tensorboard scalar data to this run.

        Args:
            time_series_data (Dict[str, float]):
                Required. Dictionary of where keys are TensorboardTimeSeries display name and values are the scalar value..
            step (int):
                Required. Step index of this data point within the run.
            wall_time (timestamp_pb2.Timestamp):
                Optional. Wall clock timestamp when this data point is
                generated by the end user.

                If not provided, this will be generated based on the value from time.time()
        """

        if not wall_time:
            wall_time = utils.get_timestamp_proto()

        ts_data = []

        if any(
            key not in self._time_series_display_name_to_id_mapping
            for key in time_series_data.keys()
        ):
            self._sync_time_series_display_name_to_id_mapping()

        for display_name, value in time_series_data.items():
            time_series_id = self._time_series_display_name_to_id_mapping.get(
                display_name
            )

            if not time_series_id:
                raise RuntimeError(
                    f"TensorboardTimeSeries with display name {display_name} has not been created in TensorboardRun {self.resource_name}."
                )

            ts_data.append(
                gca_tensorboard_data.TimeSeriesData(
                    tensorboard_time_series_id=time_series_id,
                    value_type=gca_tensorboard_time_series.TensorboardTimeSeries.ValueType.SCALAR,
                    values=[
                        gca_tensorboard_data.TimeSeriesDataPoint(
                            scalar=gca_tensorboard_data.Scalar(value=value),
                            wall_time=wall_time,
                            step=step,
                        )
                    ],
                )
            )

        self.api_client.write_tensorboard_run_data(
            tensorboard_run=self.resource_name, time_series_data=ts_data
        )

    def _get_time_series_display_name_to_id_mapping(self) -> Dict[str, str]:
        """Returns a mapping of the TimeSeries display names to resource IDs for this Run.

        Returns:
            Dict[str, str] - Dictionary mapping TensorboardTimeSeries display names to
                resource IDs of TensorboardTimeSeries in this TensorboardRun."""
        time_series = TensorboardTimeSeries.list(
            tensorboard_run_name=self.resource_name, credentials=self.credentials
        )

        return {ts.display_name: ts.name for ts in time_series}

    def _sync_time_series_display_name_to_id_mapping(self):
        """Updates the local map of TimeSeries diplay name to resource ID."""
        self._time_series_display_name_to_id_mapping = (
            self._get_time_series_display_name_to_id_mapping()
        )

    def create_tensorboard_time_series(
        self,
        display_name: str,
        value_type: Union[
            gca_tensorboard_time_series.TensorboardTimeSeries.ValueType, str
        ] = "SCALAR",
        plugin_name: str = "scalars",
        plugin_data: Optional[bytes] = None,
        description: Optional[str] = None,
    ) -> "TensorboardTimeSeries":
        """Creates a new tensorboard time series.

        Example Usage:
        ```py
        tb_ts = tensorboard_run.create_tensorboard_time_series(
            display_name='my display name',
            tensorboard_run_name='my-run'
            tensorboard_id='456'
            tensorboard_experiment_id='my-experiment'
            description='my description',
            labels={
                'key1': 'value1',
                'key2': 'value2'
            }
        )
        ```

        Args:
            display_name (str):
                Optional. User provided name of this
                TensorboardTimeSeries. This value should be
                unique among all TensorboardTimeSeries resources
                belonging to the same TensorboardRun resource
                (parent resource).
            value_type (Union[gca_tensorboard_time_series.TensorboardTimeSeries.ValueType, str]):
                Optional. Type of TensorboardTimeSeries value. One of 'SCALAR', 'TENSOR', 'BLOB_SEQUENCE'.
            plugin_name (str):
                Optional. Name of the plugin this time series pertain to. Such as Scalar, Tensor, Blob.
            plugin_data (bytes):
                Optional. Data of the current plugin, with the size limited to 65KB.
            description (str):
                Optional. Description of this TensorboardTimeseries.
        Returns:
            TensorboardTimeSeries: The TensorboardTimeSeries resource.
        """

        tb_time_series = TensorboardTimeSeries.create(
            display_name=display_name,
            tensorboard_run_name=self.resource_name,
            value_type=value_type,
            plugin_name=plugin_name,
            plugin_data=plugin_data,
            description=description,
            credentials=self.credentials,
        )

        self._time_series_display_name_to_id_mapping[
            tb_time_series.display_name
        ] = tb_time_series.name

        return tb_time_series

    def read_time_series_data(self) -> Dict[str, gca_tensorboard_data.TimeSeriesData]:
        """Read the time series data of this run.

        ```py
        time_series_data = tensorboard_run.read_time_series_data()

        print(time_series_data['loss'].values[-1].scalar.value)
        ```

        Returns:
            Dictionary of time series metric id to TimeSeriesData.
        """
        self._sync_time_series_display_name_to_id_mapping()

        resource_name_parts = self._parse_resource_name(self.resource_name)
        inverted_mapping = {
            resource_id: display_name
            for display_name, resource_id in self._time_series_display_name_to_id_mapping.items()
        }

        time_series_resource_names = [
            TensorboardTimeSeries._format_resource_name(
                time_series=resource_id, **resource_name_parts
            )
            for resource_id in inverted_mapping.keys()
        ]

        resource_name_parts.pop("experiment")
        resource_name_parts.pop("run")

        tensorboard_resource_name = Tensorboard._format_resource_name(
            **resource_name_parts
        )

        batch_size = self.READ_TIME_SERIES_BATCH_SIZE
        time_series_data_dict = {}
        for i in range(0, len(time_series_resource_names), batch_size):
            one_batch_time_series_names = time_series_resource_names[i : i + batch_size]
            read_response = self.api_client.batch_read_tensorboard_time_series_data(
                request=gca_tensorboard_service.BatchReadTensorboardTimeSeriesDataRequest(
                    tensorboard=tensorboard_resource_name,
                    time_series=one_batch_time_series_names,
                )
            )

            time_series_data_dict.update(
                {
                    inverted_mapping[data.tensorboard_time_series_id]: data
                    for data in read_response.time_series_data
                }
            )
        return time_series_data_dict


class TensorboardTimeSeries(_TensorboardServiceResource):
    """Managed tensorboard resource for Vertex AI."""

    _resource_noun = "timeSeries"
    _getter_method = "get_tensorboard_time_series"
    _list_method = "list_tensorboard_time_series"
    _delete_method = "delete_tensorboard_time_series"
    _parse_resource_name_method = "parse_tensorboard_time_series_path"
    _format_resource_name_method = "tensorboard_time_series_path"

    def __init__(
        self,
        tensorboard_time_series_name: str,
        tensorboard_id: Optional[str] = None,
        tensorboard_experiment_id: Optional[str] = None,
        tensorboard_run_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing tensorboard time series given a tensorboard time series name or ID.

        Example Usage:
        ```py
        tb_ts = aiplatform.TensorboardTimeSeries(
            tensorboard_time_series_name="projects/123/locations/us-central1/tensorboards/456/experiments/789/run/1011/timeSeries/mse"
        )

        tb_ts = aiplatform.TensorboardTimeSeries(
            tensorboard_time_series_name= "mse",
            tensorboard_id = "456",
            tensorboard_experiment_id = "789"
            tensorboard_run_id = "1011"
        )
        ```

        Args:
            tensorboard_time_series_name (str):
                Required. A fully-qualified tensorboard time series resource name or resource ID.
                Example: "projects/123/locations/us-central1/tensorboards/456/experiments/789/run/1011/timeSeries/mse" or
                "mse" when tensorboard_id, tensorboard_experiment_id, tensorboard_run_id are passed
                and project and location are initialized or passed.
            tensorboard_id (str):
                Optional. A tensorboard resource ID.
            tensorboard_experiment_id (str):
                Optional. A tensorboard experiment resource ID.
            tensorboard_run_id (str):
                Optional. A tensorboard run resource ID.
            project (str):
                Optional. Project to retrieve tensorboard from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve tensorboard from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Tensorboard. Overrides
                credentials set in aiplatform.init.
        Raises:
            ValueError: if only one of tensorboard_id or tensorboard_experiment_id is provided.
        """
        if not (
            bool(tensorboard_id)
            == bool(tensorboard_experiment_id)
            == bool(tensorboard_run_id)
        ):
            raise ValueError(
                "tensorboard_id, tensorboard_experiment_id, tensorboard_run_id must all be provided or none should be provided."
            )

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=tensorboard_time_series_name,
        )
        self._gca_resource = self._get_gca_resource(
            resource_name=tensorboard_time_series_name,
            parent_resource_name_fields={
                Tensorboard._resource_noun: tensorboard_id,
                TensorboardExperiment._resource_noun: tensorboard_experiment_id,
                TensorboardRun._resource_noun: tensorboard_run_id,
            }
            if tensorboard_id
            else tensorboard_id,
        )

    @classmethod
    def create(
        cls,
        display_name: str,
        tensorboard_run_name: str,
        tensorboard_id: Optional[str] = None,
        tensorboard_experiment_id: Optional[str] = None,
        value_type: Union[
            gca_tensorboard_time_series.TensorboardTimeSeries.ValueType, str
        ] = "SCALAR",
        plugin_name: str = "scalars",
        plugin_data: Optional[bytes] = None,
        description: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "TensorboardTimeSeries":
        """Creates a new tensorboard time series.

        Example Usage:
        ```py
        tb_ts = aiplatform.TensorboardTimeSeries.create(
            display_name='my display name',
            tensorboard_run_name='my-run'
            tensorboard_id='456'
            tensorboard_experiment_id='my-experiment'
            description='my description',
            labels={
                'key1': 'value1',
                'key2': 'value2'
            }
        )
        ```

        Args:
            display_name (str):
                Optional. User provided name of this
                TensorboardTimeSeries. This value should be
                unique among all TensorboardTimeSeries resources
                belonging to the same TensorboardRun resource
                (parent resource).
            tensorboard_run_name (str):
                Required. The resource name or ID of the TensorboardRun
                to create the TensorboardTimeseries in. Resource name format:
                ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}``

                If resource ID is provided then tensorboard_id and tensorboard_experiment_id must be provided.
            tensorboard_id (str):
                Optional. The resource ID of the Tensorboard to create the TensorboardTimeSeries in.
            tensorboard_experiment_id (str):
                Optional. The ID of the TensorboardExperiment to create the TensorboardTimeSeries in.
            value_type (Union[gca_tensorboard_time_series.TensorboardTimeSeries.ValueType, str]):
                Optional. Type of TensorboardTimeSeries value. One of 'SCALAR', 'TENSOR', 'BLOB_SEQUENCE'.
            plugin_name (str):
                Optional. Name of the plugin this time series pertain to.
            plugin_data (bytes):
                Optional. Data of the current plugin, with the size limited to 65KB.
            description (str):
                Optional. Description of this TensorboardTimeseries.
            project (str):
                Optional. Project to upload this model to. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
        Returns:
            TensorboardTimeSeries: The TensorboardTimeSeries resource.
        """

        if isinstance(value_type, str):
            value_type = getattr(
                gca_tensorboard_time_series.TensorboardTimeSeries.ValueType, value_type
            )

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        parent = utils.full_resource_name(
            resource_name=tensorboard_run_name,
            resource_noun=TensorboardRun._resource_noun,
            parse_resource_name_method=TensorboardRun._parse_resource_name,
            format_resource_name_method=TensorboardRun._format_resource_name,
            parent_resource_name_fields={
                Tensorboard._resource_noun: tensorboard_id,
                TensorboardExperiment._resource_noun: tensorboard_experiment_id,
            },
            project=project,
            location=location,
        )

        gapic_tensorboard_time_series = (
            gca_tensorboard_time_series.TensorboardTimeSeries(
                display_name=display_name,
                description=description,
                value_type=value_type,
                plugin_name=plugin_name,
                plugin_data=plugin_data,
            )
        )

        _LOGGER.log_create_with_lro(cls)

        tensorboard_time_series = api_client.create_tensorboard_time_series(
            parent=parent, tensorboard_time_series=gapic_tensorboard_time_series
        )

        _LOGGER.log_create_complete(cls, tensorboard_time_series, "tb_time_series")

        self = cls._empty_constructor(
            project=project, location=location, credentials=credentials
        )
        self._gca_resource = tensorboard_time_series

        return self

    @classmethod
    def list(
        cls,
        tensorboard_run_name: str,
        tensorboard_id: Optional[str] = None,
        tensorboard_experiment_id: Optional[str] = None,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["TensorboardTimeSeries"]:
        """List all instances of TensorboardTimeSeries in TensorboardRun.

        Example Usage:
        ```py
        aiplatform.TensorboardTimeSeries.list(
            tensorboard_run_name='projects/my-project/locations/us-central1/tensorboards/123/experiments/my-experiment/runs/my-run'
        )
        ```

        Args:
            tensorboard_run_name (str):
                Required. The resource name or ID of the TensorboardRun
                to list the TensorboardTimeseries from. Resource name format:
                ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}``

                If resource ID is provided then tensorboard_id and tensorboard_experiment_id must be provided.
            tensorboard_id (str):
                Optional. The resource ID of the Tensorboard to list the TensorboardTimeSeries from.
            tensorboard_experiment_id (str):
                Optional. The ID of the TensorboardExperiment to list the TensorboardTimeSeries from.
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
            List[TensorboardTimeSeries] - A list of TensorboardTimeSeries
        """

        parent = utils.full_resource_name(
            resource_name=tensorboard_run_name,
            resource_noun=TensorboardRun._resource_noun,
            parse_resource_name_method=TensorboardRun._parse_resource_name,
            format_resource_name_method=TensorboardRun._format_resource_name,
            parent_resource_name_fields={
                Tensorboard._resource_noun: tensorboard_id,
                TensorboardExperiment._resource_noun: tensorboard_experiment_id,
            },
            project=project,
            location=location,
        )

        return super()._list(
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
            parent=parent,
        )
