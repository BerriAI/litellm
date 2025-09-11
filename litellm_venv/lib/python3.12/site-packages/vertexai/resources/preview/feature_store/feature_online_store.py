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

import enum
from typing import (
    Dict,
    Optional,
    Sequence,
    Tuple,
)

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import (
    base,
    initializer,
    utils,
)
from google.cloud.aiplatform.compat.types import (
    feature_online_store as gca_feature_online_store,
    feature_view as gca_feature_view,
)
from vertexai.resources.preview.feature_store.feature_view import (
    FeatureView,
)
from vertexai.resources.preview.feature_store.utils import (
    IndexConfig,
)


_LOGGER = base.Logger(__name__)


@enum.unique
class FeatureOnlineStoreType(enum.Enum):
    UNKNOWN = 0
    BIGTABLE = 1
    OPTIMIZED = 2


class FeatureOnlineStore(base.VertexAiResourceNounWithFutureManager):
    """Class for managing Feature Online Store resources."""

    client_class = utils.FeatureOnlineStoreAdminClientWithOverride

    _resource_noun = "feature_online_stores"
    _getter_method = "get_feature_online_store"
    _list_method = "list_feature_online_stores"
    _delete_method = "delete_feature_online_store"
    _parse_resource_name_method = "parse_feature_online_store_path"
    _format_resource_name_method = "feature_online_store_path"
    _gca_resource: gca_feature_online_store.FeatureOnlineStore

    def __init__(
        self,
        name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing managed feature online store.

        Args:
            name:
                The resource name
                (`projects/.../locations/.../featureOnlineStores/...`) or ID.
            project:
                Project to retrieve feature online store from. If unset, the
                project set in aiplatform.init will be used.
            location:
                Location to retrieve feature online store from. If not set,
                location set in aiplatform.init will be used.
            credentials:
                Custom credentials to use to retrieve this feature online store.
                Overrides credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=name,
        )
        self._gca_resource = self._get_gca_resource(resource_name=name)

    @classmethod
    @base.optional_sync()
    def create_bigtable_store(
        cls,
        name: str,
        min_node_count: Optional[int] = 1,
        max_node_count: Optional[int] = 1,
        cpu_utilization_target: Optional[int] = 50,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = None,
        create_request_timeout: Optional[float] = None,
        sync: bool = True,
    ) -> "FeatureOnlineStore":
        """Creates a Bigtable online store.

        Example Usage:

            my_fos = vertexai.preview.FeatureOnlineStore.create_bigtable_store('my_fos')

        Args:
            name: The name of the feature online store.
            min_node_count:
                The minimum number of Bigtable nodes to scale down to.  Must be
                greater than or equal to 1.
            max_node_count:
                The maximum number of Bigtable nodes to scale up to.  Must
                satisfy min_node_count <= max_node_count <= (10 *
                min_node_count).
            cpu_utilization_target:
                A percentage of the cluster's CPU capacity. Can be from 10% to
                80%. When a cluster's CPU utilization exceeds the target that
                you have set, Bigtable immediately adds nodes to the cluster.
                When CPU utilization is substantially lower than the target,
                Bigtable removes nodes. If not set will default to 50%.
            labels:
                The labels with user-defined metadata to organize your feature
                online store. Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. See https://goo.gl/xmQnxf
                for more information on and examples of labels. No more than 64
                user labels can be associated with one feature online store
                (System labels are excluded)." System reserved label keys are
                prefixed with "aiplatform.googleapis.com/" and are immutable.
            project:
                Project to create feature online store in. If unset, the project
                set in aiplatform.init will be used.
            location:
                Location to create feature online store in. If not set, location
                set in aiplatform.init will be used.
            credentials:
                Custom credentials to use to create this feature online store.
                Overrides credentials set in aiplatform.init.
            request_metadata:
                Strings which should be sent along with the request as metadata.
            create_request_timeout:
                The timeout for the create request in seconds.
            sync:
                Whether to execute this creation synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.

        Returns:
            FeatureOnlineStore - the FeatureOnlineStore resource object.
        """

        if min_node_count < 1:
            raise ValueError("min_node_count must be greater than or equal to 1")

        if max_node_count < min_node_count:
            raise ValueError(
                "max_node_count must be greater than or equal to min_node_count"
            )
        elif 10 * min_node_count < max_node_count:
            raise ValueError(
                "max_node_count must be less than or equal to 10 * min_node_count"
            )

        if cpu_utilization_target < 10 or cpu_utilization_target > 80:
            raise ValueError("cpu_utilization_target must be between 10 and 80")

        gapic_feature_online_store = gca_feature_online_store.FeatureOnlineStore(
            bigtable=gca_feature_online_store.FeatureOnlineStore.Bigtable(
                auto_scaling=gca_feature_online_store.FeatureOnlineStore.Bigtable.AutoScaling(
                    min_node_count=min_node_count,
                    max_node_count=max_node_count,
                    cpu_utilization_target=cpu_utilization_target,
                ),
            ),
        )

        if labels:
            utils.validate_labels(labels)
            gapic_feature_online_store.labels = labels

        if request_metadata is None:
            request_metadata = ()

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        create_online_store_lro = api_client.create_feature_online_store(
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            feature_online_store=gapic_feature_online_store,
            feature_online_store_id=name,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, create_online_store_lro)

        created_online_store = create_online_store_lro.result()

        _LOGGER.log_create_complete(cls, created_online_store, "feature_online_store")

        online_store_obj = cls(
            name=created_online_store.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return online_store_obj

    @classmethod
    @base.optional_sync()
    def create_optimized_store(
        cls,
        name: str,
        enable_private_service_connect: bool = False,
        project_allowlist: Optional[Sequence[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = None,
        create_request_timeout: Optional[float] = None,
        sync: bool = True,
    ) -> "FeatureOnlineStore":
        """Creates an Optimized online store.

        Example Usage:

            my_fos = vertexai.preview.FeatureOnlineStore.create_optimized_store('my_fos')

        Args:
            name: The name of the feature online store.
            enable_private_service_connect (bool):
                Optional. If true, expose the optimized online store
                via private service connect. Otherwise the optimized online
                store will be accessible through public endpoint
            project_allowlist (MutableSequence[str]):
                A list of Projects from which the forwarding
                rule will target the service attachment. Only needed when
                enable_private_service_connect is set to true.
            labels:
                The labels with user-defined metadata to organize your feature
                online store. Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. See https://goo.gl/xmQnxf
                for more information on and examples of labels. No more than 64
                user labels can be associated with one feature online store
                (System labels are excluded)." System reserved label keys are
                prefixed with "aiplatform.googleapis.com/" and are immutable.
            project:
                Project to create feature online store in. If unset, the project
                set in aiplatform.init will be used.
            location:
                Location to create feature online store in. If not set, location
                set in aiplatform.init will be used.
            credentials:
                Custom credentials to use to create this feature online store.
                Overrides credentials set in aiplatform.init.
            request_metadata:
                Strings which should be sent along with the request as metadata.
            create_request_timeout:
                The timeout for the create request in seconds.
            sync:
                Whether to execute this creation synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.

        Returns:
            FeatureOnlineStore - the FeatureOnlineStore resource object.
        """
        if enable_private_service_connect:
            raise ValueError("private_service_connect is not supported")
        else:
            dedicated_serving_endpoint = (
                gca_feature_online_store.FeatureOnlineStore.DedicatedServingEndpoint()
            )

        gapic_feature_online_store = gca_feature_online_store.FeatureOnlineStore(
            optimized=gca_feature_online_store.FeatureOnlineStore.Optimized(),
            dedicated_serving_endpoint=dedicated_serving_endpoint,
        )

        if labels:
            utils.validate_labels(labels)
            gapic_feature_online_store.labels = labels

        if request_metadata is None:
            request_metadata = ()

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        create_online_store_lro = api_client.create_feature_online_store(
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            feature_online_store=gapic_feature_online_store,
            feature_online_store_id=name,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, create_online_store_lro)

        created_online_store = create_online_store_lro.result()

        _LOGGER.log_create_complete(cls, created_online_store, "feature_online_store")

        online_store_obj = cls(
            name=created_online_store.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return online_store_obj

    @base.optional_sync()
    def delete(self, force: bool = False, sync: bool = True) -> None:
        """Deletes this online store.

        WARNING: This deletion is permanent.

        Args:
            force:
                If set to True, all feature views under this online store will
                be deleted prior to online store deletion. Otherwise, deletion
                will only succeed if the online store has no FeatureViews.
            sync:
                Whether to execute this deletion synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.
        """

        lro = getattr(self.api_client, self._delete_method)(
            name=self.resource_name,
            force=force,
        )
        _LOGGER.log_delete_with_lro(self, lro)
        lro.result()
        _LOGGER.log_delete_complete(self)

    @property
    def feature_online_store_type(self) -> FeatureOnlineStoreType:
        if self._gca_resource.bigtable:
            return FeatureOnlineStoreType.BIGTABLE
        # Optimized is an empty proto, so self._gca_resource.optimized is always false.
        elif hasattr(self.gca_resource, "optimized"):
            return FeatureOnlineStoreType.OPTIMIZED
        else:
            raise ValueError(
                f"Online store does not have type or is unsupported by SDK: {self._gca_resource}."
            )

    @property
    def labels(self) -> Dict[str, str]:
        return self._gca_resource.labels

    @base.optional_sync()
    def create_feature_view_from_big_query(
        self,
        name: str,
        big_query_source: FeatureView.BigQuerySource,
        labels: Optional[Dict[str, str]] = None,
        sync_config: Optional[str] = None,
        index_config: Optional[IndexConfig] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = None,
        create_request_timeout: Optional[float] = None,
        sync: bool = True,
    ) -> "FeatureView":
        """Creates a FeatureView from a BigQuery source.

        Example Usage:
        ```
        existing_fos = FeatureOnlineStore('my_fos')
        new_fv = existing_fos.create_feature_view_from_bigquery(
                'my_fos',
                BigQuerySource(
                    uri='bq://my-proj/dataset/table',
                    entity_id_columns=['entity_id'],
                )
        )
        # Example for how to create an embedding FeatureView.
        embedding_fv = existing_fos.create_feature_view_from_bigquery(
                'my_fos',
                BigQuerySource(
                    uri='bq://my-proj/dataset/table',
                    entity_id_columns=['entity_id'],
                )
                index_config=IndexConfig(
                    embedding_column="embedding",
                    filter_column=["currency_code", "gender",
                    crowding_column="crowding",
                    dimentions=1536,
                    distance_measure_type=DistanceMeasureType.SQUARED_L2_DISTANCE,
                    algorithm_config=TreeAhConfig(),
                )
            )
        ```
        Args:
            name: The name of the feature view.
            big_query_source:
                The BigQuery source to load data from when a feature view sync
                runs.
            labels:
                The labels with user-defined metadata to organize your
                FeatureViews.

                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters,
                numeric characters, underscores and dashes. International
                characters are allowed.

                See https://goo.gl/xmQnxf for more information on and examples
                of labels. No more than 64 user labels can be associated with
                one FeatureOnlineStore(System labels are excluded)." System
                reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            sync_config:
                Configures when data is to be synced/updated for this
                FeatureView. At the end of the sync the latest feature values
                for each entity ID of this FeatureView are made ready for online
                serving. Example format: "TZ=America/New_York 0 9 * * *" (sync
                daily at 9 AM EST).
            project:
                Project to create feature view in. If unset, the project set in
                aiplatform.init will be used.
            location:
                Location to create feature view in. If not set, location set in
                aiplatform.init will be used.
            credentials:
                Custom credentials to use to create this feature view.
                Overrides credentials set in aiplatform.init.
            request_metadata:
                Strings which should be sent along with the request as metadata.
            create_request_timeout:
                The timeout for the create request in seconds.
            sync:
                Whether to execute this creation synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.

        Returns:
            FeatureView - the FeatureView resource object.
        """
        if not big_query_source:
            raise ValueError("Please specify valid big_query_source.")
        elif not big_query_source.uri:
            raise ValueError("Please specify URI in big_query_source.")
        elif not big_query_source.entity_id_columns:
            raise ValueError("Please specify entity ID columns in big_query_source.")

        gapic_feature_view = gca_feature_view.FeatureView(
            big_query_source=gca_feature_view.FeatureView.BigQuerySource(
                uri=big_query_source.uri,
                entity_id_columns=big_query_source.entity_id_columns,
            ),
            sync_config=gca_feature_view.FeatureView.SyncConfig(cron=sync_config)
            if sync_config
            else None,
        )

        if labels:
            utils.validate_labels(labels)
            gapic_feature_view.labels = labels

        if request_metadata is None:
            request_metadata = ()

        if index_config:
            gapic_feature_view.index_config = gca_feature_view.FeatureView.IndexConfig(
                index_config.as_dict()
            )

        api_client = self.__class__._instantiate_client(
            location=location, credentials=credentials
        )

        create_feature_view_lro = api_client.create_feature_view(
            parent=self.resource_name,
            feature_view=gapic_feature_view,
            feature_view_id=name,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(FeatureView, create_feature_view_lro)

        created_feature_view = create_feature_view_lro.result()

        _LOGGER.log_create_complete(FeatureView, created_feature_view, "feature_view")

        feature_view_obj = FeatureView(
            name=created_feature_view.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return feature_view_obj
