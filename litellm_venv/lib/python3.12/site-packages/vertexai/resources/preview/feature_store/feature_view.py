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

from dataclasses import dataclass
import re
from typing import List, Optional
from google.cloud.aiplatform import initializer
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    feature_view_sync as gca_feature_view_sync,
    feature_view as gca_feature_view,
    feature_online_store_service as fos_service,
)
import vertexai.resources.preview.feature_store.utils as fv_utils

_LOGGER = base.Logger(__name__)


class FeatureView(base.VertexAiResourceNounWithFutureManager):
    """Class for managing Feature View resources."""

    client_class = utils.FeatureOnlineStoreAdminClientWithOverride

    _resource_noun = "featureViews"
    _getter_method = "get_feature_view"
    _list_method = "list_feature_views"
    _delete_method = "delete_feature_view"
    _parse_resource_name_method = "parse_feature_view_path"
    _format_resource_name_method = "feature_view_path"
    _gca_resource: gca_feature_view.FeatureView
    _online_store_client: utils.FeatureOnlineStoreClientWithOverride

    def __init__(
        self,
        name: str,
        feature_online_store_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing managed feature view.

        Args:
            name:
                The resource name
                (`projects/.../locations/.../featureOnlineStores/.../featureViews/...`)
                or ID.
            feature_online_store_id:
                The feature online store ID. Must be passed in if name is an ID
                and not a resource path.
            project:
                Project to retrieve the feature view from. If unset, the project
                set in aiplatform.init will be used.
            location:
                Location to retrieve the feature view from. If not set, location
                set in aiplatform.init will be used.
            credentials:
                Custom credentials to use to retrieve this feature view.
                Overrides credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=name,
        )

        if re.fullmatch(
            r"projects/.+/locations/.+/featureOnlineStores/.+/featureViews/.+",
            name,
        ):
            feature_view = name
        else:
            from .feature_online_store import FeatureOnlineStore

            # Construct the feature view path using feature online store ID if
            # only the feature view ID is provided.
            if not feature_online_store_id:
                raise ValueError(
                    "Since feature view is not provided as a path, please specify"
                    + " feature_online_store_id."
                )

            feature_online_store_path = utils.full_resource_name(
                resource_name=feature_online_store_id,
                resource_noun=FeatureOnlineStore._resource_noun,
                parse_resource_name_method=FeatureOnlineStore._parse_resource_name,
                format_resource_name_method=FeatureOnlineStore._format_resource_name,
            )

            feature_view = f"{feature_online_store_path}/featureViews/{name}"

        self._gca_resource = self._get_gca_resource(resource_name=feature_view)

    @property
    def _get_online_store_client(self) -> utils.FeatureOnlineStoreClientWithOverride:
        if getattr(self, "_online_store_client", None):
            return self._online_store_client

        fos_name = fv_utils.get_feature_online_store_name(self.resource_name)
        from .feature_online_store import FeatureOnlineStore

        fos = FeatureOnlineStore(name=fos_name)

        if fos._gca_resource.bigtable.auto_scaling:
            # This is Bigtable online store.
            _LOGGER.info(f"Connecting to Bigtable online store name {fos_name}")
            self._online_store_client = initializer.global_config.create_client(
                client_class=utils.FeatureOnlineStoreClientWithOverride,
                credentials=self.credentials,
                location_override=self.location,
            )
            return self._online_store_client

        # From here, optimized serving.
        if not fos._gca_resource.dedicated_serving_endpoint.public_endpoint_domain_name:
            raise fv_utils.PublicEndpointNotFoundError(
                "Public endpoint is not created yet for the optimized online store:"
                f"{fos_name}. Please run sync and wait for it to complete."
            )

        _LOGGER.info(
            f"Public endpoint for the optimized online store {fos_name} is"
            f" {fos._gca_resource.dedicated_serving_endpoint.public_endpoint_domain_name}"
        )
        self._online_store_client = initializer.global_config.create_client(
            client_class=utils.FeatureOnlineStoreClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
            prediction_client=True,
            api_path_override=fos._gca_resource.dedicated_serving_endpoint.public_endpoint_domain_name,
        )
        return self._online_store_client

    @classmethod
    def list(
        cls,
        feature_online_store_id: str,
        filter: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["FeatureView"]:
        """List all feature view under feature_online_store_id.

        Example Usage:
        ```
        feature_views = vertexai.preview.FeatureView.list(
            feature_online_store_id="my_fos",
            filter=labels.label_key=label_value)
        ```
        Args:
            feature_online_store_id:
                Parentfeature online store ID.
            filter:
                Filter to apply on the returned feature online store.
            credentials:
                Custom credentials to use to get a list of feature views.
                Overrides credentials set in aiplatform.init.

        Returns:
            List[FeatureView] - list of FeatureView resource object.
        """
        from .feature_online_store import FeatureOnlineStore

        fos = FeatureOnlineStore(name=feature_online_store_id)
        return cls._list(
            filter=filter, credentials=credentials, parent=fos.resource_name
        )

    @base.optional_sync()
    def delete(self, sync: bool = True) -> None:
        """Deletes this feature view.

        WARNING: This deletion is permanent.

        Args:
            sync:
                Whether to execute this deletion synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.
        """
        lro = getattr(self.api_client, self._delete_method)(name=self.resource_name)
        _LOGGER.log_delete_with_lro(self, lro)
        lro.result()
        _LOGGER.log_delete_complete(self)

    def sync(self) -> "FeatureViewSync":
        """Starts an on-demand Sync for the FeatureView.

        Args: None

        Returns:
            "FeatureViewSync" - FeatureViewSync instance
        """
        sync_method = getattr(self.api_client, self.FeatureViewSync.sync_method())

        sync_request = {
            "feature_view": self.resource_name,
        }
        sync_response = sync_method(request=sync_request)

        return self.FeatureViewSync(name=sync_response.feature_view_sync)

    def get_sync(self, name) -> "FeatureViewSync":
        """Gets the FeatureViewSync resource for the given name.

        Args:
            name: The resource ID

        Returns:
            "FeatureViewSync" - FeatureViewSync instance
        """
        feature_view_path = self.resource_name
        feature_view_sync = f"{feature_view_path}/featureViewSyncs/{name}"
        return self.FeatureViewSync(name=feature_view_sync)

    def list_syncs(
        self,
        filter: Optional[str] = None,
    ) -> List["FeatureViewSync"]:
        """List all feature view under this FeatureView.

        Args:
            parent_resource_name: Fully qualified name of the parent FeatureView
              resource.
            filter: Filter to apply on the returned feature online store.

        Returns:
            List[FeatureViewSync] - list of FeatureViewSync resource object.
        """

        return self.FeatureViewSync._list(
            filter=filter, credentials=self.credentials, parent=self.resource_name
        )

    def read(
        self,
        key: List[str],
        request_timeout: Optional[float] = None,
    ) -> fv_utils.FeatureViewReadResponse:
        """Read the feature values from FeatureView.

          Example Usage:
          ```
          data = vertexai.preview.FeatureView(
              name='feature_view_name', feature_online_store_id='fos_name')
            .read(key=[12345, 6789])
            .to_dict()
          ```
        Args:
            key: The request key to read feature values for.

        Returns:
            "FeatureViewReadResponse" - FeatureViewReadResponse object. It is
            intermediate class that can be further converted by to_dict() or
            to_proto()
        """
        self.wait()
        response = self._get_online_store_client.fetch_feature_values(
            feature_view=self.resource_name,
            data_key=fos_service.FeatureViewDataKey(
                composite_key=fos_service.FeatureViewDataKey.CompositeKey(parts=key)
            ),
            timeout=request_timeout,
        )
        return fv_utils.FeatureViewReadResponse(response)

    def search(
        self,
        entity_id: Optional[str] = None,
        embedding_value: Optional[List[float]] = None,
        neighbor_count: Optional[int] = None,
        string_filters: Optional[
            List[fos_service.NearestNeighborQuery.StringFilter]
        ] = None,
        per_crowding_attribute_neighbor_count: Optional[int] = None,
        return_full_entity: bool = False,
        approximate_neighbor_candidates: Optional[int] = None,
        leaf_nodes_search_fraction: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> fv_utils.SearchNearestEntitiesResponse:
        """Search the nearest entities from FeatureView.

        Example Usage:
        ```
          data = vertexai.preview.FeatureView(
              name='feature_view_name', feature_online_store_id='fos_name')
            .search(entity_id='sample_entity')
            .to_dict()
        ```
        Args:
            entity_id: The entity id whose similar entities should be searched
              for.
            embedding_value: The embedding vector that be used for similar
              search.
            neighbor_count: The number of similar entities to be retrieved
              from feature view for each query.
            string_filters: The list of string filters.
            per_crowding_attribute_neighbor_count: Crowding is a constraint on a
            neighbor list produced by nearest neighbor search requiring that
              no more than sper_crowding_attribute_neighbor_count of the k
              neighbors returned have the same value of crowding_attribute.
              It's used for improving result diversity.
            return_full_entity: If true, return full entities including the
              features other than embeddings.
            approximate_neighbor_candidates: The number of neighbors to find via
              approximate search before exact reordering is performed; if set,
              this value must be > neighbor_count.
            leaf_nodes_search_fraction: The fraction of the number of leaves to
              search, set at query time allows user to tune search performance.
              This value increase result in both search accuracy and latency
              increase. The value should be between 0.0 and 1.0.

        Returns:
            "SearchNearestEntitiesResponse" - SearchNearestEntitiesResponse
            object. It is intermediate class that can be further converted by
            to_dict() or to_proto()
        """
        self.wait()
        if entity_id:
            embedding = None
        elif embedding_value:
            embedding = fos_service.NearestNeighborQuery.Embedding(
                value=embedding_value
            )
        else:
            raise ValueError(
                f"Either entity_id or embedding_value needs to be provided for"
                f" search."
            )
        response = self._get_online_store_client.search_nearest_entities(
            request=fos_service.SearchNearestEntitiesRequest(
                feature_view=self.resource_name,
                query=fos_service.NearestNeighborQuery(
                    entity_id=entity_id,
                    embedding=embedding,
                    neighbor_count=neighbor_count,
                    string_filters=string_filters,
                    per_crowding_attribute_neighbor_count=per_crowding_attribute_neighbor_count,  # pylint: disable=line-too-long
                    parameters=fos_service.NearestNeighborQuery.Parameters(
                        approximate_neighbor_candidates=approximate_neighbor_candidates,
                        leaf_nodes_search_fraction=leaf_nodes_search_fraction,
                    ),
                ),
                return_full_entity=return_full_entity,
            ),
            timeout=request_timeout,
        )
        return fv_utils.SearchNearestEntitiesResponse(response)

    @dataclass
    class BigQuerySource:
        uri: str
        entity_id_columns: List[str]

    class FeatureViewSync(base.VertexAiResourceNounWithFutureManager):
        """Class for managing Feature View Sync resources."""

        client_class = utils.FeatureOnlineStoreAdminClientWithOverride

        _resource_noun = "featureViewSyncs"
        _getter_method = "get_feature_view_sync"
        _list_method = "list_feature_view_syncs"
        _delete_method = "delete_feature_view"
        _sync_method = "sync_feature_view"
        _parse_resource_name_method = "parse_feature_view_sync_path"
        _format_resource_name_method = "feature_view_sync_path"
        _gca_resource: gca_feature_view_sync.FeatureViewSync

        def __init__(
            self,
            name: str,
            project: Optional[str] = None,
            location: Optional[str] = None,
            credentials: Optional[auth_credentials.Credentials] = None,
        ):
            """Retrieves an existing managed feature view sync.

            Args:
                name: The resource name
                  (`projects/.../locations/.../featureOnlineStores/.../featureViews/.../featureViewSyncs/...`)
                project: Project to retrieve the feature view from. If unset, the
                  project set in aiplatform.init will be used.
                location: Location to retrieve the feature view from. If not set,
                  location set in aiplatform.init will be used.
                credentials: Custom credentials to use to retrieve this feature view.
                  Overrides credentials set in aiplatform.init.
            """
            super().__init__(
                project=project,
                location=location,
                credentials=credentials,
                resource_name=name,
            )

            if not re.fullmatch(
                r"projects/.+/locations/.+/featureOnlineStores/.+/featureViews/.+/featureViewSyncs/.+",
                name,
            ):
                raise ValueError(
                    "name need to specify the fully qualified"
                    + " feature_view_sync resource path."
                )

            self._gca_resource = getattr(self.api_client, self._getter_method)(
                name=name, retry=base._DEFAULT_RETRY
            )

        @classmethod
        def sync_method(cls) -> str:
            """Returns the sync method."""
            return cls._sync_method
