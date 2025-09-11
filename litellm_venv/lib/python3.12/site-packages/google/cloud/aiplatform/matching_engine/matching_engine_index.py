# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from typing import Dict, List, Optional, Sequence, Tuple

from google.auth import credentials as auth_credentials
from google.protobuf import field_mask_pb2
from google.cloud.aiplatform import base
from google.cloud.aiplatform.compat.types import (
    index_service as gca_index_service,
    matching_engine_deployed_index_ref as gca_matching_engine_deployed_index_ref,
    matching_engine_index as gca_matching_engine_index,
    encryption_spec as gca_encryption_spec,
)
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.matching_engine import (
    matching_engine_index_config,
)
from google.cloud.aiplatform import utils

_LOGGER = base.Logger(__name__)


class MatchingEngineIndex(base.VertexAiResourceNounWithFutureManager):
    """Matching Engine index resource for Vertex AI."""

    client_class = utils.IndexClientWithOverride

    _resource_noun = "indexes"
    _getter_method = "get_index"
    _list_method = "list_indexes"
    _delete_method = "delete_index"
    _parse_resource_name_method = "parse_index_path"
    _format_resource_name_method = "index_path"

    def __init__(
        self,
        index_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing index given an index name or ID.

        Example Usage:

            my_index = aiplatform.MatchingEngineIndex(
                index_name='projects/123/locations/us-central1/indexes/my_index_id'
            )
            or
            my_index = aiplatform.MatchingEngineIndex(
                index_name='my_index_id'
            )

        Args:
            index_name (str):
                Required. A fully-qualified index resource name or a index ID.
                Example: "projects/123/locations/us-central1/indexes/my_index_id"
                or "my_index_id" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve index from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve index from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Index. Overrides
                credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=index_name,
        )
        self._gca_resource = self._get_gca_resource(resource_name=index_name)

    @property
    def description(self) -> str:
        """Description of the index."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.description

    @classmethod
    @base.optional_sync()
    def _create(
        cls,
        display_name: str,
        contents_delta_uri: Optional[str] = None,
        config: matching_engine_index_config.MatchingEngineIndexConfig = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync: bool = True,
        index_update_method: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
    ) -> "MatchingEngineIndex":
        """Creates a MatchingEngineIndex resource.

        Args:
            display_name (str):
                Required. The display name of the Index.
                The name can be up to 128 characters long and
                can be consist of any UTF-8 characters.
            contents_delta_uri (str):
                Optional. Allows inserting, updating  or deleting the contents of the Matching Engine Index.
                The string must be a valid Google Cloud Storage directory path. If this
                field is set when calling IndexService.UpdateIndex, then no other
                Index field can be  also updated as part of the same call.
                The expected structure and format of the files this URI points to is
                described at
                https://cloud.google.com/vertex-ai/docs/vector-search/setup/format-structure
            config (matching_engine_index_config.MatchingEngineIndexConfig):
                Required. The configuration with regard to the algorithms used for efficient search.
            description (str):
                Optional. The description of the Index.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your Index.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one
                Index(System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            project (str):
                Optional. Project to create EntityType in. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to create EntityType in. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create EntityTypes. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            sync (bool):
                Optional. Whether to execute this creation synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            index_update_method (str):
                Optional. The update method to use with this index. Choose
                stream_update or batch_update. If not set, batch update will be
                used by default.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the index. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this index and all sub-resources of this index will be
                secured by this key.
                The key needs to be in the same region as where the index is
                created.
            create_request_timeout (float):
                Optional. The timeout for the request in seconds.

        Returns:
            MatchingEngineIndex - Index resource object

        """
        index_update_method_enum = None
        if index_update_method in _INDEX_UPDATE_METHOD_TO_ENUM_VALUE:
            index_update_method_enum = _INDEX_UPDATE_METHOD_TO_ENUM_VALUE[
                index_update_method
            ]

        metadata = {"config": config.as_dict()}
        if contents_delta_uri:
            metadata = {
                "config": config.as_dict(),
                "contentsDeltaUri": contents_delta_uri,
            }

        gapic_index = gca_matching_engine_index.Index(
            display_name=display_name,
            description=description,
            metadata=metadata,
            index_update_method=index_update_method_enum,
        )

        if encryption_spec_key_name:
            encryption_spec = gca_encryption_spec.EncryptionSpec(
                kms_key_name=encryption_spec_key_name
            )
            gapic_index.encryption_spec = encryption_spec

        if labels:
            utils.validate_labels(labels)
            gapic_index.labels = labels

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        create_lro = api_client.create_index(
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            index=gapic_index,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, create_lro)

        created_index = create_lro.result(timeout=None)

        _LOGGER.log_create_complete(cls, created_index, "index")

        index_obj = cls(
            index_name=created_index.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return index_obj

    def update_metadata(
        self,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        update_request_timeout: Optional[float] = None,
    ) -> "MatchingEngineIndex":
        """Updates the metadata for this index.

        Args:
            display_name (str):
                Optional. The display name of the Index.
                The name can be up to 128 characters long and
                can be consist of any UTF-8 characters.
            description (str):
                Optional. The description of the Index.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your Indexs.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one Index
                (System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            update_request_timeout (float):
                Optional. The timeout for the request in seconds.

        Returns:
            MatchingEngineIndex - The updated index resource object.
        """

        self.wait()

        update_mask = list()

        if labels:
            utils.validate_labels(labels)
            update_mask.append("labels")

        if display_name is not None:
            update_mask.append("display_name")

        if description is not None:
            update_mask.append("description")

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)

        gapic_index = gca_matching_engine_index.Index(
            name=self.resource_name,
            display_name=display_name,
            description=description,
            labels=labels,
        )

        _LOGGER.log_action_start_against_resource(
            "Updating",
            "index",
            self,
        )

        update_lro = self.api_client.update_index(
            index=gapic_index,
            update_mask=update_mask,
            metadata=request_metadata,
            timeout=update_request_timeout,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Update", "index", self.__class__, update_lro
        )

        self._gca_resource = update_lro.result()

        _LOGGER.log_action_completed_against_resource("index", "Updated", self)

        return self

    def update_embeddings(
        self,
        contents_delta_uri: str,
        is_complete_overwrite: Optional[bool] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        update_request_timeout: Optional[float] = None,
    ) -> "MatchingEngineIndex":
        """Updates the embeddings for this index.

        Args:
            contents_delta_uri (str):
                Required. Allows inserting, updating  or deleting the contents of the Matching Engine Index.
                The string must be a valid Google Cloud Storage directory path. If this
                field is set when calling IndexService.UpdateIndex, then no other
                Index field can be  also updated as part of the same call.
                The expected structure and format of the files this URI points to is
                described at
                https://cloud.google.com/vertex-ai/docs/vector-search/setup/format-structure
            is_complete_overwrite (bool):
                Optional. If this field is set together with contentsDeltaUri when calling IndexService.UpdateIndex,
                then existing content of the Index will be replaced by the data from the contentsDeltaUri.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            update_request_timeout (float):
                Optional. The timeout for the request in seconds.

        Returns:
            MatchingEngineIndex - The updated index resource object.
        """

        self.wait()

        update_mask = list()

        if contents_delta_uri or is_complete_overwrite:
            update_mask.append("metadata")

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)

        gapic_index = gca_matching_engine_index.Index(
            name=self.resource_name,
            metadata={
                "contentsDeltaUri": contents_delta_uri,
                "isCompleteOverwrite": is_complete_overwrite,
            },
        )

        _LOGGER.log_action_start_against_resource(
            "Updating",
            "index",
            self,
        )

        update_lro = self.api_client.update_index(
            index=gapic_index,
            update_mask=update_mask,
            metadata=request_metadata,
            timeout=update_request_timeout,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Update", "index", self.__class__, update_lro
        )

        self._gca_resource = update_lro.result(timeout=None)

        _LOGGER.log_action_completed_against_resource("index", "Updated", self)

        return self

    @property
    def deployed_indexes(
        self,
    ) -> List[gca_matching_engine_deployed_index_ref.DeployedIndexRef]:
        """Returns a list of deployed index references that originate from this index.

        Returns:
            List[gca_matching_engine_deployed_index_ref.DeployedIndexRef] - Deployed index references
        """

        self.wait()

        return self._gca_resource.deployed_indexes

    @classmethod
    def create_tree_ah_index(
        cls,
        display_name: str,
        contents_delta_uri: Optional[str] = None,
        dimensions: int = None,
        approximate_neighbors_count: int = None,
        leaf_node_embedding_count: Optional[int] = None,
        leaf_nodes_to_search_percent: Optional[float] = None,
        distance_measure_type: Optional[
            matching_engine_index_config.DistanceMeasureType
        ] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync: bool = True,
        index_update_method: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
        shard_size: Optional[str] = None,
    ) -> "MatchingEngineIndex":
        """Creates a MatchingEngineIndex resource that uses the tree-AH algorithm.

        Example Usage:

            my_index = aiplatform.Index.create_tree_ah_index(
                display_name="my_display_name",
                contents_delta_uri="gs://my_bucket/embeddings",
                dimensions=1,
                approximate_neighbors_count=150,
                distance_measure_type="SQUARED_L2_DISTANCE",
                leaf_node_embedding_count=100,
                leaf_nodes_to_search_percent=50,
                description="my description",
                labels={ "label_name": "label_value" },
            )

        Args:
            display_name (str):
                Required. The display name of the Index.
                The name can be up to 128 characters long and
                can be consist of any UTF-8 characters.
            contents_delta_uri (str):
                Optional. Allows inserting, updating  or deleting the contents of the Matching Engine Index.
                The string must be a valid Google Cloud Storage directory path. If this
                field is set when calling IndexService.UpdateIndex, then no other
                Index field can be  also updated as part of the same call.
                The expected structure and format of the files this URI points to is
                described at
                https://cloud.google.com/vertex-ai/docs/vector-search/setup/format-structure
            dimensions (int):
                Required. The number of dimensions of the input vectors.
            approximate_neighbors_count (int):
                Required. The default number of neighbors to find via approximate search before exact reordering is
                performed. Exact reordering is a procedure where results returned by an
                approximate search algorithm are reordered via a more expensive distance computation.
            leaf_node_embedding_count (int):
                Optional. Number of embeddings on each leaf node. The default value is 1000 if not set.
            leaf_nodes_to_search_percent (float):
                Optional. The default percentage of leaf nodes that any query may be searched. Must be in
                range 1-100, inclusive. The default value is 10 (means 10%) if not set.
            distance_measure_type (matching_engine_index_config.DistanceMeasureType):
                Optional. The distance measure used in nearest neighbor search.
            description (str):
                Optional. The description of the Index.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your Index.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one
                Index(System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            project (str):
                Optional. Project to create EntityType in. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to create EntityType in. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create EntityTypes. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            sync (bool):
                Optional. Whether to execute this creation synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            index_update_method (str):
                Optional. The update method to use with this index. Choose
                STREAM_UPDATE or BATCH_UPDATE. If not set, batch update will be
                used by default.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the index. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this index and all sub-resources of this index will be
                secured by this key.
                The key needs to be in the same region as where the index is
                created.
            create_request_timeout (float):
                Optional. The timeout for the request in seconds.
            shard_size (str):
                Optional. The size of each shard. Index will get resharded
                based on specified shard size. During serving, each shard will
                be served on a separate node and will scale independently.

                Choose one of the following:
                    SHARD_SIZE_SMALL
                    SHARD_SIZE_MEDIUM
                    SHARD_SIZE_LARGE


        Returns:
            MatchingEngineIndex - Index resource object

        """

        algorithm_config = matching_engine_index_config.TreeAhConfig(
            leaf_node_embedding_count=leaf_node_embedding_count,
            leaf_nodes_to_search_percent=leaf_nodes_to_search_percent,
        )

        config = matching_engine_index_config.MatchingEngineIndexConfig(
            dimensions=dimensions,
            algorithm_config=algorithm_config,
            approximate_neighbors_count=approximate_neighbors_count,
            distance_measure_type=distance_measure_type,
            shard_size=shard_size,
        )

        return cls._create(
            display_name=display_name,
            contents_delta_uri=contents_delta_uri,
            config=config,
            description=description,
            labels=labels,
            project=project,
            location=location,
            credentials=credentials,
            request_metadata=request_metadata,
            sync=sync,
            index_update_method=index_update_method,
            encryption_spec_key_name=encryption_spec_key_name,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    def create_brute_force_index(
        cls,
        display_name: str,
        contents_delta_uri: Optional[str] = None,
        dimensions: int = None,
        distance_measure_type: Optional[
            matching_engine_index_config.DistanceMeasureType
        ] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync: bool = True,
        index_update_method: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
        shard_size: Optional[str] = None,
    ) -> "MatchingEngineIndex":
        """Creates a MatchingEngineIndex resource that uses the brute force algorithm.

        Example Usage:

            my_index = aiplatform.Index.create_brute_force_index(
                display_name="my_display_name",
                contents_delta_uri="gs://my_bucket/embeddings",
                dimensions=1,
                approximate_neighbors_count=150,
                distance_measure_type="SQUARED_L2_DISTANCE",
                description="my description",
                labels={ "label_name": "label_value" },
            )

        Args:
            display_name (str):
                Required. The display name of the Index.
                The name can be up to 128 characters long and
                can be consist of any UTF-8 characters.
            contents_delta_uri (str):
                Optional. Allows inserting, updating  or deleting the contents of the Matching Engine Index.
                The string must be a valid Google Cloud Storage directory path. If this
                field is set when calling IndexService.UpdateIndex, then no other
                Index field can be  also updated as part of the same call.
                The expected structure and format of the files this URI points to is
                described at
                https://cloud.google.com/vertex-ai/docs/vector-search/setup/format-structure
            dimensions (int):
                Required. The number of dimensions of the input vectors.
            distance_measure_type (matching_engine_index_config.DistanceMeasureType):
                Optional. The distance measure used in nearest neighbor search.
            description (str):
                Optional. The description of the Index.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your Index.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one
                Index(System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            project (str):
                Optional. Project to create EntityType in. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to create EntityType in. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create EntityTypes. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            sync (bool):
                Optional. Whether to execute this creation synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            index_update_method (str):
                Optional. The update method to use with this index. Choose
                stream_update or batch_update. If not set, batch update will be
                used by default.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the index. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this index and all sub-resources of this index will be
                secured by this key.
                The key needs to be in the same region as where the index is
                created.
            create_request_timeout (float):
                Optional. The timeout for the request in seconds.
            shard_size (str):
                Optional. The size of each shard. Index will get resharded
                based on specified shard size. During serving, each shard will
                be served on a separate node and will scale independently.

                If not set, shard size is default to SHARD_SIZE_MEDIUM.

                Choose one of the following:
                    SHARD_SIZE_SMALL
                    SHARD_SIZE_MEDIUM
                    SHARD_SIZE_LARGE

        Returns:
            MatchingEngineIndex - Index resource object

        """

        algorithm_config = matching_engine_index_config.BruteForceConfig()

        config = matching_engine_index_config.MatchingEngineIndexConfig(
            dimensions=dimensions,
            algorithm_config=algorithm_config,
            distance_measure_type=distance_measure_type,
            shard_size=shard_size,
        )

        return cls._create(
            display_name=display_name,
            contents_delta_uri=contents_delta_uri,
            config=config,
            description=description,
            labels=labels,
            project=project,
            location=location,
            credentials=credentials,
            request_metadata=request_metadata,
            sync=sync,
            index_update_method=index_update_method,
            encryption_spec_key_name=encryption_spec_key_name,
            create_request_timeout=create_request_timeout,
        )

    def upsert_datapoints(
        self,
        datapoints: Sequence[gca_matching_engine_index.IndexDatapoint],
        update_mask: Optional[Sequence[str]] = None,
    ) -> "MatchingEngineIndex":
        """Upsert datapoints to this index.

        Args:
            datapoints (Sequence[gca_matching_engine_index.IndexDatapoint]):
                Required. Datapoints to be upserted to this index.
            update_mask (Sequence[str]):
                Optional. Update mask is used to specify the fields to be
                overwritten in the datapoints by the update. The fields
                specified in the update_mask are relative to each IndexDatapoint
                inside datapoints, not the full request.
                Updatable fields:
                    Use `all_restricts` to update both `restricts` and
                    `numeric_restricts`.

        Returns:
            MatchingEngineIndex - Index resource object

        """

        self.wait()

        _LOGGER.log_action_start_against_resource(
            "Upserting datapoints",
            "index",
            self,
        )

        self.api_client.upsert_datapoints(
            gca_index_service.UpsertDatapointsRequest(
                index=self.resource_name,
                datapoints=datapoints,
                update_mask=(
                    field_mask_pb2.FieldMask(paths=update_mask) if update_mask else None
                ),
            )
        )

        _LOGGER.log_action_completed_against_resource(
            "index", "Upserted datapoints", self
        )

        return self

    def remove_datapoints(
        self,
        datapoint_ids: Sequence[str],
    ) -> "MatchingEngineIndex":
        """Remove datapoints for this index.

        Args:
            datapoints_ids (Sequence[str]):
                Required. The list of datapoints ids to be deleted.

        Returns:
            MatchingEngineIndex - Index resource object
        """

        self.wait()

        _LOGGER.log_action_start_against_resource(
            "Removing datapoints",
            "index",
            self,
        )

        self.api_client.remove_datapoints(
            gca_index_service.RemoveDatapointsRequest(
                index=self.resource_name,
                datapoint_ids=datapoint_ids,
            )
        )

        _LOGGER.log_action_completed_against_resource(
            "index", "Removed datapoints", self
        )

        return self


_INDEX_UPDATE_METHOD_TO_ENUM_VALUE = {
    "STREAM_UPDATE": gca_matching_engine_index.Index.IndexUpdateMethod.STREAM_UPDATE,
    "BATCH_UPDATE": gca_matching_engine_index.Index.IndexUpdateMethod.BATCH_UPDATE,
}
