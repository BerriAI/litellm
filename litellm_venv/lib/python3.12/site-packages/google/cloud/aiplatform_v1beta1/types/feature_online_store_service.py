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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.aiplatform_v1beta1.types import featurestore_online_service
from google.protobuf import struct_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "FeatureViewDataFormat",
        "FeatureViewDataKey",
        "FetchFeatureValuesRequest",
        "FetchFeatureValuesResponse",
        "StreamingFetchFeatureValuesRequest",
        "StreamingFetchFeatureValuesResponse",
        "NearestNeighborQuery",
        "SearchNearestEntitiesRequest",
        "NearestNeighbors",
        "SearchNearestEntitiesResponse",
    },
)


class FeatureViewDataFormat(proto.Enum):
    r"""Format of the data in the Feature View.

    Values:
        FEATURE_VIEW_DATA_FORMAT_UNSPECIFIED (0):
            Not set. Will be treated as the KeyValue
            format.
        KEY_VALUE (1):
            Return response data in key-value format.
        PROTO_STRUCT (2):
            Return response data in proto Struct format.
    """
    FEATURE_VIEW_DATA_FORMAT_UNSPECIFIED = 0
    KEY_VALUE = 1
    PROTO_STRUCT = 2


class FeatureViewDataKey(proto.Message):
    r"""Lookup key for a feature view.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        key (str):
            String key to use for lookup.

            This field is a member of `oneof`_ ``key_oneof``.
        composite_key (google.cloud.aiplatform_v1beta1.types.FeatureViewDataKey.CompositeKey):
            The actual Entity ID will be composed from
            this struct. This should match with the way ID
            is defined in the FeatureView spec.

            This field is a member of `oneof`_ ``key_oneof``.
    """

    class CompositeKey(proto.Message):
        r"""ID that is comprised from several parts (columns).

        Attributes:
            parts (MutableSequence[str]):
                Parts to construct Entity ID. Should match
                with the same ID columns as defined in
                FeatureView in the same order.
        """

        parts: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    key: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="key_oneof",
    )
    composite_key: CompositeKey = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="key_oneof",
        message=CompositeKey,
    )


class FetchFeatureValuesRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreService.FetchFeatureValues][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreService.FetchFeatureValues].
    All the features under the requested feature view will be returned.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        id (str):
            Simple ID. The whole string will be used as
            is to identify Entity to fetch feature values
            for.

            This field is a member of `oneof`_ ``entity_id``.
        feature_view (str):
            Required. FeatureView resource format
            ``projects/{project}/locations/{location}/featureOnlineStores/{featureOnlineStore}/featureViews/{featureView}``
        data_key (google.cloud.aiplatform_v1beta1.types.FeatureViewDataKey):
            Optional. The request key to fetch feature
            values for.
        data_format (google.cloud.aiplatform_v1beta1.types.FeatureViewDataFormat):
            Optional. Response data format. If not set,
            [FeatureViewDataFormat.KEY_VALUE][google.cloud.aiplatform.v1beta1.FeatureViewDataFormat.KEY_VALUE]
            will be used.
        format_ (google.cloud.aiplatform_v1beta1.types.FetchFeatureValuesRequest.Format):
            Specify response data format. If not set, KeyValue format
            will be used. Deprecated. Use
            [FetchFeatureValuesRequest.data_format][google.cloud.aiplatform.v1beta1.FetchFeatureValuesRequest.data_format].
    """

    class Format(proto.Enum):
        r"""Format of the response data.

        Values:
            FORMAT_UNSPECIFIED (0):
                Not set. Will be treated as the KeyValue
                format.
            KEY_VALUE (1):
                Return response data in key-value format.
            PROTO_STRUCT (2):
                Return response data in proto Struct format.
        """
        _pb_options = {"deprecated": True}
        FORMAT_UNSPECIFIED = 0
        KEY_VALUE = 1
        PROTO_STRUCT = 2

    id: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="entity_id",
    )
    feature_view: str = proto.Field(
        proto.STRING,
        number=1,
    )
    data_key: "FeatureViewDataKey" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="FeatureViewDataKey",
    )
    data_format: "FeatureViewDataFormat" = proto.Field(
        proto.ENUM,
        number=7,
        enum="FeatureViewDataFormat",
    )
    format_: Format = proto.Field(
        proto.ENUM,
        number=5,
        enum=Format,
    )


class FetchFeatureValuesResponse(proto.Message):
    r"""Response message for
    [FeatureOnlineStoreService.FetchFeatureValues][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreService.FetchFeatureValues]

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        key_values (google.cloud.aiplatform_v1beta1.types.FetchFeatureValuesResponse.FeatureNameValuePairList):
            Feature values in KeyValue format.

            This field is a member of `oneof`_ ``format``.
        proto_struct (google.protobuf.struct_pb2.Struct):
            Feature values in proto Struct format.

            This field is a member of `oneof`_ ``format``.
        data_key (google.cloud.aiplatform_v1beta1.types.FeatureViewDataKey):
            The data key associated with this response. Will only be
            populated for
            [FeatureOnlineStoreService.StreamingFetchFeatureValues][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreService.StreamingFetchFeatureValues]
            RPCs.
    """

    class FeatureNameValuePairList(proto.Message):
        r"""Response structure in the format of key (feature name) and
        (feature) value pair.

        Attributes:
            features (MutableSequence[google.cloud.aiplatform_v1beta1.types.FetchFeatureValuesResponse.FeatureNameValuePairList.FeatureNameValuePair]):
                List of feature names and values.
        """

        class FeatureNameValuePair(proto.Message):
            r"""Feature name & value pair.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                value (google.cloud.aiplatform_v1beta1.types.FeatureValue):
                    Feature value.

                    This field is a member of `oneof`_ ``data``.
                name (str):
                    Feature short name.
            """

            value: featurestore_online_service.FeatureValue = proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="data",
                message=featurestore_online_service.FeatureValue,
            )
            name: str = proto.Field(
                proto.STRING,
                number=1,
            )

        features: MutableSequence[
            "FetchFeatureValuesResponse.FeatureNameValuePairList.FeatureNameValuePair"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="FetchFeatureValuesResponse.FeatureNameValuePairList.FeatureNameValuePair",
        )

    key_values: FeatureNameValuePairList = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="format",
        message=FeatureNameValuePairList,
    )
    proto_struct: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="format",
        message=struct_pb2.Struct,
    )
    data_key: "FeatureViewDataKey" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="FeatureViewDataKey",
    )


class StreamingFetchFeatureValuesRequest(proto.Message):
    r"""Request message for
    [FeatureOnlineStoreService.StreamingFetchFeatureValues][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreService.StreamingFetchFeatureValues].
    For the entities requested, all features under the requested feature
    view will be returned.

    Attributes:
        feature_view (str):
            Required. FeatureView resource format
            ``projects/{project}/locations/{location}/featureOnlineStores/{featureOnlineStore}/featureViews/{featureView}``
        data_keys (MutableSequence[google.cloud.aiplatform_v1beta1.types.FeatureViewDataKey]):

        data_format (google.cloud.aiplatform_v1beta1.types.FeatureViewDataFormat):
            Specify response data format. If not set,
            KeyValue format will be used.
    """

    feature_view: str = proto.Field(
        proto.STRING,
        number=1,
    )
    data_keys: MutableSequence["FeatureViewDataKey"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="FeatureViewDataKey",
    )
    data_format: "FeatureViewDataFormat" = proto.Field(
        proto.ENUM,
        number=3,
        enum="FeatureViewDataFormat",
    )


class StreamingFetchFeatureValuesResponse(proto.Message):
    r"""Response message for
    [FeatureOnlineStoreService.StreamingFetchFeatureValues][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreService.StreamingFetchFeatureValues].

    Attributes:
        status (google.rpc.status_pb2.Status):
            Response status. If OK, then
            [StreamingFetchFeatureValuesResponse.data][google.cloud.aiplatform.v1beta1.StreamingFetchFeatureValuesResponse.data]
            will be populated. Otherwise
            [StreamingFetchFeatureValuesResponse.data_keys_with_error][google.cloud.aiplatform.v1beta1.StreamingFetchFeatureValuesResponse.data_keys_with_error]
            will be populated with the appropriate data keys. The error
            only applies to the listed data keys - the stream will
            remain open for further
            [FeatureOnlineStoreService.StreamingFetchFeatureValuesRequest][]
            requests.
        data (MutableSequence[google.cloud.aiplatform_v1beta1.types.FetchFeatureValuesResponse]):

        data_keys_with_error (MutableSequence[google.cloud.aiplatform_v1beta1.types.FeatureViewDataKey]):

    """

    status: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )
    data: MutableSequence["FetchFeatureValuesResponse"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="FetchFeatureValuesResponse",
    )
    data_keys_with_error: MutableSequence["FeatureViewDataKey"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="FeatureViewDataKey",
    )


class NearestNeighborQuery(proto.Message):
    r"""A query to find a number of similar entities.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        entity_id (str):
            Optional. The entity id whose similar entities should be
            searched for. If embedding is set, search will use embedding
            instead of entity_id.

            This field is a member of `oneof`_ ``instance``.
        embedding (google.cloud.aiplatform_v1beta1.types.NearestNeighborQuery.Embedding):
            Optional. The embedding vector that be used
            for similar search.

            This field is a member of `oneof`_ ``instance``.
        neighbor_count (int):
            Optional. The number of similar entities to
            be retrieved from feature view for each query.
        string_filters (MutableSequence[google.cloud.aiplatform_v1beta1.types.NearestNeighborQuery.StringFilter]):
            Optional. The list of string filters.
        per_crowding_attribute_neighbor_count (int):
            Optional. Crowding is a constraint on a neighbor list
            produced by nearest neighbor search requiring that no more
            than sper_crowding_attribute_neighbor_count of the k
            neighbors returned have the same value of
            crowding_attribute. It's used for improving result
            diversity.
        parameters (google.cloud.aiplatform_v1beta1.types.NearestNeighborQuery.Parameters):
            Optional. Parameters that can be set to tune
            query on the fly.
    """

    class Embedding(proto.Message):
        r"""The embedding vector.

        Attributes:
            value (MutableSequence[float]):
                Optional. Individual value in the embedding.
        """

        value: MutableSequence[float] = proto.RepeatedField(
            proto.FLOAT,
            number=1,
        )

    class StringFilter(proto.Message):
        r"""String filter is used to search a subset of the entities by using
        boolean rules on string columns. For example: if a query specifies
        string filter with 'name = color, allow_tokens = {red, blue},
        deny_tokens = {purple}',' then that query will match entities that
        are red or blue, but if those points are also purple, then they will
        be excluded even if they are red/blue. Only string filter is
        supported for now, numeric filter will be supported in the near
        future.

        Attributes:
            name (str):
                Required. Column names in BigQuery that used
                as filters.
            allow_tokens (MutableSequence[str]):
                Optional. The allowed tokens.
            deny_tokens (MutableSequence[str]):
                Optional. The denied tokens.
        """

        name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        allow_tokens: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )
        deny_tokens: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=3,
        )

    class Parameters(proto.Message):
        r"""Parameters that can be overrided in each query to tune query
        latency and recall.

        Attributes:
            approximate_neighbor_candidates (int):
                Optional. The number of neighbors to find via approximate
                search before exact reordering is performed; if set, this
                value must be > neighbor_count.
            leaf_nodes_search_fraction (float):
                Optional. The fraction of the number of
                leaves to search, set at query time allows user
                to tune search performance. This value increase
                result in both search accuracy and latency
                increase. The value should be between 0.0 and
                1.0.
        """

        approximate_neighbor_candidates: int = proto.Field(
            proto.INT32,
            number=1,
        )
        leaf_nodes_search_fraction: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )

    entity_id: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="instance",
    )
    embedding: Embedding = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="instance",
        message=Embedding,
    )
    neighbor_count: int = proto.Field(
        proto.INT32,
        number=3,
    )
    string_filters: MutableSequence[StringFilter] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=StringFilter,
    )
    per_crowding_attribute_neighbor_count: int = proto.Field(
        proto.INT32,
        number=5,
    )
    parameters: Parameters = proto.Field(
        proto.MESSAGE,
        number=7,
        message=Parameters,
    )


class SearchNearestEntitiesRequest(proto.Message):
    r"""The request message for
    [FeatureOnlineStoreService.SearchNearestEntities][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreService.SearchNearestEntities].

    Attributes:
        feature_view (str):
            Required. FeatureView resource format
            ``projects/{project}/locations/{location}/featureOnlineStores/{featureOnlineStore}/featureViews/{featureView}``
        query (google.cloud.aiplatform_v1beta1.types.NearestNeighborQuery):
            Required. The query.
        return_full_entity (bool):
            Optional. If set to true, the full entities
            (including all vector values and metadata) of
            the nearest neighbors are returned; otherwise
            only entity id of the nearest neighbors will be
            returned. Note that returning full entities will
            significantly increase the latency and cost of
            the query.
    """

    feature_view: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: "NearestNeighborQuery" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="NearestNeighborQuery",
    )
    return_full_entity: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class NearestNeighbors(proto.Message):
    r"""Nearest neighbors for one query.

    Attributes:
        neighbors (MutableSequence[google.cloud.aiplatform_v1beta1.types.NearestNeighbors.Neighbor]):
            All its neighbors.
    """

    class Neighbor(proto.Message):
        r"""A neighbor of the query vector.

        Attributes:
            entity_id (str):
                The id of the similar entity.
            distance (float):
                The distance between the neighbor and the
                query vector.
            entity_key_values (google.cloud.aiplatform_v1beta1.types.FetchFeatureValuesResponse):
                The attributes of the neighbor, e.g. filters, crowding and
                metadata Note that full entities are returned only when
                "return_full_entity" is set to true. Otherwise, only the
                "entity_id" and "distance" fields are populated.
        """

        entity_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        distance: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )
        entity_key_values: "FetchFeatureValuesResponse" = proto.Field(
            proto.MESSAGE,
            number=3,
            message="FetchFeatureValuesResponse",
        )

    neighbors: MutableSequence[Neighbor] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=Neighbor,
    )


class SearchNearestEntitiesResponse(proto.Message):
    r"""Response message for
    [FeatureOnlineStoreService.SearchNearestEntities][google.cloud.aiplatform.v1beta1.FeatureOnlineStoreService.SearchNearestEntities]

    Attributes:
        nearest_neighbors (google.cloud.aiplatform_v1beta1.types.NearestNeighbors):
            The nearest neighbors of the query entity.
    """

    nearest_neighbors: "NearestNeighbors" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="NearestNeighbors",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
