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

from google.cloud.aiplatform_v1beta1.types import index


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "FindNeighborsRequest",
        "FindNeighborsResponse",
        "ReadIndexDatapointsRequest",
        "ReadIndexDatapointsResponse",
    },
)


class FindNeighborsRequest(proto.Message):
    r"""The request message for
    [MatchService.FindNeighbors][google.cloud.aiplatform.v1beta1.MatchService.FindNeighbors].

    Attributes:
        index_endpoint (str):
            Required. The name of the index endpoint. Format:
            ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
        deployed_index_id (str):
            The ID of the DeployedIndex that will serve the request.
            This request is sent to a specific IndexEndpoint, as per the
            IndexEndpoint.network. That IndexEndpoint also has
            IndexEndpoint.deployed_indexes, and each such index has a
            DeployedIndex.id field. The value of the field below must
            equal one of the DeployedIndex.id fields of the
            IndexEndpoint that is being called for this request.
        queries (MutableSequence[google.cloud.aiplatform_v1beta1.types.FindNeighborsRequest.Query]):
            The list of queries.
        return_full_datapoint (bool):
            If set to true, the full datapoints
            (including all vector values and restricts) of
            the nearest neighbors are returned. Note that
            returning full datapoint will significantly
            increase the latency and cost of the query.
    """

    class Query(proto.Message):
        r"""A query to find a number of the nearest neighbors (most
        similar vectors) of a vector.

        Attributes:
            datapoint (google.cloud.aiplatform_v1beta1.types.IndexDatapoint):
                Required. The datapoint/vector whose nearest
                neighbors should be searched for.
            neighbor_count (int):
                The number of nearest neighbors to be
                retrieved from database for each query. If not
                set, will use the default from the service
                configuration
                (https://cloud.google.com/vertex-ai/docs/matching-engine/configuring-indexes#nearest-neighbor-search-config).
            per_crowding_attribute_neighbor_count (int):
                Crowding is a constraint on a neighbor list produced by
                nearest neighbor search requiring that no more than some
                value k' of the k neighbors returned have the same value of
                crowding_attribute. It's used for improving result
                diversity. This field is the maximum number of matches with
                the same crowding tag.
            approximate_neighbor_count (int):
                The number of neighbors to find via
                approximate search before exact reordering is
                performed. If not set, the default value from
                scam config is used; if set, this value must be
                > 0.
            fraction_leaf_nodes_to_search_override (float):
                The fraction of the number of leaves to search, set at query
                time allows user to tune search performance. This value
                increase result in both search accuracy and latency
                increase. The value should be between 0.0 and 1.0. If not
                set or set to 0.0, query uses the default value specified in
                NearestNeighborSearchConfig.TreeAHConfig.fraction_leaf_nodes_to_search.
        """

        datapoint: index.IndexDatapoint = proto.Field(
            proto.MESSAGE,
            number=1,
            message=index.IndexDatapoint,
        )
        neighbor_count: int = proto.Field(
            proto.INT32,
            number=2,
        )
        per_crowding_attribute_neighbor_count: int = proto.Field(
            proto.INT32,
            number=3,
        )
        approximate_neighbor_count: int = proto.Field(
            proto.INT32,
            number=4,
        )
        fraction_leaf_nodes_to_search_override: float = proto.Field(
            proto.DOUBLE,
            number=5,
        )

    index_endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    deployed_index_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    queries: MutableSequence[Query] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=Query,
    )
    return_full_datapoint: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class FindNeighborsResponse(proto.Message):
    r"""The response message for
    [MatchService.FindNeighbors][google.cloud.aiplatform.v1beta1.MatchService.FindNeighbors].

    Attributes:
        nearest_neighbors (MutableSequence[google.cloud.aiplatform_v1beta1.types.FindNeighborsResponse.NearestNeighbors]):
            The nearest neighbors of the query
            datapoints.
    """

    class Neighbor(proto.Message):
        r"""A neighbor of the query vector.

        Attributes:
            datapoint (google.cloud.aiplatform_v1beta1.types.IndexDatapoint):
                The datapoint of the neighbor. Note that full datapoints are
                returned only when "return_full_datapoint" is set to true.
                Otherwise, only the "datapoint_id" and "crowding_tag" fields
                are populated.
            distance (float):
                The distance between the neighbor and the
                query vector.
        """

        datapoint: index.IndexDatapoint = proto.Field(
            proto.MESSAGE,
            number=1,
            message=index.IndexDatapoint,
        )
        distance: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )

    class NearestNeighbors(proto.Message):
        r"""Nearest neighbors for one query.

        Attributes:
            id (str):
                The ID of the query datapoint.
            neighbors (MutableSequence[google.cloud.aiplatform_v1beta1.types.FindNeighborsResponse.Neighbor]):
                All its neighbors.
        """

        id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        neighbors: MutableSequence[
            "FindNeighborsResponse.Neighbor"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message="FindNeighborsResponse.Neighbor",
        )

    nearest_neighbors: MutableSequence[NearestNeighbors] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=NearestNeighbors,
    )


class ReadIndexDatapointsRequest(proto.Message):
    r"""The request message for
    [MatchService.ReadIndexDatapoints][google.cloud.aiplatform.v1beta1.MatchService.ReadIndexDatapoints].

    Attributes:
        index_endpoint (str):
            Required. The name of the index endpoint. Format:
            ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
        deployed_index_id (str):
            The ID of the DeployedIndex that will serve
            the request.
        ids (MutableSequence[str]):
            IDs of the datapoints to be searched for.
    """

    index_endpoint: str = proto.Field(
        proto.STRING,
        number=1,
    )
    deployed_index_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


class ReadIndexDatapointsResponse(proto.Message):
    r"""The response message for
    [MatchService.ReadIndexDatapoints][google.cloud.aiplatform.v1beta1.MatchService.ReadIndexDatapoints].

    Attributes:
        datapoints (MutableSequence[google.cloud.aiplatform_v1beta1.types.IndexDatapoint]):
            The result list of datapoints.
    """

    datapoints: MutableSequence[index.IndexDatapoint] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=index.IndexDatapoint,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
