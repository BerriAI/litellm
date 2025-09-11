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

import abc
import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional


# This file mirrors the configuration options as defined in gs://google-cloud-aiplatform/schema/matchingengine/metadata/nearest_neighbor_search_1.0.0.yaml
class DistanceMeasureType(enum.Enum):
    """The distance measure used in nearest neighbor search."""

    # Dot Product Distance. Defined as a negative of the dot product
    DOT_PRODUCT_DISTANCE = "DOT_PRODUCT_DISTANCE"
    # Euclidean (L_2) Distance
    SQUARED_L2_DISTANCE = "SQUARED_L2_DISTANCE"
    # Manhattan (L_1) Distance
    L1_DISTANCE = "L1_DISTANCE"
    # Cosine Distance. Defined as 1 - cosine similarity.
    COSINE_DISTANCE = "COSINE_DISTANCE"


class FeatureNormType(enum.Enum):
    """Type of normalization to be carried out on each vector."""

    # Unit L2 normalization type.
    UNIT_L2_NORM = "UNIT_L2_NORM"
    # No normalization type is specified.
    NONE = "NONE"


class AlgorithmConfig(abc.ABC):
    """Base class for configuration options for matching algorithm."""

    def as_dict(self) -> Dict:
        """Returns the configuration as a dictionary.

        Returns:
            Dict[str, Any]
        """
        pass


@dataclass
class TreeAhConfig(AlgorithmConfig):
    """Configuration options for using the tree-AH algorithm (Shallow tree + Asymmetric Hashing).
    Please refer to this paper for more details: https://arxiv.org/abs/1908.10396

    Args:
        leaf_node_embedding_count (int):
            Optional. Number of embeddings on each leaf node. The default value is 1000 if not set.
        leaf_nodes_to_search_percent (float):
            The default percentage of leaf nodes that any query may be searched. Must be in
            range 1-100, inclusive. The default value is 10 (means 10%) if not set.
    """

    leaf_node_embedding_count: Optional[int] = None
    leaf_nodes_to_search_percent: Optional[float] = None

    def as_dict(self) -> Dict:
        """Returns the configuration as a dictionary.

        Returns:
            Dict[str, Any]
        """

        return {
            "treeAhConfig": {
                "leafNodeEmbeddingCount": self.leaf_node_embedding_count,
                "leafNodesToSearchPercent": self.leaf_nodes_to_search_percent,
            }
        }


@dataclass
class BruteForceConfig(AlgorithmConfig):
    """Configuration options for using brute force search, which simply
    implements the standard linear search in the database for each query.
    """

    def as_dict(self) -> Dict:
        """Returns the configuration as a dictionary.

        Returns:
            Dict[str, Any]
        """
        return {"bruteForceConfig": {}}


@dataclass
class MatchingEngineIndexConfig:
    """Configuration options for using the tree-AH algorithm (Shallow tree + Asymmetric Hashing).
    Please refer to this paper for more details: https://arxiv.org/abs/1908.10396

    Args:
        dimensions (int):
            Required. The number of dimensions of the input vectors.
        algorithm_config (AlgorithmConfig):
            Required. The configuration with regard to the algorithms used for efficient search.
        approximate_neighbors_count (int):
            Optional. The default number of neighbors to find via approximate search before exact reordering is
            performed. Exact reordering is a procedure where results returned by an
            approximate search algorithm are reordered via a more expensive distance computation.

            Required if tree-AH algorithm is used.
        shard_size (str):
            Optional. The size of each shard. Index will get resharded the
            based on specified shard size. During serving,
            each shard will be served on a separate node and will scale
            independently.
        distance_measure_type (DistanceMeasureType):
            Optional. The distance measure used in nearest neighbor search.
    """

    dimensions: int
    algorithm_config: AlgorithmConfig
    approximate_neighbors_count: Optional[int] = None
    distance_measure_type: Optional[DistanceMeasureType] = None
    shard_size: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Returns the configuration as a dictionary.

        Returns:
            Dict[str, Any]
        """
        res = {
            "dimensions": self.dimensions,
            "algorithmConfig": self.algorithm_config.as_dict(),
            "approximateNeighborsCount": self.approximate_neighbors_count,
            "distanceMeasureType": self.distance_measure_type,
            "shardSize": self.shard_size,
        }
        return res
