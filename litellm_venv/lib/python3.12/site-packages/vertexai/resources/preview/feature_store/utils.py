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
from dataclasses import dataclass
import enum
import proto
from typing_extensions import override
from typing import Any, Dict, List, Optional
from google.cloud.aiplatform.compat.types import (
    feature_online_store_service as fos_service,
)


def get_feature_online_store_name(online_store_name: str) -> str:
    """Extract Feature Online Store's name from FeatureView's full resource name.

    Args:
        online_store_name: Full resource name is projects/project_number/
        locations/us-central1/featureOnlineStores/fos_name/featureViews/fv_name

    Returns:
        str: feature online store name.
    """
    arr = online_store_name.split("/")
    return arr[5]


class PublicEndpointNotFoundError(RuntimeError):
    """Public endpoint has not been created yet."""


@dataclass
class FeatureViewReadResponse:
    _response: fos_service.FetchFeatureValuesResponse

    def __init__(self, response: fos_service.FetchFeatureValuesResponse):
        self._response = response

    def to_dict(self) -> Dict[str, Any]:
        return proto.Message.to_dict(self._response.key_values)

    def to_proto(self) -> fos_service.FetchFeatureValuesResponse:
        return self._response


@dataclass
class SearchNearestEntitiesResponse:
    _response: fos_service.SearchNearestEntitiesResponse

    def __init__(self, response: fos_service.SearchNearestEntitiesResponse):
        self._response = response

    def to_dict(self) -> Dict[str, Any]:
        return proto.Message.to_dict(self._response.nearest_neighbors)

    def to_proto(self) -> fos_service.SearchNearestEntitiesResponse:
        return self._response


class DistanceMeasureType(enum.Enum):
    """The distance measure used in nearest neighbor search."""

    DISTANCE_MEASURE_TYPE_UNSPECIFIED = 0
    # Euclidean (L_2) Distance.
    SQUARED_L2_DISTANCE = 1
    # Cosine Distance. Defined as 1 - cosine similarity.
    COSINE_DISTANCE = 2
    # Dot Product Distance. Defined as a negative of the dot product.
    DOT_PRODUCT_DISTANCE = 3


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
    """

    leaf_node_embedding_count: Optional[int] = None

    @override
    def as_dict(self) -> Dict:
        return {"leaf_node_embedding_count": self.leaf_node_embedding_count}


@dataclass
class BruteForceConfig(AlgorithmConfig):
    """Configuration options for using brute force search.
    It simply implements the standard linear search in the database for
    each query.
    """

    @override
    def as_dict(self) -> Dict[str, Any]:
        return {"bruteForceConfig": {}}


@dataclass
class IndexConfig:
    """Configuration options for the Vertex FeatureView for embedding."""

    embedding_column: str
    filter_column: List[str]
    crowding_column: str
    dimentions: Optional[int]
    distance_measure_type: DistanceMeasureType
    algorithm_config: AlgorithmConfig

    def as_dict(self) -> Dict[str, Any]:
        """Returns the configuration as a dictionary.

        Returns:
            Dict[str, Any]
        """
        config = {
            "embedding_column": self.embedding_column,
            "filter_columns": self.filter_column,
            "crowding_column": self.crowding_column,
            "embedding_dimension": self.dimentions,
            "distance_measure_type": self.distance_measure_type.value,
        }
        if isinstance(self.algorithm_config, TreeAhConfig):
            config["tree_ah_config"] = self.algorithm_config.as_dict()
        else:
            config["brute_force_config"] = self.algorithm_config.as_dict()
        return config
