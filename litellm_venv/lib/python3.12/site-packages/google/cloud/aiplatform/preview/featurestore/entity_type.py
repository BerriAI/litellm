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

from typing import Dict, List, Union

from google.cloud.aiplatform import base
from google.cloud.aiplatform.compat.types import (
    featurestore_online_service_v1beta1 as gca_featurestore_online_service_v1beta1,
)
from google.cloud.aiplatform.compat.types import (
    types_v1beta1 as gca_types_v1beta1,
)

from google.cloud.aiplatform.featurestore import _entity_type

_LOGGER = base.Logger(__name__)


class EntityType(_entity_type._EntityType):
    """Preview EntityType resource for Vertex AI."""

    # TODO(b/262275273): Remove preview v1beta1 implementation of `write_feature_values`
    # when GA implementation can write multiple payloads per request. Currently, GA
    # supports one payload per request.
    def write_feature_values(
        self,
        instances: Union[
            List[gca_featurestore_online_service_v1beta1.WriteFeatureValuesPayload],
            Dict[
                str,
                Dict[
                    str,
                    Union[
                        int,
                        str,
                        float,
                        bool,
                        bytes,
                        List[int],
                        List[str],
                        List[float],
                        List[bool],
                    ],
                ],
            ],
            "pd.DataFrame",  # type: ignore # noqa: F821 - skip check for undefined name 'pd'
        ],
    ) -> "EntityType":
        """Streaming ingestion. Write feature values directly to Feature Store.

        ```
        my_entity_type = aiplatform.EntityType(
            entity_type_name="my_entity_type_id",
            featurestore_id="my_featurestore_id",
        )

        # writing feature values from a pandas DataFrame
        my_dataframe = pd.DataFrame(
            data = [
                {"entity_id": "movie_01", "average_rating": 4.9},
                {"entity_id": "movie_02", "average_rating": 4.5},
            ],
            columns=["entity_id", "average_rating"],
        )
        my_dataframe = my_df.set_index("entity_id")

        my_entity_type.preview.write_feature_values(
            instances=my_df
        )

        # writing feature values from a Python dict
        my_data_dict = {
                "movie_03" : {"average_rating": 3.7},
                "movie_04" : {"average_rating": 2.5},
        }

        my_entity_type.preview.write_feature_values(
            instances=my_data_dict
        )

        # writing feature values from a list of WriteFeatureValuesPayload objects
        payloads = [
            gca_featurestore_online_service_v1beta1.WriteFeatureValuesPayload(
                entity_id="movie_05",
                feature_values=gca_featurestore_online_service_v1beta1.FeatureValue(
                    double_value=4.9
                )
            )
        ]

        my_entity_type.preview.write_feature_values(
            instances=payloads
        )

        # reading back written feature values
        my_entity_type.read(
            entity_ids=["movie_01", "movie_02", "movie_03", "movie_04", "movie_05"]
        )
        ```

        Args:
            instances (
                Union[
                    List[gca_featurestore_online_service_v1beta1.WriteFeatureValuesPayload],
                    Dict[str, Dict[str, Union[int, str, float, bool, bytes,
                        List[int], List[str], List[float], List[bool]]]],
                    pd.Dataframe]):
                Required. Feature values to be written to the Feature Store that
                can take the form of a list of WriteFeatureValuesPayload objects,
                a Python dict of the form {entity_id : {feature_id : feature_value}, ...},
                or a pandas Dataframe, where the index column holds the unique entity
                ID strings and each remaining column represents a feature.  Each row
                in the pandas Dataframe represents an entity, which has an entity ID
                and its associated feature values.

        Returns:
            EntityType - The updated EntityType object.
        """

        if isinstance(instances, Dict):
            payloads = self._generate_payloads(instances=instances)
        elif isinstance(instances, List):
            payloads = instances
        else:
            instances_dict = instances.to_dict(orient="index")
            payloads = self._generate_payloads(instances=instances_dict)

        _LOGGER.log_action_start_against_resource(
            "Writing",
            "feature values",
            self,
        )

        self._featurestore_online_client.select_version("v1beta1").write_feature_values(
            entity_type=self.resource_name, payloads=payloads
        )

        _LOGGER.log_action_completed_against_resource("feature values", "written", self)

        return self

    @classmethod
    def _generate_payloads(
        cls,
        instances: Dict[
            str,
            Dict[
                str,
                Union[
                    int,
                    str,
                    float,
                    bool,
                    bytes,
                    List[int],
                    List[str],
                    List[float],
                    List[bool],
                ],
            ],
        ],
    ) -> List[gca_featurestore_online_service_v1beta1.WriteFeatureValuesPayload]:
        """Helper method used to generate GAPIC WriteFeatureValuesPayloads from
        a Python dict.

        Args:
            instances (Dict[str, Dict[str, Union[int, str, float, bool, bytes,
                List[int], List[str], List[float], List[bool]]]]):
                Required. Dict mapping entity IDs to their corresponding features.

        Returns:
            List[gca_featurestore_online_service_v1beta1.WriteFeatureValuesPayload] -
            A list of WriteFeatureValuesPayload objects ready to be written to the Feature Store.
        """
        payloads = []
        for entity_id, features in instances.items():
            feature_values = {}
            for feature_id, value in features.items():
                feature_value = cls._convert_value_to_gapic_feature_value(
                    feature_id=feature_id, value=value
                )
                feature_values[feature_id] = feature_value
            payload = gca_featurestore_online_service_v1beta1.WriteFeatureValuesPayload(
                entity_id=entity_id, feature_values=feature_values
            )
            payloads.append(payload)

        return payloads

    @classmethod
    def _convert_value_to_gapic_feature_value(
        cls,
        feature_id: str,
        value: Union[
            int, str, float, bool, bytes, List[int], List[str], List[float], List[bool]
        ],
    ) -> gca_featurestore_online_service_v1beta1.FeatureValue:
        """Helper method that converts a Python literal value or a list of
        literals to a GAPIC FeatureValue.

        Args:
            feature_id (str):
                Required. Name of a feature.
            value (Union[int, str, float, bool, bytes,
                List[int], List[str], List[float], List[bool]]]):
                Required. Python literal value or list of Python literals to
                be converted to a GAPIC FeatureValue.

        Returns:
            gca_featurestore_online_service_v1beta1.FeatureValue - GAPIC object
            that represents the value of a feature.

        Raises:
            ValueError if a list has values that are not all of the same type.
            ValueError if feature type is not supported.
        """
        if isinstance(value, bool):
            feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                bool_value=value
            )
        elif isinstance(value, str):
            feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                string_value=value
            )
        elif isinstance(value, int):
            feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                int64_value=value
            )
        elif isinstance(value, float):
            feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                double_value=value
            )
        elif isinstance(value, bytes):
            feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                bytes_value=value
            )
        elif isinstance(value, List):
            if all([isinstance(item, bool) for item in value]):
                feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                    bool_array_value=gca_types_v1beta1.BoolArray(values=value)
                )
            elif all([isinstance(item, str) for item in value]):
                feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                    string_array_value=gca_types_v1beta1.StringArray(values=value)
                )
            elif all([isinstance(item, int) for item in value]):
                feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                    int64_array_value=gca_types_v1beta1.Int64Array(values=value)
                )
            elif all([isinstance(item, float) for item in value]):
                feature_value = gca_featurestore_online_service_v1beta1.FeatureValue(
                    double_array_value=gca_types_v1beta1.DoubleArray(values=value)
                )
            else:
                raise ValueError(
                    f"Cannot infer feature value for feature {feature_id} with "
                    f"value {value}! Please ensure every value in the list "
                    f"is the same type (either int, str, float, bool)."
                )

        else:
            raise ValueError(
                f"Cannot infer feature value for feature {feature_id} with "
                f"value {value}! {type(value)} type is not supported. "
                f"Please ensure value type is an int, str, float, bool, "
                f"bytes, or a list of int, str, float, bool."
            )
        return feature_value
