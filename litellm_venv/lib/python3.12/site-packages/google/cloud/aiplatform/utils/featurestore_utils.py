# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
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

import re
from typing import Dict, NamedTuple, Optional

from google.cloud.aiplatform.compat.services import featurestore_service_client
from google.cloud.aiplatform.compat.types import (
    feature as gca_feature,
    featurestore_service as gca_featurestore_service,
)
from google.cloud.aiplatform import utils

CompatFeaturestoreServiceClient = featurestore_service_client.FeaturestoreServiceClient

RESOURCE_ID_PATTERN_REGEX = r"[a-z_][a-z0-9_]{0,59}"
GCS_SOURCE_TYPE = {"csv", "avro"}
GCS_DESTINATION_TYPE = {"csv", "tfrecord"}

_FEATURE_VALUE_TYPE_UNSPECIFIED = "VALUE_TYPE_UNSPECIFIED"

FEATURE_STORE_VALUE_TYPE_TO_BQ_DATA_TYPE_MAP = {
    "BOOL": {"field_type": "BOOL"},
    "BOOL_ARRAY": {"field_type": "BOOL", "mode": "REPEATED"},
    "DOUBLE": {"field_type": "FLOAT64"},
    "DOUBLE_ARRAY": {"field_type": "FLOAT64", "mode": "REPEATED"},
    "INT64": {"field_type": "INT64"},
    "INT64_ARRAY": {"field_type": "INT64", "mode": "REPEATED"},
    "STRING": {"field_type": "STRING"},
    "STRING_ARRAY": {"field_type": "STRING", "mode": "REPEATED"},
    "BYTES": {"field_type": "BYTES"},
}


def validate_id(resource_id: str) -> None:
    """Validates feature store resource ID pattern.

    Args:
        resource_id (str):
            Required. Feature Store resource ID.

    Raises:
        ValueError if resource_id is invalid.
    """
    if not re.compile(r"^" + RESOURCE_ID_PATTERN_REGEX + r"$").match(resource_id):
        raise ValueError("Resource ID {resource_id} is not a valied resource id.")


def validate_feature_id(feature_id: str) -> None:
    """Validates feature ID.

    Args:
        feature_id (str):
            Required. Feature resource ID.

    Raises:
        ValueError if feature_id is invalid.
    """
    match = re.compile(r"^" + RESOURCE_ID_PATTERN_REGEX + r"$").match(feature_id)

    if not match:
        raise ValueError(
            f"The value of feature_id may be up to 60 characters, and valid characters are `[a-z0-9_]`. "
            f"The first character cannot be a number. Instead, get {feature_id}."
        )

    reserved_words = ["entity_id", "feature_timestamp", "arrival_timestamp"]
    if feature_id.lower() in reserved_words:
        raise ValueError(
            "The feature_id can not be any of the reserved_words: `%s`"
            % ("`, `".join(reserved_words))
        )


def validate_value_type(value_type: str) -> None:
    """Validates user provided feature value_type string.

    Args:
        value_type (str):
            Required. Immutable. Type of Feature value.
            One of BOOL, BOOL_ARRAY, DOUBLE, DOUBLE_ARRAY, INT64, INT64_ARRAY, STRING, STRING_ARRAY, BYTES.

    Raises:
        ValueError if value_type is invalid or unspecified.
    """
    if getattr(gca_feature.Feature.ValueType, value_type, None) in (
        gca_feature.Feature.ValueType.VALUE_TYPE_UNSPECIFIED,
        None,
    ):
        raise ValueError(
            f"Given value_type `{value_type}` invalid or unspecified. "
            f"Choose one of {gca_feature.Feature.ValueType._member_names_} except `{_FEATURE_VALUE_TYPE_UNSPECIFIED}`"
        )


class _FeatureConfig(NamedTuple):
    """Configuration for feature creation.

    Usage:

    config = _FeatureConfig(
        feature_id='my_feature_id',
        value_type='int64',
        description='my description',
        labels={'my_key': 'my_value'},
    )
    """

    feature_id: str
    value_type: str = _FEATURE_VALUE_TYPE_UNSPECIFIED
    description: Optional[str] = None
    labels: Optional[Dict[str, str]] = None

    def _get_feature_id(self) -> str:
        """Validates and returns the feature_id.

        Returns:
            str - valid feature ID.

        Raise:
            ValueError if feature_id is invalid
        """

        # Raises ValueError if invalid feature_id
        validate_feature_id(feature_id=self.feature_id)

        return self.feature_id

    def _get_value_type_enum(self) -> int:
        """Validates value_type and returns the enum of the value type.

        Returns:
            int - valid value type enum.
        """

        # Raises ValueError if invalid value_type
        validate_value_type(value_type=self.value_type)

        value_type_enum = getattr(gca_feature.Feature.ValueType, self.value_type)

        return value_type_enum

    def get_create_feature_request(
        self,
    ) -> gca_featurestore_service.CreateFeatureRequest:
        """Return create feature request."""

        gapic_feature = gca_feature.Feature(
            value_type=self._get_value_type_enum(),
        )

        if self.labels:
            utils.validate_labels(self.labels)
            gapic_feature.labels = self.labels

        if self.description:
            gapic_feature.description = self.description

        create_feature_request = gca_featurestore_service.CreateFeatureRequest(
            feature=gapic_feature, feature_id=self._get_feature_id()
        )

        return create_feature_request
