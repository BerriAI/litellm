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

import copy
from typing import Dict, Optional, Union


class AvroOptions:
    """Options if source format is set to AVRO."""

    _SOURCE_FORMAT = "AVRO"
    _RESOURCE_NAME = "avroOptions"

    def __init__(self):
        self._properties = {}

    @property
    def use_avro_logical_types(self) -> Optional[bool]:
        """[Optional] If sourceFormat is set to 'AVRO', indicates whether to
        interpret logical types as the corresponding BigQuery data type (for
        example, TIMESTAMP), instead of using the raw type (for example,
        INTEGER).

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#AvroOptions.FIELDS.use_avro_logical_types
        """
        return self._properties.get("useAvroLogicalTypes")

    @use_avro_logical_types.setter
    def use_avro_logical_types(self, value):
        self._properties["useAvroLogicalTypes"] = value

    @classmethod
    def from_api_repr(cls, resource: Dict[str, bool]) -> "AvroOptions":
        """Factory: construct an instance from a resource dict.

        Args:
            resource (Dict[str, bool]):
                Definition of a :class:`~.format_options.AvroOptions` instance in
                the same representation as is returned from the API.

        Returns:
            :class:`~.format_options.AvroOptions`:
                Configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, bool]:
                A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)


class ParquetOptions:
    """Additional options if the PARQUET source format is used."""

    _SOURCE_FORMAT = "PARQUET"
    _RESOURCE_NAME = "parquetOptions"

    def __init__(self):
        self._properties = {}

    @property
    def enum_as_string(self) -> bool:
        """Indicates whether to infer Parquet ENUM logical type as STRING instead of
        BYTES by default.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ParquetOptions.FIELDS.enum_as_string
        """
        return self._properties.get("enumAsString")

    @enum_as_string.setter
    def enum_as_string(self, value: bool) -> None:
        self._properties["enumAsString"] = value

    @property
    def enable_list_inference(self) -> bool:
        """Indicates whether to use schema inference specifically for Parquet LIST
        logical type.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ParquetOptions.FIELDS.enable_list_inference
        """
        return self._properties.get("enableListInference")

    @enable_list_inference.setter
    def enable_list_inference(self, value: bool) -> None:
        self._properties["enableListInference"] = value

    @property
    def map_target_type(self) -> Optional[Union[bool, str]]:
        """Indicates whether to simplify the representation of parquet maps to only show keys and values."""

        return self._properties.get("mapTargetType")

    @map_target_type.setter
    def map_target_type(self, value: str) -> None:
        """Sets the map target type.

        Args:
          value: The map target type (eg ARRAY_OF_STRUCT).
        """
        self._properties["mapTargetType"] = value

    @classmethod
    def from_api_repr(cls, resource: Dict[str, bool]) -> "ParquetOptions":
        """Factory: construct an instance from a resource dict.

        Args:
            resource (Dict[str, bool]):
                Definition of a :class:`~.format_options.ParquetOptions` instance in
                the same representation as is returned from the API.

        Returns:
            :class:`~.format_options.ParquetOptions`:
                Configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, bool]:
                A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)
