# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
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
from typing import Optional, Dict, Sequence, Union
from google.cloud.aiplatform import schema

from google.cloud.aiplatform.compat.types import (
    io as gca_io,
    dataset as gca_dataset,
)


class Datasource(abc.ABC):
    """An abstract class that sets dataset_metadata."""

    @property
    @abc.abstractmethod
    def dataset_metadata(self):
        """Dataset Metadata."""
        pass


class DatasourceImportable(abc.ABC):
    """An abstract class that sets import_data_config."""

    @property
    @abc.abstractmethod
    def import_data_config(self):
        """Import Data Config."""
        pass


class TabularDatasource(Datasource):
    """Datasource for creating a tabular dataset for Vertex AI."""

    def __init__(
        self,
        gcs_source: Optional[Union[str, Sequence[str]]] = None,
        bq_source: Optional[str] = None,
    ):
        """Creates a tabular datasource.

        Args:
            gcs_source (Union[str, Sequence[str]]):
                Cloud Storage URI of one or more files. Only CSV files are supported.
                The first line of the CSV file is used as the header.
                If there are multiple files, the header is the first line of
                the lexicographically first file, the other files must either
                contain the exact same header or omit the header.
                examples:
                    str: "gs://bucket/file.csv"
                    Sequence[str]: ["gs://bucket/file1.csv", "gs://bucket/file2.csv"]
            bq_source (str):
                The URI of a BigQuery table.
                example:
                    "bq://project.dataset.table_name"

        Raises:
            ValueError: If source configuration is not valid.
        """

        dataset_metadata = None

        if gcs_source and isinstance(gcs_source, str):
            gcs_source = [gcs_source]

        if gcs_source and bq_source:
            raise ValueError("Only one of gcs_source or bq_source can be set.")

        if not any([gcs_source, bq_source]):
            raise ValueError("One of gcs_source or bq_source must be set.")

        if gcs_source:
            dataset_metadata = {"inputConfig": {"gcsSource": {"uri": gcs_source}}}
        elif bq_source:
            dataset_metadata = {"inputConfig": {"bigquerySource": {"uri": bq_source}}}

        self._dataset_metadata = dataset_metadata

    @property
    def dataset_metadata(self) -> Optional[Dict]:
        """Dataset Metadata."""
        return self._dataset_metadata


class NonTabularDatasource(Datasource):
    """Datasource for creating an empty non-tabular dataset for Vertex AI."""

    @property
    def dataset_metadata(self) -> Optional[Dict]:
        return None


class NonTabularDatasourceImportable(NonTabularDatasource, DatasourceImportable):
    """Datasource for creating a non-tabular dataset for Vertex AI and
    importing data to the dataset."""

    def __init__(
        self,
        gcs_source: Union[str, Sequence[str]],
        import_schema_uri: str,
        data_item_labels: Optional[Dict] = None,
    ):
        """Creates a non-tabular datasource.

        Args:
            gcs_source (Union[str, Sequence[str]]):
                Required. The Google Cloud Storage location for the input content.
                Google Cloud Storage URI(-s) to the input file(s).

                Examples:
                    str: "gs://bucket/file.csv"
                    Sequence[str]: ["gs://bucket/file1.csv", "gs://bucket/file2.csv"]
            import_schema_uri (str):
                Required. Points to a YAML file stored on Google Cloud
                Storage describing the import format. Validation will be
                done against the schema. The schema is defined as an
                `OpenAPI 3.0.2 Schema
            data_item_labels (Dict):
                Labels that will be applied to newly imported DataItems. If
                an identical DataItem as one being imported already exists
                in the Dataset, then these labels will be appended to these
                of the already existing one, and if labels with identical
                key is imported before, the old label value will be
                overwritten. If two DataItems are identical in the same
                import data operation, the labels will be combined and if
                key collision happens in this case, one of the values will
                be picked randomly. Two DataItems are considered identical
                if their content bytes are identical (e.g. image bytes or
                pdf bytes). These labels will be overridden by Annotation
                labels specified inside index file refenced by
                ``import_schema_uri``,
                e.g. jsonl file.
        """
        super().__init__()
        self._gcs_source = [gcs_source] if isinstance(gcs_source, str) else gcs_source
        self._import_schema_uri = import_schema_uri
        self._data_item_labels = data_item_labels

    @property
    def import_data_config(self) -> gca_dataset.ImportDataConfig:
        """Import Data Config."""
        return gca_dataset.ImportDataConfig(
            gcs_source=gca_io.GcsSource(uris=self._gcs_source),
            import_schema_uri=self._import_schema_uri,
            data_item_labels=self._data_item_labels,
        )


def create_datasource(
    metadata_schema_uri: str,
    import_schema_uri: Optional[str] = None,
    gcs_source: Optional[Union[str, Sequence[str]]] = None,
    bq_source: Optional[str] = None,
    data_item_labels: Optional[Dict] = None,
) -> Datasource:
    """Creates a datasource
    Args:
        metadata_schema_uri (str):
            Required. Points to a YAML file stored on Google Cloud Storage
            describing additional information about the Dataset. The schema
            is defined as an OpenAPI 3.0.2 Schema Object. The schema files
            that can be used here are found in gs://google-cloud-
            aiplatform/schema/dataset/metadata/.
        import_schema_uri (str):
            Points to a YAML file stored on Google Cloud
            Storage describing the import format. Validation will be
            done against the schema. The schema is defined as an
            `OpenAPI 3.0.2 Schema
        gcs_source (Union[str, Sequence[str]]):
            The Google Cloud Storage location for the input content.
            Google Cloud Storage URI(-s) to the input file(s).

            Examples:
                str: "gs://bucket/file.csv"
                Sequence[str]: ["gs://bucket/file1.csv", "gs://bucket/file2.csv"]
        bq_source (str):
            BigQuery URI to the input table.
            example:
                "bq://project.dataset.table_name"
        data_item_labels (Dict):
            Labels that will be applied to newly imported DataItems. If
            an identical DataItem as one being imported already exists
            in the Dataset, then these labels will be appended to these
            of the already existing one, and if labels with identical
            key is imported before, the old label value will be
            overwritten. If two DataItems are identical in the same
            import data operation, the labels will be combined and if
            key collision happens in this case, one of the values will
            be picked randomly. Two DataItems are considered identical
            if their content bytes are identical (e.g. image bytes or
            pdf bytes). These labels will be overridden by Annotation
            labels specified inside index file refenced by
            ``import_schema_uri``,
            e.g. jsonl file.

    Returns:
        datasource (Datasource)

    Raises:
        ValueError: When below scenarios happen:
        - import_schema_uri is identified for creating TabularDatasource
        - either import_schema_uri or gcs_source is missing for creating NonTabularDatasourceImportable
    """

    if metadata_schema_uri == schema.dataset.metadata.tabular:
        if import_schema_uri:
            raise ValueError("tabular dataset does not support data import.")
        return TabularDatasource(gcs_source, bq_source)

    if metadata_schema_uri == schema.dataset.metadata.time_series:
        if import_schema_uri:
            raise ValueError("time series dataset does not support data import.")
        return TabularDatasource(gcs_source, bq_source)

    if not import_schema_uri and not gcs_source:
        return NonTabularDatasource()
    elif import_schema_uri and gcs_source:
        return NonTabularDatasourceImportable(
            gcs_source, import_schema_uri, data_item_labels
        )
    else:
        raise ValueError(
            "nontabular dataset requires both import_schema_uri and gcs_source for data import."
        )
