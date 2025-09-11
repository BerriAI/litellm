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

from typing import Dict, Optional, Sequence, Tuple, Union

from google.auth import credentials as auth_credentials

from google.cloud import bigquery
from google.cloud.aiplatform import base
from google.cloud.aiplatform import datasets
from google.cloud.aiplatform.datasets import _datasources
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import schema
from google.cloud.aiplatform import utils

_AUTOML_TRAINING_MIN_ROWS = 1000

_LOGGER = base.Logger(__name__)


class TabularDataset(datasets._ColumnNamesDataset):
    """A managed tabular dataset resource for Vertex AI.

    Use this class to work with tabular datasets. You can use a CSV file, BigQuery, or a pandas
    [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
    to create a tabular dataset. For more information about paging through
    BigQuery data, see [Read data with BigQuery API using
    pagination](https://cloud.google.com/bigquery/docs/paging-results). For more
    information about tabular data, see [Tabular
    data](https://cloud.google.com/vertex-ai/docs/training-overview#tabular_data).

    The following code shows you how to create and import a tabular
    dataset with a CSV file.

    ```py
    my_dataset = aiplatform.TabularDataset.create(
        display_name="my-dataset", gcs_source=['gs://path/to/my/dataset.csv'])
    ```

    The following code shows you how to create and import a tabular
    dataset in two distinct steps.

    ```py
    my_dataset = aiplatform.TextDataset.create(
        display_name="my-dataset")

    my_dataset.import(
        gcs_source=['gs://path/to/my/dataset.csv']
        import_schema_uri=aiplatform.schema.dataset.ioformat.text.multi_label_classification
    )
    ```

    If you create a tabular dataset with a pandas
    [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html),
    you need to use a BigQuery table to stage the data for Vertex AI:

    ```py
    my_dataset = aiplatform.TabularDataset.create_from_dataframe(
        df_source=my_pandas_dataframe,
        staging_path=f"bq://{bq_dataset_id}.table-unique"
    )
    ```

    """

    _supported_metadata_schema_uris: Optional[Tuple[str]] = (
        schema.dataset.metadata.tabular,
    )

    @classmethod
    def create(
        cls,
        display_name: Optional[str] = None,
        gcs_source: Optional[Union[str, Sequence[str]]] = None,
        bq_source: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "TabularDataset":
        """Creates a tabular dataset.

        Args:
            display_name (str):
                Optional. The user-defined name of the dataset. The name must
                contain 128 or fewer UTF-8 characters.
            gcs_source (Union[str, Sequence[str]]):
                Optional. The URI to one or more Google Cloud Storage buckets that contain
                your datasets. For example, `str: "gs://bucket/file.csv"` or
                `Sequence[str]: ["gs://bucket/file1.csv",
                "gs://bucket/file2.csv"]`.
            bq_source (str):
                Optional. The URI to a BigQuery table that's used as an input source. For
                example, `bq://project.dataset.table_name`.
            project (str):
                Optional. The name of the Google Cloud project to which this
                `TabularDataset` is uploaded. This overrides the project that
                was set by `aiplatform.init`.
            location (str):
                Optional. The Google Cloud region where this dataset is uploaded. This
                region overrides the region that was set by `aiplatform.init`.
            credentials (auth_credentials.Credentials):
                Optional. The credentials that are used to upload the `TabularDataset`.
                These credentials override the credentials set by
                `aiplatform.init`.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings that contain metadata that's sent with the request.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your
                Vertex AI Tensorboards. The maximum length of a key and of a
                value is 64 unicode characters. Labels and keys can contain only
                lowercase letters, numeric characters, underscores, and dashes.
                International characters are allowed. No more than 64 user
                labels can be associated with one Tensorboard (system labels are
                excluded). For more information and examples of using labels, see
                [Using labels to organize Google Cloud Platform resources](https://goo.gl/xmQnxf).
                System reserved label keys are prefixed with
                `aiplatform.googleapis.com/` and are immutable.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key that's used to protect the dataset. The
                format of the key is
                `projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key`.
                The key needs to be in the same region as where the compute
                resource is created.

                If `encryption_spec_key_name` is set, this `TabularDataset` and
                all of its sub-resources are secured by this key.

                This `encryption_spec_key_name` overrides the
                `encryption_spec_key_name` set by `aiplatform.init`.
            sync (bool):
                If `true`, the `create` method creates a tabular dataset
                synchronously. If `false`, the `create` method creates a tabular
                dataset asynchronously.
            create_request_timeout (float):
                Optional. The number of seconds for the timeout of the create
                request.

        Returns:
            tabular_dataset (TabularDataset):
                An instantiated representation of the managed `TabularDataset` resource.
        """
        if not display_name:
            display_name = cls._generate_display_name()
        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        metadata_schema_uri = schema.dataset.metadata.tabular

        datasource = _datasources.create_datasource(
            metadata_schema_uri=metadata_schema_uri,
            gcs_source=gcs_source,
            bq_source=bq_source,
        )

        return cls._create_and_import(
            api_client=api_client,
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            display_name=display_name,
            metadata_schema_uri=metadata_schema_uri,
            datasource=datasource,
            project=project or initializer.global_config.project,
            location=location or initializer.global_config.location,
            credentials=credentials or initializer.global_config.credentials,
            request_metadata=request_metadata,
            labels=labels,
            encryption_spec=initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            ),
            sync=sync,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    def create_from_dataframe(
        cls,
        df_source: "pd.DataFrame",  # noqa: F821 - skip check for undefined name 'pd'
        staging_path: str,
        bq_schema: Optional[Union[str, bigquery.SchemaField]] = None,
        display_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "TabularDataset":
        """Creates a new tabular dataset from a pandas `DataFrame`.

        Args:
            df_source (pd.DataFrame):
                Required. A pandas
                [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
                containing the source data for ingestion as a `TabularDataset`.
                This method uses the data types from the provided `DataFrame`
                when the `TabularDataset` is created.
            staging_path (str):
                Required. The BigQuery table used to stage the data for Vertex
                AI. Because Vertex AI maintains a reference to this source to
                create the `TabularDataset`, you shouldn't delete this BigQuery
                table. For example: `bq://my-project.my-dataset.my-table`.
                If the specified BigQuery table doesn't exist, then the table is
                created for you. If the provided BigQuery table already exists,
                and the schemas of the BigQuery table and your DataFrame match,
                then the data in your local `DataFrame` is appended to the table.
                The location of the BigQuery table must conform to the
                [BigQuery location requirements](https://cloud.google.com/vertex-ai/docs/general/locations#bq-locations).
            bq_schema (Optional[Union[str, bigquery.SchemaField]]):
                Optional. If not set, BigQuery autodetects the schema using the
                column types of your `DataFrame`. If set, BigQuery uses the
                schema you provide when the staging table is created. For more
                information,
                see the BigQuery
                [`LoadJobConfig.schema`](https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.job.LoadJobConfig#google_cloud_bigquery_job_LoadJobConfig_schema)
                property.
            display_name (str):
                Optional. The user-defined name of the `Dataset`. The name must
                contain 128 or fewer UTF-8 characters.
            project (str):
                Optional. The project to upload this dataset to. This overrides
                the project set using `aiplatform.init`.
            location (str):
                Optional. The location to upload this dataset to. This overrides
                the location set using `aiplatform.init`.
            credentials (auth_credentials.Credentials):
                Optional. The custom credentials used to upload this dataset.
                This overrides credentials set using `aiplatform.init`.
        Returns:
            tabular_dataset (TabularDataset):
                An instantiated representation of the managed `TabularDataset` resource.
        """

        if staging_path.startswith("bq://"):
            bq_staging_path = staging_path[len("bq://") :]
        else:
            raise ValueError(
                "Only BigQuery staging paths are supported. Provide a staging path in the format `bq://your-project.your-dataset.your-table`."
            )

        try:
            import pyarrow  # noqa: F401 - skip check for 'pyarrow' which is required when using 'google.cloud.bigquery'
        except ImportError:
            raise ImportError(
                "Pyarrow is not installed, and is required to use the BigQuery client."
                'Please install the SDK using "pip install google-cloud-aiplatform[datasets]"'
            )
        import pandas.api.types as pd_types

        if any(
            [
                pd_types.is_datetime64_any_dtype(df_source[column])
                for column in df_source.columns
            ]
        ):
            _LOGGER.info(
                "Received datetime-like column in the dataframe. Please note that the column could be interpreted differently in BigQuery depending on which major version you are using. For more information, please reference the BigQuery v3 release notes here: https://github.com/googleapis/python-bigquery/releases/tag/v3.0.0"
            )

        if len(df_source) < _AUTOML_TRAINING_MIN_ROWS:
            _LOGGER.info(
                "Your DataFrame has %s rows and AutoML requires %s rows to train on tabular data. You can still train a custom model once your dataset has been uploaded to Vertex, but you will not be able to use AutoML for training."
                % (len(df_source), _AUTOML_TRAINING_MIN_ROWS),
            )

        bigquery_client = bigquery.Client(
            project=project or initializer.global_config.project,
            credentials=credentials or initializer.global_config.credentials,
        )

        try:
            parquet_options = bigquery.format_options.ParquetOptions()
            parquet_options.enable_list_inference = True

            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                parquet_options=parquet_options,
            )

            if bq_schema:
                job_config.schema = bq_schema

            job = bigquery_client.load_table_from_dataframe(
                dataframe=df_source, destination=bq_staging_path, job_config=job_config
            )

            job.result()

        finally:
            dataset_from_dataframe = cls.create(
                display_name=display_name,
                bq_source=staging_path,
                project=project,
                location=location,
                credentials=credentials,
            )

        return dataset_from_dataframe

    def import_data(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not support 'import_data'"
        )
