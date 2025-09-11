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

import io
import os
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from google.cloud import bigquery
from google.cloud import storage
from google.cloud.aiplatform import compat
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform_v1beta1.services import (
    evaluation_service as gapic_evaluation_services,
)

if TYPE_CHECKING:
    import pandas as pd

_BQ_PREFIX = "bq://"
_GCS_PREFIX = "gs://"


class _EvaluationServiceAsyncClientWithOverride(utils.ClientWithOverride):
    _is_temporary = False
    _default_version = compat.V1BETA1
    _version_map = (
        (
            compat.V1BETA1,
            gapic_evaluation_services.EvaluationServiceAsyncClient,
        ),
    )


def create_evaluation_service_async_client(
    api_base_path_override: Optional[str] = None,
) -> _EvaluationServiceAsyncClientWithOverride:
    """Creates an aync client for the evaluation service.

    Args:
      api_base_path_override: Optional. Override default api base path.

    Returns:
      Instantiated Vertex AI EvaluationService async client with optional overrides.
    """
    return initializer.global_config.create_client(
        client_class=_EvaluationServiceAsyncClientWithOverride,
        location_override=initializer.global_config.location,
        api_base_path_override=api_base_path_override,
    )


def load_dataset(source: Union[str, "pd.DataFrame", Dict[str, Any]]) -> "pd.DataFrame":
    """Loads dataset from various sources into a DataFrame.

    Args:
        source: The data source. Can be the following formats:
          - pd.DataFrame: Used directly for evaluation.
          - dict: Converted to a pandas DataFrame before evaluation.
          - str: Interpreted as a file path or URI. Supported formats include:
              * Local JSONL or CSV files:  Loaded from the local filesystem.
              * GCS JSONL or CSV files: Loaded from Google Cloud Storage
                  (e.g., 'gs://bucket/data.csv').
              * BigQuery table URI: Loaded from Google Cloud BigQuery
                  (e.g., 'bq://project-id.dataset.table_name').

    Returns:
        The dataset in pandas DataFrame format.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Pandas is not installed. Please install the SDK using "pip install'
            ' google-cloud-aiplatform[rapid_evaluation]"'
        )

    if isinstance(source, pd.DataFrame):
        return source.copy()
    elif isinstance(source, dict):
        return pd.DataFrame(source)
    elif isinstance(source, str):
        if source.startswith(_BQ_PREFIX):
            return _load_bigquery(source[len(_BQ_PREFIX) :])

        _, extension = os.path.splitext(source)
        file_type = extension.lower()[1:]

        if file_type == "jsonl":
            return _load_jsonl(source)
        elif file_type == "csv":
            return _load_csv(source)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    else:
        raise TypeError(
            "Unsupported dataset type. Must be DataFrame, dictionary, or" " filepath."
        )


def _load_jsonl(filepath: str) -> "pd.DataFrame":
    """Loads data from a JSONL file into a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Pandas is not installed. Please install the SDK using "pip install'
            ' google-cloud-aiplatform[rapid_evaluation]"'
        )
    if filepath.startswith(_GCS_PREFIX):
        file_contents = _read_gcs_file_contents(filepath)
        return pd.read_json(file_contents, lines=True)
    else:
        with open(filepath, "r") as f:
            return pd.read_json(f, lines=True)


def _load_csv(filepath: str) -> "pd.DataFrame":
    """Loads data from a CSV file into a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            'Pandas is not installed. Please install the SDK using "pip install'
            ' google-cloud-aiplatform[rapid_evaluation]"'
        )
    if filepath.startswith(_GCS_PREFIX):
        file_contents = _read_gcs_file_contents(filepath)
        return pd.read_csv(io.StringIO(file_contents), encoding="utf-8")
    else:
        return pd.read_csv(filepath, encoding="utf-8")


def _load_bigquery(table_id: str) -> "pd.DataFrame":
    """Loads data from a BigQuery table into a DataFrame."""

    client = bigquery.Client(
        project=initializer.global_config.project,
        credentials=initializer.global_config.credentials,
    )
    table = client.get_table(table_id)
    return client.list_rows(table).to_dataframe()


def _read_gcs_file_contents(filepath: str) -> str:
    """Reads the contents of a file from Google Cloud Storage.

    Args:
        filepath: The GCS file path (e.g., 'gs://bucket_name/file.csv')

    Returns:
        The contents of the file.
    """

    client = storage.Client(
        project=initializer.global_config.project,
        credentials=initializer.global_config.credentials,
    )
    bucket_name, blob_path = filepath[len(_GCS_PREFIX) :].split("/", 1)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_string().decode("utf-8")
