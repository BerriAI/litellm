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

from types import ModuleType
from typing import Any, Dict, Optional
from urllib import request

from google.auth import credentials as auth_credentials
from google.auth import transport
from google.cloud import storage
from google.cloud.aiplatform.constants import pipeline as pipeline_constants

# Pattern for an Artifact Registry URL.
_VALID_AR_URL = pipeline_constants._VALID_AR_URL

# Pattern for any JSON or YAML file over HTTPS.
_VALID_HTTPS_URL = pipeline_constants._VALID_HTTPS_URL


def load_yaml(
    path: str,
    project: Optional[str] = None,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> Dict[str, Any]:
    """Loads data from a YAML document.

    Args:
      path (str):
          Required. The path of the YAML document. It can be a local path, a
          Google Cloud Storage URI, an Artifact Registry URI, or an HTTPS URI.
      project (str):
          Optional. Project to initiate the Storage client with.
      credentials (auth_credentials.Credentials):
          Optional. Credentials to use with Storage Client.

    Returns:
      A Dict object representing the YAML document.
    """
    if path.startswith("gs://"):
        return _load_yaml_from_gs_uri(path, project, credentials)
    elif path.startswith("http://") or path.startswith("https://"):
        if _VALID_AR_URL.match(path):
            return _load_yaml_from_https_uri(path, credentials)
        elif _VALID_HTTPS_URL.match(path):
            return _load_yaml_from_https_uri(path)
        else:
            raise ValueError(
                "Invalid HTTPS URI. If not using Artifact Registry, please "
                "ensure the URI ends with .json, .yaml, or .yml."
            )
    else:
        return _load_yaml_from_local_file(path)


def _maybe_import_yaml() -> ModuleType:
    """Tries to import the PyYAML module."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is not installed and is required to parse PipelineJob or "
            'PipelineSpec files. Please install the SDK using "pip install '
            'google-cloud-aiplatform[pipelines]"'
        )
    return yaml


def _load_yaml_from_gs_uri(
    uri: str,
    project: Optional[str] = None,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> Dict[str, Any]:
    """Loads data from a YAML document referenced by a GCS URI.

    Args:
      path (str):
          Required. GCS URI for YAML document.
      project (str):
          Optional. Project to initiate the Storage client with.
      credentials (auth_credentials.Credentials):
          Optional. Credentials to use with Storage Client.

    Returns:
      A Dict object representing the YAML document.
    """
    yaml = _maybe_import_yaml()
    storage_client = storage.Client(project=project, credentials=credentials)
    blob = storage.Blob.from_string(uri, storage_client)
    return yaml.safe_load(blob.download_as_bytes())


def _load_yaml_from_local_file(file_path: str) -> Dict[str, Any]:
    """Loads data from a YAML local file.

    Args:
      file_path (str):
          Required. The local file path of the YAML document.

    Returns:
      A Dict object representing the YAML document.
    """
    yaml = _maybe_import_yaml()
    with open(file_path) as f:
        return yaml.safe_load(f)


def _load_yaml_from_https_uri(
    uri: str,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> Dict[str, Any]:
    """Loads data from a YAML document referenced by a Artifact Registry URI.

    Args:
      uri (str):
          Required. Artifact Registry URI for YAML document.
      credentials (auth_credentials.Credentials):
          Optional. Credentials to use with Artifact Registry.

    Returns:
      A Dict object representing the YAML document.
    """
    yaml = _maybe_import_yaml()
    req = request.Request(uri)

    if credentials:
        if not credentials.valid:
            credentials.refresh(transport.requests.Request())
        if credentials.token:
            req.add_header("Authorization", "Bearer " + credentials.token)
    response = request.urlopen(req)

    return yaml.safe_load(response.read().decode("utf-8"))
