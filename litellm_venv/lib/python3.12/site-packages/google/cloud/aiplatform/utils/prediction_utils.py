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

import shutil
import inspect
import logging
import os
from pathlib import Path
import re
from typing import Any, Optional, Sequence, Tuple, Type

from google.cloud import storage
from google.cloud.aiplatform.constants import prediction
from google.cloud.aiplatform.utils import path_utils

_logger = logging.getLogger(__name__)


REGISTRY_REGEX = re.compile(r"^([\w\-]+\-docker\.pkg\.dev|([\w]+\.|)gcr\.io)")
GCS_URI_PREFIX = "gs://"


def inspect_source_from_class(
    custom_class: Type[Any],
    src_dir: str,
) -> Tuple[str, str]:
    """Inspects the source file from a custom class and returns its import path.

    Args:
        custom_class (Type[Any]):
            Required. The custom class needs to be inspected for the source file.
        src_dir (str):
            Required. The path to the local directory including all needed files.
            The source file of the custom class must be in this directory.

    Returns:
        (import_from, class_name): the source file path in python import format
            and the custom class name.

    Raises:
        ValueError: If the source file of the custom class is not in the source
            directory.
    """
    src_dir_abs_path = Path(src_dir).expanduser().resolve()

    custom_class_name = custom_class.__name__

    custom_class_path = Path(inspect.getsourcefile(custom_class)).resolve()
    if not path_utils._is_relative_to(custom_class_path, src_dir_abs_path):
        raise ValueError(
            f'The file implementing "{custom_class_name}" must be in "{src_dir}".'
        )

    custom_class_import_path = custom_class_path.relative_to(src_dir_abs_path)
    custom_class_import_path = custom_class_import_path.with_name(
        custom_class_import_path.stem
    )
    custom_class_import = custom_class_import_path.as_posix().replace(os.sep, ".")

    return custom_class_import, custom_class_name


def is_registry_uri(image_uri: str) -> bool:
    """Checks whether the image uri is in container registry or artifact registry.

    Args:
        image_uri (str):
            The image uri to check if it is in container registry or artifact registry.

    Returns:
        True if the image uri is in container registry or artifact registry.
    """
    return REGISTRY_REGEX.match(image_uri) is not None


def get_prediction_aip_http_port(
    serving_container_ports: Optional[Sequence[int]] = None,
) -> int:
    """Gets the used prediction container port from serving container ports.

    If containerSpec.ports is specified during Model or LocalModel creation time, retrieve
    the first entry in this field. Otherwise use the default value of 8080. The environment
    variable AIP_HTTP_PORT will be set to this value.
    See https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements
    for more details.

    Args:
        serving_container_ports (Sequence[int]):
            Optional. Declaration of ports that are exposed by the container. This field is
            primarily informational, it gives Vertex AI information about the
            network connections the container uses. Listing or not a port here has
            no impact on whether the port is actually exposed, any port listening on
            the default "0.0.0.0" address inside a container will be accessible from
            the network.

    Returns:
        The first element in the serving_container_ports. If there is no any values in it,
        return the default http port.
    """
    return (
        serving_container_ports[0]
        if serving_container_ports is not None and len(serving_container_ports) > 0
        else prediction.DEFAULT_AIP_HTTP_PORT
    )


def download_model_artifacts(artifact_uri: str) -> None:
    """Prepares model artifacts in the current working directory.

    If artifact_uri is a GCS uri, the model artifacts will be downloaded to the current
    working directory.
    If artifact_uri is a local directory, the model artifacts will be copied to the current
    working directory.

    Args:
        artifact_uri (str):
            Required. The artifact uri that includes model artifacts.
    """
    if artifact_uri.startswith(GCS_URI_PREFIX):
        matches = re.match(f"{GCS_URI_PREFIX}(.*?)/(.*)", artifact_uri)
        bucket_name, prefix = matches.groups()

        gcs_client = storage.Client()
        blobs = gcs_client.list_blobs(bucket_name, prefix=prefix)
        for blob in blobs:
            name_without_prefix = blob.name[len(prefix) :]
            name_without_prefix = (
                name_without_prefix[1:]
                if name_without_prefix.startswith("/")
                else name_without_prefix
            )
            file_split = name_without_prefix.split("/")
            directory = "/".join(file_split[0:-1])
            Path(directory).mkdir(parents=True, exist_ok=True)
            if name_without_prefix and not name_without_prefix.endswith("/"):
                blob.download_to_filename(name_without_prefix)
    else:
        # Copy files to the current working directory.
        shutil.copytree(artifact_uri, ".", dirs_exist_ok=True)
