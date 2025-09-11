# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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

import google.auth
import google.auth.transport.requests
import logging
import ray
import re
from immutabledict import immutabledict

from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.utils import resource_manager_utils

SUPPORTED_RAY_VERSIONS = immutabledict({"2.4": "2.4.0", "2.9": "2.9.3"})
SUPPORTED_PY_VERSION = ["3.10"]

# Artifact Repository available regions.
_AVAILABLE_REGIONS = ["us", "europe", "asia"]
# If region is not available, assume using the default region.
_DEFAULT_REGION = "us"

_PERSISTENT_RESOURCE_NAME_PATTERN = "projects/{}/locations/{}/persistentResources/{}"
_VALID_RESOURCE_NAME_REGEX = "[a-z][a-zA-Z0-9._-]{0,127}"
_DASHBOARD_URI_SUFFIX = "aiplatform-training.googleusercontent.com"


def valid_resource_name(resource_name):
    """Check if address is a valid resource name."""
    resource_name_split = resource_name.split("/")
    if not (
        len(resource_name_split) == 6
        and resource_name_split[0] == "projects"
        and resource_name_split[2] == "locations"
        and resource_name_split[4] == "persistentResources"
    ):
        raise ValueError(
            "[Ray on Vertex AI]: Address must be in the following "
            "format: vertex_ray://projects/<project_num>/locations/<region>/persistentResources/<pr_id> "
            "or vertex_ray://<pr_id>."
        )


def maybe_reconstruct_resource_name(address) -> str:
    """Reconstruct full persistent resource name if only id was given."""
    if re.match("^{}$".format(_VALID_RESOURCE_NAME_REGEX), address):
        # Assume only cluster name (persistent resource id) was given.
        logging.info(
            "[Ray on Vertex AI]: Cluster name was given as address, reconstructing full resource name"
        )
        return _PERSISTENT_RESOURCE_NAME_PATTERN.format(
            resource_manager_utils.get_project_number(
                initializer.global_config.project
            ),
            initializer.global_config.location,
            address,
        )

    return address


def get_local_ray_version():
    ray_version = ray.__version__.split(".")
    if len(ray_version) == 3:
        ray_version = ray_version[:2]
    return ".".join(ray_version)


def get_image_uri(ray_version, python_version, enable_cuda):
    """Image uri for a given ray version and python version."""
    if ray_version not in SUPPORTED_RAY_VERSIONS:
        raise ValueError(
            "[Ray on Vertex AI]: The supported Ray versions are %s (%s) and %s (%s)."
            % (
                list(SUPPORTED_RAY_VERSIONS.keys())[0],
                list(SUPPORTED_RAY_VERSIONS.values())[0],
                list(SUPPORTED_RAY_VERSIONS.keys())[1],
                list(SUPPORTED_RAY_VERSIONS.values())[1],
            )
        )
    if python_version not in SUPPORTED_PY_VERSION:
        raise ValueError("[Ray on Vertex AI]: The supported Python version is 3.10.")

    location = initializer.global_config.location
    region = location.split("-")[0]
    if region not in _AVAILABLE_REGIONS:
        region = _DEFAULT_REGION
    ray_version = ray_version.replace(".", "-")
    if enable_cuda:
        return f"{region}-docker.pkg.dev/vertex-ai/training/ray-gpu.{ray_version}.py310:latest"
    else:
        return f"{region}-docker.pkg.dev/vertex-ai/training/ray-cpu.{ray_version}.py310:latest"


def get_versions_from_image_uri(image_uri):
    """Get ray version and python version from image uri."""
    logging.info(f"[Ray on Vertex AI]: Getting versions from image uri: {image_uri}")
    image_label = image_uri.split("/")[-1].split(":")[0]
    py_version = image_label[-3] + "." + image_label[-2:]
    ray_version = image_label.split(".")[1].replace("-", ".")
    if ray_version in SUPPORTED_RAY_VERSIONS and py_version in SUPPORTED_PY_VERSION:
        return py_version, ray_version
    else:
        # May not parse custom image and get the versions correctly
        return None, None


def valid_dashboard_address(address):
    """Check if address is a valid dashboard uri."""
    return address.endswith(_DASHBOARD_URI_SUFFIX)


def get_bearer_token():
    """Get bearer token through Application Default Credentials."""
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # creds.valid is False, and creds.token is None
    # Need to refresh credentials to populate those
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token
