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

import collections

try:
    import docker
except ImportError:
    raise ImportError(
        "Docker is not installed and is required to run containers. "
        'Please install the SDK using `pip install "google-cloud-aiplatform[prediction]>=1.16.0"`.'
    )


Package = collections.namedtuple("Package", ["script", "package_path", "python_module"])
Image = collections.namedtuple("Image", ["name", "default_home", "default_workdir"])
DEFAULT_HOME = "/home"
DEFAULT_WORKDIR = "/usr/app"
DEFAULT_MOUNTED_MODEL_DIRECTORY = "/tmp_cpr_local_model"


def check_image_exists_locally(image_name: str) -> bool:
    """Checks if an image exists locally.

    Args:
        image_name (str):
            Required. The name of the image.

    Returns:
        Whether the image exists locally.
    """
    client = docker.from_env()
    try:
        _ = client.images.get(image_name)
        return True
    except (docker.errors.ImageNotFound, docker.errors.APIError):
        return False
