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

import re
from typing import Optional
import warnings

from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.constants import prediction
from packaging import version


def get_prebuilt_prediction_container_uri(
    framework: str,
    framework_version: str,
    region: Optional[str] = None,
    accelerator: str = "cpu",
) -> str:
    """
    Get a Vertex AI pre-built prediction Docker container URI for
    a given framework, version, region, and accelerator use.

    Example usage:
    ```
        uri = aiplatform.helpers.get_prebuilt_prediction_container_uri(
                framework="tensorflow",
                framework_version="2.6",
                accelerator="gpu"
        )

        model = aiplatform.Model.upload(
            display_name="boston_housing_",
            artifact_uri="gs://my-bucket/my-model/",
            serving_container_image_uri=uri
        )
    ```

    Args:
        framework (str):
            Required. The ML framework of the pre-built container. For example,
            `"tensorflow"`, `"xgboost"`, or `"sklearn"`
        framework_version (str):
            Required. The version of the specified ML framework as a string.
        region (str):
            Optional. AI region or multi-region. Used to select the correct
            Artifact Registry multi-region repository and reduce latency.
            Must start with `"us"`, `"asia"` or `"europe"`.
            Default is location set by `aiplatform.init()`.
        accelerator (str):
            Optional. The type of accelerator support provided by container. For
            example: `"cpu"` or `"gpu"`
            Default is `"cpu"`.

    Returns:
        uri (str):
            A Vertex AI prediction container URI

    Raises:
        ValueError: If containers for provided framework are unavailable or the
        container does not support the specified version, accelerator, or region.
    """
    URI_MAP = prediction._SERVING_CONTAINER_URI_MAP
    DOCS_URI_MESSAGE = (
        f"See {prediction._SERVING_CONTAINER_DOCUMENTATION_URL} "
        "for complete list of supported containers"
    )

    # If region not provided, use initializer location
    region = region or initializer.global_config.location
    region = region.split("-", 1)[0]
    framework = framework.lower()

    if not URI_MAP.get(region):
        raise ValueError(
            f"Unsupported container region `{region}`, supported regions are "
            f"{', '.join(URI_MAP.keys())}. "
            f"{DOCS_URI_MESSAGE}"
        )

    if not URI_MAP[region].get(framework):
        raise ValueError(
            f"No containers found for framework `{framework}`. Supported frameworks are "
            f"{', '.join(URI_MAP[region].keys())} {DOCS_URI_MESSAGE}"
        )

    if not URI_MAP[region][framework].get(accelerator):
        raise ValueError(
            f"{framework} containers do not support `{accelerator}` accelerator. Supported accelerators "
            f"are {', '.join(URI_MAP[region][framework].keys())}. {DOCS_URI_MESSAGE}"
        )

    final_uri = URI_MAP[region][framework][accelerator].get(framework_version)

    if not final_uri:
        raise ValueError(
            f"No serving container for `{framework}` version `{framework_version}` "
            f"with accelerator `{accelerator}` found. Supported versions "
            f"include {', '.join(URI_MAP[region][framework][accelerator].keys())}. {DOCS_URI_MESSAGE}"
        )

    return final_uri


def is_prebuilt_prediction_container_uri(image_uri: str) -> bool:
    """Checks whether the image is pre-built by Vertex AI prediction.

    Args:
        image_uri (str):
            Required. The image uri to be checked whether it is prebuilt by Vertex
            AI prediction.

    Returns:
        If the image is prebuilt by Vertex AI prediction.
    """
    return re.fullmatch(prediction.CONTAINER_URI_REGEX, image_uri) is not None


# TODO(b/264191784) Deduplicate this method
def _get_closest_match_prebuilt_container_uri(
    framework: str,
    framework_version: str,
    region: Optional[str] = None,
    accelerator: str = "cpu",
) -> str:
    """Return a pre-built container uri that is suitable for a specific framework and version.

    If there is no exact match for the given version, the closest one that is
    higher than the input version will be used.

    Args:
        framework (str):
            Required. The ML framework of the pre-built container. For example,
            `"tensorflow"`, `"xgboost"`, or `"sklearn"`
        framework_version (str):
            Required. The version of the specified ML framework as a string.
        region (str):
            Optional. AI region or multi-region. Used to select the correct
            Artifact Registry multi-region repository and reduce latency.
            Must start with `"us"`, `"asia"` or `"europe"`.
            Default is location set by `aiplatform.init()`.
        accelerator (str):
            Optional. The type of accelerator support provided by container. For
            example: `"cpu"` or `"gpu"`
            Default is `"cpu"`.

    Returns:
        A string representing the pre-built container uri.

    Raises:
        ValueError: If the framework doesn't have suitable pre-built container.
    """
    URI_MAP = prediction._SERVING_CONTAINER_URI_MAP
    DOCS_URI_MESSAGE = (
        f"See {prediction._SERVING_CONTAINER_DOCUMENTATION_URL} "
        "for complete list of supported containers"
    )

    # If region not provided, use initializer location
    region = region or initializer.global_config.location
    region = region.split("-", 1)[0]
    framework = framework.lower()

    if not URI_MAP.get(region):
        raise ValueError(
            f"Unsupported container region `{region}`, supported regions are "
            f"{', '.join(URI_MAP.keys())}. "
            f"{DOCS_URI_MESSAGE}"
        )

    if not URI_MAP[region].get(framework):
        raise ValueError(
            f"No containers found for framework `{framework}`. Supported frameworks are "
            f"{', '.join(URI_MAP[region].keys())} {DOCS_URI_MESSAGE}"
        )

    if not URI_MAP[region][framework].get(accelerator):
        raise ValueError(
            f"{framework} containers do not support `{accelerator}` accelerator. Supported accelerators "
            f"are {', '.join(URI_MAP[region][framework].keys())}. {DOCS_URI_MESSAGE}"
        )

    framework_version = version.Version(framework_version)
    available_version_list = [
        version.Version(available_version)
        for available_version in URI_MAP[region][framework][accelerator].keys()
    ]
    try:
        closest_version = min(
            [
                available_version
                for available_version in available_version_list
                if available_version >= framework_version
                # manually implement Version.major for packaging < 20.0
                and available_version._version.release[0]
                == framework_version._version.release[0]
            ]
        )
    except ValueError:
        raise ValueError(
            f"You are using `{framework}` version `{framework_version}`. "
            f"Vertex pre-built containers support up to `{framework}` version "
            f"`{max(available_version_list)}` and don't assume forward compatibility. "
            f"Please build your own custom container. {DOCS_URI_MESSAGE}"
        ) from None

    if closest_version != framework_version:
        warnings.warn(
            f"No exact match for `{framework}` version `{framework_version}`. "
            f"Pre-built container for `{framework}` version `{closest_version}` is used. "
            f"{DOCS_URI_MESSAGE}"
        )

    final_uri = URI_MAP[region][framework][accelerator].get(str(closest_version))

    return final_uri
