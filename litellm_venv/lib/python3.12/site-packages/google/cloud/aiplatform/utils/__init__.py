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


import abc
import datetime
import pathlib
import logging
import re
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Tuple
import uuid

from google.protobuf import timestamp_pb2

from google.api_core import client_options
from google.api_core import gapic_v1
from google.auth import credentials as auth_credentials
from google.cloud import storage

from google.cloud.aiplatform import compat
from google.cloud.aiplatform.constants import base as constants
from google.cloud.aiplatform import initializer

from google.cloud.aiplatform.compat.services import (
    dataset_service_client_v1beta1,
    deployment_resource_pool_service_client_v1beta1,
    endpoint_service_client_v1beta1,
    extension_execution_service_client_v1beta1,
    extension_registry_service_client_v1beta1,
    feature_online_store_admin_service_client_v1beta1,
    feature_online_store_service_client_v1beta1,
    featurestore_online_serving_service_client_v1beta1,
    featurestore_service_client_v1beta1,
    index_service_client_v1beta1,
    index_endpoint_service_client_v1beta1,
    job_service_client_v1beta1,
    match_service_client_v1beta1,
    metadata_service_client_v1beta1,
    model_service_client_v1beta1,
    pipeline_service_client_v1beta1,
    prediction_service_client_v1beta1,
    prediction_service_async_client_v1beta1,
    schedule_service_client_v1beta1,
    tensorboard_service_client_v1beta1,
    vizier_service_client_v1beta1,
    model_garden_service_client_v1beta1,
    persistent_resource_service_client_v1beta1,
    reasoning_engine_service_client_v1beta1,
    reasoning_engine_execution_service_client_v1beta1,
)
from google.cloud.aiplatform.compat.services import (
    dataset_service_client_v1,
    endpoint_service_client_v1,
    feature_online_store_admin_service_client_v1,
    feature_online_store_service_client_v1,
    featurestore_online_serving_service_client_v1,
    featurestore_service_client_v1,
    index_service_client_v1,
    index_endpoint_service_client_v1,
    job_service_client_v1,
    metadata_service_client_v1,
    model_garden_service_client_v1,
    model_service_client_v1,
    pipeline_service_client_v1,
    prediction_service_client_v1,
    prediction_service_async_client_v1,
    schedule_service_client_v1,
    tensorboard_service_client_v1,
    vizier_service_client_v1,
    persistent_resource_service_client_v1,
)

from google.cloud.aiplatform.compat.types import (
    accelerator_type as gca_accelerator_type,
)

VertexAiServiceClient = TypeVar(
    "VertexAiServiceClient",
    # v1beta1
    dataset_service_client_v1beta1.DatasetServiceClient,
    deployment_resource_pool_service_client_v1beta1.DeploymentResourcePoolServiceClient,
    endpoint_service_client_v1beta1.EndpointServiceClient,
    feature_online_store_admin_service_client_v1beta1.FeatureOnlineStoreAdminServiceClient,
    feature_online_store_service_client_v1beta1.FeatureOnlineStoreServiceClient,
    featurestore_online_serving_service_client_v1beta1.FeaturestoreOnlineServingServiceClient,
    featurestore_service_client_v1beta1.FeaturestoreServiceClient,
    index_service_client_v1beta1.IndexServiceClient,
    index_endpoint_service_client_v1beta1.IndexEndpointServiceClient,
    model_service_client_v1beta1.ModelServiceClient,
    prediction_service_client_v1beta1.PredictionServiceClient,
    prediction_service_async_client_v1beta1.PredictionServiceAsyncClient,
    pipeline_service_client_v1beta1.PipelineServiceClient,
    job_service_client_v1beta1.JobServiceClient,
    match_service_client_v1beta1.MatchServiceClient,
    metadata_service_client_v1beta1.MetadataServiceClient,
    schedule_service_client_v1beta1.ScheduleServiceClient,
    tensorboard_service_client_v1beta1.TensorboardServiceClient,
    vizier_service_client_v1beta1.VizierServiceClient,
    # v1
    dataset_service_client_v1.DatasetServiceClient,
    endpoint_service_client_v1.EndpointServiceClient,
    feature_online_store_admin_service_client_v1.FeatureOnlineStoreAdminServiceClient,
    feature_online_store_service_client_v1.FeatureOnlineStoreServiceClient,
    featurestore_online_serving_service_client_v1.FeaturestoreOnlineServingServiceClient,
    featurestore_service_client_v1.FeaturestoreServiceClient,
    metadata_service_client_v1.MetadataServiceClient,
    model_service_client_v1.ModelServiceClient,
    prediction_service_client_v1.PredictionServiceClient,
    prediction_service_async_client_v1.PredictionServiceAsyncClient,
    pipeline_service_client_v1.PipelineServiceClient,
    job_service_client_v1.JobServiceClient,
    schedule_service_client_v1.ScheduleServiceClient,
    tensorboard_service_client_v1.TensorboardServiceClient,
    vizier_service_client_v1.VizierServiceClient,
)


RESOURCE_ID_PATTERN = re.compile(r"^[\w-]+$")


def validate_id(resource_id: str):
    """Validate resource ID.

    Args:
        resource_id (str): Resource id.
    Raises:
        ValueError: If resource id is not a valid format.

    """
    if not RESOURCE_ID_PATTERN.match(resource_id):
        raise ValueError(f"Resource {resource_id} is not a valid resource id.")


def full_resource_name(
    resource_name: str,
    resource_noun: str,
    parse_resource_name_method: Callable[[str], Dict[str, str]],
    format_resource_name_method: Callable[..., str],
    parent_resource_name_fields: Optional[Dict[str, str]] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    resource_id_validator: Optional[Callable[[str], None]] = None,
) -> str:
    """Returns fully qualified resource name.

    Args:
        resource_name (str):
            Required. A fully-qualified Vertex AI resource name or
            resource ID.
        resource_noun (str):
            Required. A resource noun to validate the resource name against.
            For example, you would pass "datasets" to validate
            "projects/123/locations/us-central1/datasets/456".
        parse_resource_name_method (Callable[[str], Dict[str,str]]):
            Required. Method that parses a resource name into its segment parts.
            These are generally included with GAPIC clients.
        format_resource_name_method (Callable[..., str]):
            Required. Method that takes segment parts of resource names and returns
            the formated resource name. These are generally included with GAPIC clients.
        parent_resource_name_fields (Dict[str, str]):
            Optional. Dictionary of segment parts where key is the resource noun and
            values are the resource ids.
            For example:
                {
                    "metadataStores": "123"
                }
        project (str):
            Optional. project to retrieve resource_noun from. If not set, project
            set in aiplatform.init will be used.
        location (str):
            Optional. location to retrieve resource_noun from. If not set, location
            set in aiplatform.init will be used.
        resource_id_validator (Callable[str, None]):
            Optional. Function that validates the resource ID. Overrides the default validator, validate_id.
            Should take a resource ID as string and raise ValueError if invalid.

    Returns:
        resource_name (str):
            A fully-qualified Vertex AI resource name.
    """
    # Fully qualified resource name, e.g., "projects/.../locations/.../datasets/12345" or
    # "projects/.../locations/.../metadataStores/.../contexts/12345"
    fields = parse_resource_name_method(resource_name)
    if fields:
        return resource_name

    resource_id_validator = resource_id_validator or validate_id

    user_project = project or initializer.global_config.project
    user_location = location or initializer.global_config.location

    validate_region(user_location)
    resource_id_validator(resource_name)

    format_args = {
        "location": user_location,
        "project": user_project,
        convert_camel_case_resource_noun_to_snake_case(resource_noun): resource_name,
    }

    if parent_resource_name_fields:
        format_args.update(
            {
                convert_camel_case_resource_noun_to_snake_case(key): value
                for key, value in parent_resource_name_fields.items()
            }
        )

    return format_resource_name_method(**format_args)


# Resource nouns that are not plural in their resource names.
# Userd below to avoid conversion from plural to singular.
_SINGULAR_RESOURCE_NOUNS = {"time_series"}
_SINGULAR_RESOURCE_NOUNS_MAP = {"indexes": "index"}


def convert_camel_case_resource_noun_to_snake_case(resource_noun: str) -> str:
    """Converts camel case to snake case to map resource name parts to GAPIC parameter names.

    Args:
        resource_noun (str): The resource noun in camel case to covert.
    Returns:
        Singular snake case resource noun.
    """
    snake_case = re.sub("([A-Z]+)", r"_\1", resource_noun).lower()

    # plural to singular
    if snake_case in _SINGULAR_RESOURCE_NOUNS or not snake_case.endswith("s"):
        return snake_case
    elif snake_case in _SINGULAR_RESOURCE_NOUNS_MAP:
        return _SINGULAR_RESOURCE_NOUNS_MAP[snake_case]
    else:
        return snake_case[:-1]


def validate_display_name(display_name: str):
    """Verify display name is at most 128 chars.

    Args:
        display_name: display name to verify
    Raises:
        ValueError: display name is longer than 128 characters
    """
    if len(display_name) > 128:
        raise ValueError("Display name needs to be less than 128 characters.")


def validate_labels(labels: Dict[str, str]):
    """Validate labels.

    Args:
        labels: labels to verify
    Raises:
        ValueError: if labels is not a mapping of string key value pairs.
    """
    for k, v in labels.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                "Expect labels to be a mapping of string key value pairs. "
                'Got "{}".'.format(labels)
            )


def validate_region(region: str) -> bool:
    """Validates region against supported regions.

    Args:
        region: region to validate
    Returns:
        bool: True if no errors raised
    Raises:
        ValueError: If region is not in supported regions.
    """
    if not region:
        raise ValueError(
            f"Please provide a region, select from {constants.SUPPORTED_REGIONS}"
        )

    region = region.lower()
    if region not in constants.SUPPORTED_REGIONS:
        raise ValueError(
            f"Unsupported region for Vertex AI, select from {constants.SUPPORTED_REGIONS}"
        )

    return True


def validate_accelerator_type(accelerator_type: str) -> bool:
    """Validates user provided accelerator_type string for training and
    prediction.

    Args:
        accelerator_type (str):
            Represents a hardware accelerator type.
    Returns:
        bool: True if valid accelerator_type
    Raises:
        ValueError if accelerator type is invalid.
    """
    if accelerator_type not in gca_accelerator_type.AcceleratorType._member_names_:
        raise ValueError(
            f"Given accelerator_type `{accelerator_type}` invalid. "
            f"Choose one of {gca_accelerator_type.AcceleratorType._member_names_}"
        )
    return True


def extract_bucket_and_prefix_from_gcs_path(gcs_path: str) -> Tuple[str, Optional[str]]:
    """Given a complete GCS path, return the bucket name and prefix as a tuple.

    Example Usage:

        bucket, prefix = extract_bucket_and_prefix_from_gcs_path(
            "gs://example-bucket/path/to/folder"
        )

        # bucket = "example-bucket"
        # prefix = "path/to/folder"

    Args:
        gcs_path (str):
            Required. A full path to a Google Cloud Storage folder or resource.
            Can optionally include "gs://" prefix or end in a trailing slash "/".

    Returns:
        Tuple[str, Optional[str]]
            A (bucket, prefix) pair from provided GCS path. If a prefix is not
            present, a None will be returned in its place.
    """
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    if gcs_path.endswith("/"):
        gcs_path = gcs_path[:-1]

    gcs_parts = gcs_path.split("/", 1)
    gcs_bucket = gcs_parts[0]
    gcs_blob_prefix = None if len(gcs_parts) == 1 else gcs_parts[1]

    return (gcs_bucket, gcs_blob_prefix)


def extract_project_and_location_from_parent(
    parent: str,
) -> Dict[str, str]:
    """Given a complete parent resource name, return the project and location as a dict.

    Example Usage:

        parent_resources = extract_project_and_location_from_parent(
            "projects/123/locations/us-central1/datasets/456"
        )

        parent_resources["project"] = "123"
        parent_resources["location"] = "us-central1"

    Args:
        parent (str):
            Required. A complete parent resource name.

    Returns:
        Dict[str, str]
            A project, location dict from provided parent resource name.
    """
    parent_resources = re.match(
        r"^projects/(?P<project>.+?)/locations/(?P<location>.+?)(/|$)", parent
    )
    return parent_resources.groupdict() if parent_resources else {}


class ClientWithOverride:
    class WrappedClient:
        """Wrapper class for client that creates client at API invocation
        time."""

        def __init__(
            self,
            client_class: Type[VertexAiServiceClient],
            client_options: client_options.ClientOptions,
            client_info: gapic_v1.client_info.ClientInfo,
            credentials: Optional[auth_credentials.Credentials] = None,
            transport: Optional[str] = None,
        ):
            """Stores parameters needed to instantiate client.

            Args:
                client_class (VertexAiServiceClient):
                    Required. Class of the client to use.
                client_options (client_options.ClientOptions):
                    Required. Client options to pass to client.
                client_info (gapic_v1.client_info.ClientInfo):
                    Required. Client info to pass to client.
                credentials (auth_credentials.credentials):
                    Optional. Client credentials to pass to client.
                transport (str):
                    Optional. Transport type to pass to client.
                    NOTE: "rest" transport functionality is currently in a
                    beta state (preview).
            """

            self._client_class = client_class
            self._credentials = credentials
            self._client_options = client_options
            self._client_info = client_info
            self._api_transport = transport

        def __getattr__(self, name: str) -> Any:
            """Instantiates client and returns attribute of the client."""

            kwargs = dict(
                credentials=self._credentials,
                client_options=self._client_options,
                client_info=self._client_info,
            )

            if self._api_transport is not None:
                kwargs["transport"] = self._api_transport

            temporary_client = self._client_class(**kwargs)

            return getattr(temporary_client, name)

    @property
    @abc.abstractmethod
    def _is_temporary(self) -> bool:
        pass

    @property
    @classmethod
    @abc.abstractmethod
    def _default_version(self) -> str:
        pass

    @property
    @classmethod
    @abc.abstractmethod
    def _version_map(self) -> Tuple:
        pass

    @property
    def api_endpoint(self) -> str:
        """Default API endpoint used by this client."""
        client = self._clients[self._default_version]

        if self._is_temporary:
            return client._client_options.api_endpoint
        else:
            return client._transport._host.split(":")[0]

    def __init__(
        self,
        client_options: client_options.ClientOptions,
        client_info: gapic_v1.client_info.ClientInfo,
        credentials: Optional[auth_credentials.Credentials] = None,
        transport: Optional[str] = None,
    ):
        """Stores parameters needed to instantiate client.

        Args:
            client_options (client_options.ClientOptions):
                Required. Client options to pass to client.
            client_info (gapic_v1.client_info.ClientInfo):
                Required. Client info to pass to client.
            credentials (auth_credentials.credentials):
                Optional. Client credentials to pass to client.
            transport (str):
                Optional. Transport type to pass to client.
                NOTE: "rest" transport functionality is currently in a
                beta state (preview).
        """
        kwargs = dict(
            credentials=credentials,
            client_options=client_options,
            client_info=client_info,
        )

        if transport is not None:
            kwargs["transport"] = transport

        self._clients = {
            version: self.WrappedClient(
                client_class=client_class,
                client_options=client_options,
                client_info=client_info,
                credentials=credentials,
                transport=transport,
            )
            if self._is_temporary
            else client_class(**kwargs)
            for version, client_class in self._version_map
        }

    def __getattr__(self, name: str) -> Any:
        """Instantiates client and returns attribute of the client."""
        return getattr(self._clients[self._default_version], name)

    def select_version(self, version: str) -> VertexAiServiceClient:
        return self._clients[version]

    @classmethod
    def get_gapic_client_class(
        cls, version: Optional[str] = None
    ) -> Type[VertexAiServiceClient]:
        """Gets the underyilng GAPIC client.

        Used to access class and static methods without instantiating.

        Args:
            version (str):
                Optional. Version of client to retreive otherwise the default version is returned.
        Retuns:
            Underlying GAPIC client for this wrapper and version.
        """
        return dict(cls._version_map)[version or cls._default_version]


class DatasetClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, dataset_service_client_v1.DatasetServiceClient),
        (compat.V1BETA1, dataset_service_client_v1beta1.DatasetServiceClient),
    )


class DeploymentResourcePoolClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.V1BETA1
    _version_map = (
        (
            compat.V1BETA1,
            deployment_resource_pool_service_client_v1beta1.DeploymentResourcePoolServiceClient,
        ),
    )


class EndpointClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, endpoint_service_client_v1.EndpointServiceClient),
        (compat.V1BETA1, endpoint_service_client_v1beta1.EndpointServiceClient),
    )


class ExtensionExecutionClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.V1BETA1
    _version_map = (
        (
            compat.V1BETA1,
            extension_execution_service_client_v1beta1.ExtensionExecutionServiceClient,
        ),
    )


class ExtensionRegistryClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.V1BETA1
    _version_map = (
        (
            compat.V1BETA1,
            extension_registry_service_client_v1beta1.ExtensionRegistryServiceClient,
        ),
    )


class IndexClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, index_service_client_v1.IndexServiceClient),
        (compat.V1BETA1, index_service_client_v1beta1.IndexServiceClient),
    )


class IndexEndpointClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, index_endpoint_service_client_v1.IndexEndpointServiceClient),
        (
            compat.V1BETA1,
            index_endpoint_service_client_v1beta1.IndexEndpointServiceClient,
        ),
    )


class FeatureOnlineStoreAdminClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (
            compat.V1,
            feature_online_store_admin_service_client_v1.FeatureOnlineStoreAdminServiceClient,
        ),
        (
            compat.V1BETA1,
            feature_online_store_admin_service_client_v1beta1.FeatureOnlineStoreAdminServiceClient,
        ),
    )


class FeatureOnlineStoreClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (
            compat.V1,
            feature_online_store_service_client_v1.FeatureOnlineStoreServiceClient,
        ),
        (
            compat.V1BETA1,
            feature_online_store_service_client_v1beta1.FeatureOnlineStoreServiceClient,
        ),
    )


class FeaturestoreClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, featurestore_service_client_v1.FeaturestoreServiceClient),
        (compat.V1BETA1, featurestore_service_client_v1beta1.FeaturestoreServiceClient),
    )


class FeaturestoreOnlineServingClientWithOverride(ClientWithOverride):
    _is_temporary = False
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (
            compat.V1,
            featurestore_online_serving_service_client_v1.FeaturestoreOnlineServingServiceClient,
        ),
        (
            compat.V1BETA1,
            featurestore_online_serving_service_client_v1beta1.FeaturestoreOnlineServingServiceClient,
        ),
    )


class JobClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, job_service_client_v1.JobServiceClient),
        (compat.V1BETA1, job_service_client_v1beta1.JobServiceClient),
    )


class ModelClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, model_service_client_v1.ModelServiceClient),
        (compat.V1BETA1, model_service_client_v1beta1.ModelServiceClient),
    )


class PipelineClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, pipeline_service_client_v1.PipelineServiceClient),
        (compat.V1BETA1, pipeline_service_client_v1beta1.PipelineServiceClient),
    )


class PipelineJobClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, pipeline_service_client_v1.PipelineServiceClient),
        (compat.V1BETA1, pipeline_service_client_v1beta1.PipelineServiceClient),
    )


class ScheduleClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, schedule_service_client_v1.ScheduleServiceClient),
        (compat.V1BETA1, schedule_service_client_v1beta1.ScheduleServiceClient),
    )


class PredictionClientWithOverride(ClientWithOverride):
    _is_temporary = False
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, prediction_service_client_v1.PredictionServiceClient),
        (compat.V1BETA1, prediction_service_client_v1beta1.PredictionServiceClient),
    )


class PredictionAsyncClientWithOverride(ClientWithOverride):
    _is_temporary = False
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, prediction_service_async_client_v1.PredictionServiceAsyncClient),
        (
            compat.V1BETA1,
            prediction_service_async_client_v1beta1.PredictionServiceAsyncClient,
        ),
    )


class MatchClientWithOverride(ClientWithOverride):
    _is_temporary = False
    _default_version = compat.V1BETA1
    _version_map = ((compat.V1BETA1, match_service_client_v1beta1.MatchServiceClient),)


class MetadataClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, metadata_service_client_v1.MetadataServiceClient),
        (compat.V1BETA1, metadata_service_client_v1beta1.MetadataServiceClient),
    )


class TensorboardClientWithOverride(ClientWithOverride):
    _is_temporary = False
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, tensorboard_service_client_v1.TensorboardServiceClient),
        (compat.V1BETA1, tensorboard_service_client_v1beta1.TensorboardServiceClient),
    )


class VizierClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, vizier_service_client_v1.VizierServiceClient),
        (compat.V1BETA1, vizier_service_client_v1beta1.VizierServiceClient),
    )


class ModelGardenClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (compat.V1, model_garden_service_client_v1.ModelGardenServiceClient),
        (compat.V1BETA1, model_garden_service_client_v1beta1.ModelGardenServiceClient),
    )


class PersistentResourceClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.DEFAULT_VERSION
    _version_map = (
        (
            compat.V1,
            persistent_resource_service_client_v1.PersistentResourceServiceClient,
        ),
        (
            compat.V1BETA1,
            persistent_resource_service_client_v1beta1.PersistentResourceServiceClient,
        ),
    )


class ReasoningEngineClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.V1BETA1
    _version_map = (
        (
            compat.V1BETA1,
            reasoning_engine_service_client_v1beta1.ReasoningEngineServiceClient,
        ),
    )


class ReasoningEngineExecutionClientWithOverride(ClientWithOverride):
    _is_temporary = True
    _default_version = compat.V1BETA1
    _version_map = (
        (
            compat.V1BETA1,
            reasoning_engine_execution_service_client_v1beta1.ReasoningEngineExecutionServiceClient,
        ),
    )


VertexAiServiceClientWithOverride = TypeVar(
    "VertexAiServiceClientWithOverride",
    DatasetClientWithOverride,
    EndpointClientWithOverride,
    FeaturestoreClientWithOverride,
    JobClientWithOverride,
    ModelClientWithOverride,
    MatchClientWithOverride,
    PipelineClientWithOverride,
    PipelineJobClientWithOverride,
    PredictionClientWithOverride,
    MetadataClientWithOverride,
    ScheduleClientWithOverride,
    TensorboardClientWithOverride,
    VizierClientWithOverride,
    ModelGardenClientWithOverride,
    PersistentResourceClientWithOverride,
    ReasoningEngineClientWithOverride,
    ReasoningEngineExecutionClientWithOverride,
)


class LoggingFilter(logging.Filter):
    def __init__(self, warning_level: int):
        self._warning_level = warning_level

    def filter(self, record):
        return record.levelname == self._warning_level


def _timestamped_gcs_dir(root_gcs_path: str, dir_name_prefix: str) -> str:
    """Composes a timestamped GCS directory.

    Args:
        root_gcs_path: GCS path to put the timestamped directory.
        dir_name_prefix: Prefix to add the timestamped directory.
    Returns:
        Timestamped gcs directory path in root_gcs_path.
    """
    timestamp = datetime.datetime.now().isoformat(sep="-", timespec="milliseconds")
    dir_name = "-".join([dir_name_prefix, timestamp])
    if root_gcs_path.endswith("/"):
        root_gcs_path = root_gcs_path[:-1]
    gcs_path = "/".join([root_gcs_path, dir_name])
    if not gcs_path.startswith("gs://"):
        return "gs://" + gcs_path
    return gcs_path


def _timestamped_copy_to_gcs(
    local_file_path: str,
    gcs_dir: str,
    project: Optional[str] = None,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> str:
    """Copies a local file to a GCS path.

    The file copied to GCS is the name of the local file prepended with an
    "aiplatform-{timestamp}-" string.

    Args:
        local_file_path (str): Required. Local file to copy to GCS.
        gcs_dir (str):
            Required. The GCS directory to copy to.
        project (str):
            Project that contains the staging bucket. Default will be used if not
            provided. Model Builder callers should pass this in.
        credentials (auth_credentials.Credentials):
            Custom credentials to use with bucket. Model Builder callers should pass
            this in.
    Returns:
        gcs_path (str): The path of the copied file in gcs.
    """

    gcs_bucket, gcs_blob_prefix = extract_bucket_and_prefix_from_gcs_path(gcs_dir)

    local_file_name = pathlib.Path(local_file_path).name
    timestamp = datetime.datetime.now().isoformat(sep="-", timespec="milliseconds")
    blob_path = "-".join(["aiplatform", timestamp, local_file_name])

    if gcs_blob_prefix:
        blob_path = "/".join([gcs_blob_prefix, blob_path])

    # TODO(b/171202993) add user agent
    client = storage.Client(project=project, credentials=credentials)
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_file_path)

    gcs_path = "".join(["gs://", "/".join([blob.bucket.name, blob.name])])
    return gcs_path


def get_timestamp_proto(
    time: Optional[datetime.datetime] = None,
) -> timestamp_pb2.Timestamp:
    """Gets timestamp proto of a given time.
    Args:
        time (datetime.datetime):
            Optional. A user provided time. Default to datetime.datetime.now() if not given.
    Returns:
        timestamp_pb2.Timestamp: timestamp proto of the given time, not have higher than millisecond precision.
    """
    if not time:
        time = datetime.datetime.now()

    time_str = time.isoformat(sep=" ", timespec="milliseconds")
    time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")

    timestamp_proto = timestamp_pb2.Timestamp()
    timestamp_proto.FromDatetime(time)

    return timestamp_proto


def timestamped_unique_name() -> str:
    """Composes a timestamped unique name.

    Returns:
        A string representing a unique name.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    unique_id = uuid.uuid4().hex[0:5]
    return f"{timestamp}-{unique_id}"
