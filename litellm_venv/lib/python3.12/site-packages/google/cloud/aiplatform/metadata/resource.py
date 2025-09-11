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
import collections
import re
import threading
from copy import deepcopy
from typing import Dict, Optional, Union, Any, List

import proto
from google.api_core import exceptions
from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import base, initializer
from google.cloud.aiplatform import metadata
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import artifact as gca_artifact
from google.cloud.aiplatform.compat.types import context as gca_context
from google.cloud.aiplatform.compat.types import execution as gca_execution

_LOGGER = base.Logger(__name__)


class _Resource(base.VertexAiResourceNounWithFutureManager, abc.ABC):
    """Metadata Resource for Vertex AI"""

    client_class = utils.MetadataClientWithOverride
    _delete_method = None

    def __init__(
        self,
        resource_name: Optional[str] = None,
        resource: Optional[
            Union[gca_context.Context, gca_artifact.Artifact, gca_execution.Execution]
        ] = None,
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing Metadata resource given a resource name or ID.

        Args:
            resource_name (str):
                A fully-qualified resource name or ID
                Example: "projects/123/locations/us-central1/metadataStores/default/<resource_noun>/my-resource".
                or "my-resource" when project and location are initialized or passed. if ``resource`` is provided, this
                should not be set.
            resource (Union[gca_context.Context, gca_artifact.Artifact, gca_execution.Execution]):
                The proto.Message that contains the full information of the resource. If both set, this field overrides
                ``resource_name`` field.
            metadata_store_id (str):
                MetadataStore to retrieve resource from. If not set, metadata_store_id is set to "default".
                If resource_name is a fully-qualified resource, its metadata_store_id overrides this one.
            project (str):
                Optional project to retrieve the resource from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional location to retrieve the resource from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to retrieve this resource. Overrides
                credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
        )

        if resource:
            self._gca_resource = resource
        else:
            full_resource_name = utils.full_resource_name(
                resource_name=resource_name,
                resource_noun=self._resource_noun,
                parse_resource_name_method=self._parse_resource_name,
                format_resource_name_method=self._format_resource_name,
                parent_resource_name_fields={
                    metadata.metadata_store._MetadataStore._resource_noun: metadata_store_id
                },
                project=self.project,
                location=self.location,
            )

            self._gca_resource = getattr(self.api_client, self._getter_method)(
                name=full_resource_name, retry=base._DEFAULT_RETRY
            )

        self._threading_lock = threading.Lock()

    @property
    def metadata(self) -> Dict:
        return self.to_dict()["metadata"]

    @property
    def schema_title(self) -> str:
        return self._gca_resource.schema_title

    @property
    def description(self) -> str:
        return self._gca_resource.description

    @property
    def display_name(self) -> str:
        return self._gca_resource.display_name

    @property
    def schema_version(self) -> str:
        return self._gca_resource.schema_version

    @classmethod
    def get_or_create(
        cls,
        resource_id: str,
        schema_title: str,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "_Resource":
        """Retrieves or Creates (if it does not exist) a Metadata resource.

        Args:
            resource_id (str):
                Required. The <resource_id> portion of the resource name with the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/<resource_id>.
            schema_title (str):
                Required. schema_title identifies the schema title used by the resource.
            display_name (str):
                Optional. The user-defined name of the resource.
            schema_version (str):
                Optional. schema_version specifies the version used by the resource.
                If not set, defaults to use the latest version.
            description (str):
                Optional. Describes the purpose of the resource to be created.
            metadata (Dict):
                Optional. Contains the metadata information that will be stored in the resource.
            metadata_store_id (str):
                The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Project used to retrieve or create this resource. Overrides project set in
                aiplatform.init.
            location (str):
                Location used to retrieve or create this resource. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials used to retrieve or create this resource. Overrides
                credentials set in aiplatform.init.

        Returns:
            resource (_Resource):
                Instantiated representation of the managed Metadata resource.

        """

        resource = cls._get(
            resource_name=resource_id,
            metadata_store_id=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )
        if not resource:
            _LOGGER.info(f"Creating Resource {resource_id}")
            resource = cls._create(
                resource_id=resource_id,
                schema_title=schema_title,
                display_name=display_name,
                schema_version=schema_version,
                description=description,
                metadata=metadata,
                metadata_store_id=metadata_store_id,
                project=project,
                location=location,
                credentials=credentials,
            )
        return resource

    @classmethod
    def get(
        cls,
        resource_id: str,
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "_Resource":
        """Retrieves a Metadata resource.

        Args:
            resource_id (str):
                Required. The <resource_id> portion of the resource name with the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/<resource_id>.
            metadata_store_id (str):
                The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Project used to retrieve or create this resource. Overrides project set in
                aiplatform.init.
            location (str):
                Location used to retrieve or create this resource. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials used to retrieve or create this resource. Overrides
                credentials set in aiplatform.init.

        Returns:
            resource (_Resource):
                Instantiated representation of the managed Metadata resource or None if no resource was found.

        """
        resource = cls._get(
            resource_name=resource_id,
            metadata_store_id=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )
        return resource

    def sync_resource(self):
        """Syncs local resource with the resource in metadata store."""
        self._gca_resource = getattr(self.api_client, self._getter_method)(
            name=self.resource_name, retry=base._DEFAULT_RETRY
        )

    @staticmethod
    def _nested_update_metadata(
        gca_resource: Union[
            gca_context.Context, gca_execution.Execution, gca_artifact.Artifact
        ],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Helper method to update gca_resource in place.

        Performs a one-level deep nested update on the metadata field.

        Args:
            gca_resource (Union[gca_context.Context, gca_execution.Execution, gca_artifact.Artifact]):
                Required. Metadata Protobuf resource. This proto's metadata will be
                updated in place.
            metadata (Dict[str, Any]):
                Optional. Metadata dictionary to merge into gca_resource.metadata.
        """

        if metadata:
            if gca_resource.metadata:
                for key, value in metadata.items():
                    # Note: This only support nested dictionaries one level deep
                    if isinstance(value, collections.abc.Mapping):
                        gca_resource.metadata[key].update(value)
                    else:
                        gca_resource.metadata[key] = value
            else:
                gca_resource.metadata = metadata

    def update(
        self,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Updates an existing Metadata resource with new metadata.

        Args:
            metadata (Dict):
                Optional. metadata contains the updated metadata information.
            description (str):
                Optional. Description describes the resource to be updated.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to update this resource. Overrides
                credentials set in aiplatform.init.
        """
        if not hasattr(self, "_threading_lock"):
            self._threading_lock = threading.Lock()

        with self._threading_lock:
            gca_resource = deepcopy(self._gca_resource)
            if metadata:
                self._nested_update_metadata(
                    gca_resource=gca_resource, metadata=metadata
                )
            if description:
                gca_resource.description = description

            api_client = self._instantiate_client(credentials=credentials)
            # TODO: if etag is not valid sync and retry
            update_gca_resource = self._update_resource(
                client=api_client,
                resource=gca_resource,
            )
            self._gca_resource = update_gca_resource

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,  # pylint: disable=redefined-builtin
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        order_by: Optional[str] = None,
    ) -> List["_Resource"]:
        """List resources that match the list filter in target metadataStore.

        Args:
            filter (str):
                Optional. A query to filter available resources for
                matching results.
            metadata_store_id (str):
                The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Project used to create this resource. Overrides project set in
                aiplatform.init.
            location (str):
                Location used to create this resource. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials used to create this resource. Overrides
                credentials set in aiplatform.init.
            order_by (str):
              Optional. How the list of messages is ordered.
              Specify the values to order by and an ordering operation. The
              default sorting order is ascending. To specify descending order
              for a field, users append a " desc" suffix; for example: "foo
              desc, bar". Subfields are specified with a ``.`` character, such
              as foo.bar. see https://google.aip.dev/132#ordering for more
              details.

        Returns:
            resources (sequence[_Resource]):
                a list of managed Metadata resource.

        """
        parent = (
            initializer.global_config.common_location_path(
                project=project, location=location
            )
            + f"/metadataStores/{metadata_store_id}"
        )

        return super().list(
            filter=filter,
            project=project,
            location=location,
            credentials=credentials,
            parent=parent,
            order_by=order_by,
        )

    @classmethod
    def _create(
        cls,
        resource_id: str,
        schema_title: str,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        metadata_store_id: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> Optional["_Resource"]:
        """Creates a new Metadata resource.

        Args:
            resource_id (str):
                Required. The <resource_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/<resource_id>.
            schema_title (str):
                Required. schema_title identifies the schema title used by the resource.
            display_name (str):
                Optional. The user-defined name of the resource.
            schema_version (str):
                Optional. schema_version specifies the version used by the resource.
                If not set, defaults to use the latest version.
            description (str):
                Optional. Describes the purpose of the resource to be created.
            metadata (Dict):
                Optional. Contains the metadata information that will be stored in the resource.
            metadata_store_id (str):
                The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Project used to create this resource. Overrides project set in
                aiplatform.init.
            location (str):
                Location used to create this resource. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials used to create this resource. Overrides
                credentials set in aiplatform.init.

        Returns:
            resource (_Resource):
                Instantiated representation of the managed Metadata resource.

        """
        api_client = cls._instantiate_client(location=location, credentials=credentials)

        parent = (
            initializer.global_config.common_location_path(
                project=project, location=location
            )
            + f"/metadataStores/{metadata_store_id}"
        )

        try:
            resource = cls._create_resource(
                client=api_client,
                parent=parent,
                resource_id=resource_id,
                schema_title=schema_title,
                display_name=display_name,
                schema_version=schema_version,
                description=description,
                metadata=metadata,
            )
        except exceptions.AlreadyExists:
            _LOGGER.info(f"Resource '{resource_id}' already exist")
            return

        self = cls._empty_constructor(
            project=project,
            location=location,
            credentials=credentials,
        )

        self._gca_resource = resource
        self._threading_lock = threading.Lock()
        return self

    @classmethod
    def _get(
        cls,
        resource_name: str,
        metadata_store_id: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> Optional["_Resource"]:
        """Returns a metadata Resource.

        Args:
            resource_name (str):
                A fully-qualified resource name or resource ID
                Example: "projects/123/locations/us-central1/metadataStores/default/<resource_noun>/my-resource".
                or "my-resource" when project and location are initialized or passed.
            metadata_store_id (str):
                The metadata_store_id portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/my-resource
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Project to get this resource from. Overrides project set in
                aiplatform.init.
            location (str):
                Location to get this resource from. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to get this resource. Overrides
                credentials set in aiplatform.init.

        Returns:
            resource (Optional[_Resource]):
                An optional instantiated representation of the managed Metadata resource.

        """

        try:
            return cls(
                resource_name,
                metadata_store_id=metadata_store_id,
                project=project,
                location=location,
                credentials=credentials,
            )
        except exceptions.NotFound:
            _LOGGER.info(f"Resource {resource_name} not found.")

    @classmethod
    @abc.abstractmethod
    def _create_resource(
        cls,
        client: utils.MetadataClientWithOverride,
        parent: str,
        resource_id: str,
        schema_title: str,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> proto.Message:
        """Create resource method."""
        pass

    @classmethod
    @abc.abstractmethod
    def _update_resource(
        cls,
        client: utils.MetadataClientWithOverride,
        resource: proto.Message,
    ) -> proto.Message:
        """Update resource method."""
        pass

    @staticmethod
    def _extract_metadata_store_id(resource_name, resource_noun) -> str:
        """Extracts the metadata store id from the resource name.

        Args:
            resource_name (str):
                Required. A fully-qualified metadata resource name. For example
                projects/{project}/locations/{location}/metadataStores/{metadata_store_id}/{resource_noun}/{resource_id}.
            resource_noun (str):
                Required. The resource_noun portion of the resource_name
        Returns:
            metadata_store_id (str):
                The metadata store id for the particular resource name.
        Raises:
            ValueError: If it does not exist.
        """
        pattern = re.compile(
            r"^projects\/(?P<project>[\w-]+)\/locations\/(?P<location>[\w-]+)\/metadataStores\/(?P<store>[\w-]+)\/"
            + resource_noun
            + r"\/(?P<id>[\w-]+)(?P<version>@[\w-]+)?$"
        )
        match = pattern.match(resource_name)
        if not match:
            raise ValueError(
                f"failed to extract metadata_store_id from resource {resource_name}"
            )
        return match["store"]
