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

from typing import Optional, Dict, List, Sequence

import proto
import re
import threading

from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.constants import base as base_constants
from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.compat.types import context as gca_context
from google.cloud.aiplatform.compat.types import (
    lineage_subgraph as gca_lineage_subgraph,
)
from google.cloud.aiplatform.compat.types import (
    metadata_service as gca_metadata_service,
)
from google.cloud.aiplatform.metadata import artifact
from google.cloud.aiplatform.metadata import execution
from google.cloud.aiplatform.metadata import metadata_store
from google.cloud.aiplatform.metadata import resource
from google.api_core.exceptions import Aborted

_ETAG_ERROR_MAX_RETRY_COUNT = 5
_ETAG_ERROR_REGEX = re.compile(
    r"Specified Context \`etag\`: \`(\d+)\` does not match server \`etag\`: \`(\d+)\`"
)


class Context(resource._Resource):
    """Metadata Context resource for Vertex AI"""

    _resource_noun = "contexts"
    _getter_method = "get_context"
    _delete_method = "delete_context"
    _parse_resource_name_method = "parse_context_path"
    _format_resource_name_method = "context_path"
    _list_method = "list_contexts"

    @property
    def parent_contexts(self) -> Sequence[str]:
        """The parent context resource names of this context."""
        return self.gca_resource.parent_contexts

    def add_artifacts_and_executions(
        self,
        artifact_resource_names: Optional[Sequence[str]] = None,
        execution_resource_names: Optional[Sequence[str]] = None,
    ):
        """Associate Executions and attribute Artifacts to a given Context.

        Args:
            artifact_resource_names (Sequence[str]):
                Optional. The full resource name of Artifacts to attribute to the Context.
            execution_resource_names (Sequence[str]):
                Optional. The full resource name of Executions to associate with the Context.
        """
        self.api_client.add_context_artifacts_and_executions(
            context=self.resource_name,
            artifacts=artifact_resource_names,
            executions=execution_resource_names,
        )

    def get_artifacts(self) -> List[artifact.Artifact]:
        """Returns all Artifact attributed to this Context.

        Returns:
            artifacts(List[Artifacts]): All Artifacts under this context.
        """
        return artifact.Artifact.list(
            filter=metadata_utils._make_filter_string(in_context=[self.resource_name]),
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )

    @classmethod
    def create(
        cls,
        schema_title: str,
        *,
        resource_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        metadata_store_id: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "Context":
        """Creates a new Metadata Context.

        Args:
            schema_title (str):
                Required. schema_title identifies the schema title used by the Context.
                Please reference https://cloud.google.com/vertex-ai/docs/ml-metadata/system-schemas.
            resource_id (str):
                Optional. The <resource_id> portion of the Context name with
                the format. This is globally unique in a metadataStore:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/Contexts/<resource_id>.
            display_name (str):
                Optional. The user-defined name of the Context.
            schema_version (str):
                Optional. schema_version specifies the version used by the Context.
                If not set, defaults to use the latest version.
            description (str):
                Optional. Describes the purpose of the Context to be created.
            metadata (Dict):
                Optional. Contains the metadata information that will be stored in the Context.
            metadata_store_id (str):
                Optional. The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/Contexts/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Optional. Project used to create this Context. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location used to create this Context. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to create this Context. Overrides
                credentials set in aiplatform.init.

        Returns:
            Context: Instantiated representation of the managed Metadata Context.
        """
        # Add User Agent Header for metrics tracking if one is not specified
        # If one is already specified this call was initiated by a sub class.
        if not base_constants.USER_AGENT_SDK_COMMAND:
            base_constants.USER_AGENT_SDK_COMMAND = (
                "aiplatform.metadata.context.Context.create"
            )

        return cls._create(
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

    # TODO() refactor code to move _create to _Resource class.
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
    ) -> "Context":
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
        appended_user_agent = []
        if base_constants.USER_AGENT_SDK_COMMAND:
            appended_user_agent = [
                f"sdk_command/{base_constants.USER_AGENT_SDK_COMMAND}"
            ]
            # Reset the value for the USER_AGENT_SDK_COMMAND to avoid counting future unrelated api calls.
            base_constants.USER_AGENT_SDK_COMMAND = ""

        api_client = cls._instantiate_client(
            location=location,
            credentials=credentials,
            appended_user_agent=appended_user_agent,
        )

        parent = utils.full_resource_name(
            resource_name=metadata_store_id,
            resource_noun=metadata_store._MetadataStore._resource_noun,
            parse_resource_name_method=metadata_store._MetadataStore._parse_resource_name,
            format_resource_name_method=metadata_store._MetadataStore._format_resource_name,
            project=project,
            location=location,
        )

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

        self = cls._empty_constructor(
            project=project, location=location, credentials=credentials
        )
        self._gca_resource = resource
        self._threading_lock = threading.Lock()

        return self

    @classmethod
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
        gapic_context = gca_context.Context(
            schema_title=schema_title,
            schema_version=schema_version,
            display_name=display_name,
            description=description,
            metadata=metadata if metadata else {},
        )
        return client.create_context(
            parent=parent,
            context=gapic_context,
            context_id=resource_id,
        )

    def update(
        self,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Updates an existing Metadata Context with new metadata.

        This is implemented with retry on etag errors, up to
        _ETAG_ERROR_MAX_RETRY_COUNT times.
        Args:
            metadata (Dict):
                Optional. metadata contains the updated metadata information.
            description (str):
                Optional. Description describes the resource to be updated.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to update this resource. Overrides
                credentials set in aiplatform.init.
        """
        for _ in range(_ETAG_ERROR_MAX_RETRY_COUNT - 1):
            try:
                super().update(
                    metadata=metadata, description=description, credentials=credentials
                )
                return
            except Aborted as aborted_exception:
                regex_match = _ETAG_ERROR_REGEX.match(aborted_exception.message)
                if regex_match:
                    local_etag = regex_match.group(1)
                    server_etag = regex_match.group(2)
                    if local_etag < server_etag:
                        self.sync_resource()
                        continue
                raise aborted_exception

        # Expose result/exception directly in the last retry.
        super().update(
            metadata=metadata, description=description, credentials=credentials
        )

    @classmethod
    def _update_resource(
        cls,
        client: utils.MetadataClientWithOverride,
        resource: proto.Message,
    ) -> proto.Message:
        """Update Contexts with given input.

        Args:
            client (utils.MetadataClientWithOverride):
                Required. client to send require to Metadata Service.
            resource (proto.Message):
                Required. The proto.Message which contains the update information for the resource.
        """

        return client.update_context(context=resource)

    @classmethod
    def _list_resources(
        cls,
        client: utils.MetadataClientWithOverride,
        parent: str,
        filter: Optional[str] = None,  # pylint: disable=redefined-builtin
        order_by: Optional[str] = None,
    ):
        """List Contexts in the parent path that matches the filter.

        Args:
            client (utils.MetadataClientWithOverride):
                Required. client to send require to Metadata Service.
            parent (str):
                Required. The path where Contexts are stored.
            filter (str):
                Optional. filter string to restrict the list result
            order_by (str):
              Optional. How the list of messages is ordered. Specify the
              values to order by and an ordering operation. The default sorting
              order is ascending. To specify descending order for a field, users
              append a " desc" suffix; for example: "foo desc, bar". Subfields
              are specified with a ``.`` character, such as foo.bar. see
              https://google.aip.dev/132#ordering for more details.

        Returns:
            List of Contexts.
        """

        list_request = gca_metadata_service.ListContextsRequest(
            parent=parent,
            filter=filter,
            order_by=order_by,
        )
        return client.list_contexts(request=list_request)

    def add_context_children(self, contexts: List["Context"]):
        """Adds the provided contexts as children of this context.

        Args:
            contexts (List[_Context]): Contexts to add as children.
        """
        self.api_client.add_context_children(
            context=self.resource_name,
            child_contexts=[c.resource_name for c in contexts],
        )

    def query_lineage_subgraph(self) -> gca_lineage_subgraph.LineageSubgraph:
        """Queries lineage subgraph of this context.

        Returns:
            lineage subgraph(gca_lineage_subgraph.LineageSubgraph): Lineage subgraph of this Context.
        """

        return self.api_client.query_context_lineage_subgraph(
            context=self.resource_name, retry=base._DEFAULT_RETRY
        )

    def get_executions(self) -> List[execution.Execution]:
        """Returns Executions associated to this context.

        Returns:
            executions (List[Executions]): Executions associated to this context.
        """
        return execution.Execution.list(
            filter=metadata_utils._make_filter_string(in_context=[self.resource_name]),
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )
