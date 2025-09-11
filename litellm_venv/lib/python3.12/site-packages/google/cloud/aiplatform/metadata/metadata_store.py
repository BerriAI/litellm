# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
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

import logging
from typing import Optional

from google.api_core import exceptions
from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import base, initializer
from google.cloud.aiplatform import compat
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import metadata_store as gca_metadata_store
from google.cloud.aiplatform.constants import base as base_constants


class _MetadataStore(base.VertexAiResourceNounWithFutureManager):
    """Managed MetadataStore resource for Vertex AI"""

    client_class = utils.MetadataClientWithOverride
    _is_client_prediction_client = False
    _resource_noun = "metadataStores"
    _getter_method = "get_metadata_store"
    _delete_method = "delete_metadata_store"
    _parse_resource_name_method = "parse_metadata_store_path"
    _format_resource_name_method = "metadata_store_path"

    def __init__(
        self,
        metadata_store_name: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing MetadataStore given a MetadataStore name or ID.

        Args:
            metadata_store_name (str):
                Optional. A fully-qualified MetadataStore resource name or metadataStore ID.
                Example: "projects/123/locations/us-central1/metadataStores/my-store" or
                "my-store" when project and location are initialized or passed.
                If not set, metadata_store_name will be set to "default".
            project (str):
                Optional project to retrieve resource from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional location to retrieve resource from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.

        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
        )
        self._gca_resource = self._get_gca_resource(resource_name=metadata_store_name)

    @classmethod
    def get_or_create(
        cls,
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
    ) -> "_MetadataStore":
        """ "Retrieves or Creates (if it does not exist) a Metadata Store.

        Args:
            metadata_store_id (str):
                The <metadatastore> portion of the resource name with the format:
                projects/123/locations/us-central1/metadataStores/<metadatastore>
                If not provided, the MetadataStore's ID will be set to "default" to create a default MetadataStore.
            project (str):
                Project used to retrieve or create the metadata store. Overrides project set in
                aiplatform.init.
            location (str):
                Location used to retrieve or create the metadata store. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials used to retrieve or create the metadata store. Overrides
                credentials set in aiplatform.init.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the metadata store. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this MetadataStore and all sub-resources of this MetadataStore will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.


        Returns:
            metadata_store (_MetadataStore):
                Instantiated representation of the managed metadata store resource.

        """
        store = cls._get(
            metadata_store_name=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )
        if not store:
            store = cls._create(
                metadata_store_id=metadata_store_id,
                project=project,
                location=location,
                credentials=credentials,
                encryption_spec_key_name=encryption_spec_key_name,
            )
        return store

    @classmethod
    def _create(
        cls,
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
    ) -> "_MetadataStore":
        """Creates a new MetadataStore if it does not exist.

        Args:
            metadata_store_id (str):
                The <metadatastore> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadatastore>
                If not provided, the MetadataStore's ID will be set to "default" to create a default MetadataStore.
            project (str):
                Project used to create the metadata store. Overrides project set in
                aiplatform.init.
            location (str):
                Location used to create the metadata store. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials used to create the metadata store. Overrides
                credentials set in aiplatform.init.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the metadata store. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this MetadataStore and all sub-resources of this MetadataStore will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.


        Returns:
            metadata_store (_MetadataStore):
                Instantiated representation of the managed metadata store resource.

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

        gapic_metadata_store = gca_metadata_store.MetadataStore(
            encryption_spec=initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name,
                select_version=compat.DEFAULT_VERSION,
            )
        )

        try:
            api_client.create_metadata_store(
                parent=initializer.global_config.common_location_path(
                    project=project, location=location
                ),
                metadata_store=gapic_metadata_store,
                metadata_store_id=metadata_store_id,
            ).result()
        except exceptions.AlreadyExists:
            logging.info(f"MetadataStore '{metadata_store_id}' already exists")

        return cls(
            metadata_store_name=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )

    @classmethod
    def _get(
        cls,
        metadata_store_name: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> Optional["_MetadataStore"]:
        """Returns a MetadataStore resource.

        Args:
            metadata_store_name (str):
                Optional. A fully-qualified MetadataStore resource name or metadataStore ID.
                Example: "projects/123/locations/us-central1/metadataStores/my-store" or
                "my-store" when project and location are initialized or passed.
                If not set, metadata_store_name will be set to "default".
            project (str):
                Optional project to retrieve the metadata store from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional location to retrieve the metadata store from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Custom credentials to retrieve this metadata store. Overrides
                credentials set in aiplatform.init.

        Returns:
            metadata_store (Optional[_MetadataStore]):
                An optional instantiated representation of the managed Metadata Store resource.
        """

        try:
            return cls(
                metadata_store_name=metadata_store_name,
                project=project,
                location=location,
                credentials=credentials,
            )
        except exceptions.NotFound:
            logging.info(f"MetadataStore {metadata_store_name} not found.")

    @classmethod
    def ensure_default_metadata_store_exists(
        cls,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_key_spec_name: Optional[str] = None,
    ):
        """Helpers method to ensure the `default` MetadataStore exists in this project and location.

        Args:
            project (str):
                Optional. Project to retrieve resource from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve resource from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the metadata store. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this MetadataStore and all sub-resources of this MetadataStore will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
        """

        cls.get_or_create(
            project=project,
            location=location,
            credentials=credentials,
            encryption_spec_key_name=encryption_key_spec_name,
        )
