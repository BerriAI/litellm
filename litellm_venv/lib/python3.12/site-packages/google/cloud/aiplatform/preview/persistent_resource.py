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

from typing import Dict, List, Optional, Union

from google.api_core import operation
from google.api_core import retry
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.services import (
    persistent_resource_service_client_v1beta1 as persistent_resource_service_client_compat,
)
from google.cloud.aiplatform_v1beta1.types import (
    encryption_spec as gca_encryption_spec_compat,
)
from google.cloud.aiplatform_v1beta1.types import (
    persistent_resource as gca_persistent_resource_compat,
)

from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


_LOGGER = base.Logger(__name__)
_DEFAULT_RETRY = retry.Retry()


class PersistentResource(base.VertexAiResourceNounWithFutureManager):
    """Managed PersistentResource feature for Vertex AI (Preview)."""

    client_class = utils.PersistentResourceClientWithOverride
    _resource_noun = "persistentResource"
    _getter_method = "get_persistent_resource"
    _list_method = "list_persistent_resources"
    _delete_method = "delete_persistent_resource"
    _parse_resource_name_method = "parse_persistent_resource_path"
    _format_resource_name_method = "persistent_resource_path"

    def __init__(
        self,
        persistent_resource_id: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves the PersistentResource and instantiates its representation.

        Args:
            persistent_resource_id (str):
                Required.
            project (str):
                Project this PersistentResource is in. Overrides
                project set in aiplatform.init.
            location (str):
                Location this PersistentResource is in. Overrides
                location set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to manage this PersistentResource.
                Overrides credentials set in aiplatform.init.
        """
        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=persistent_resource_id,
        )

        self._gca_resource = self._get_gca_resource(
            resource_name=persistent_resource_id
        )

    @property
    def display_name(self) -> Optional[str]:
        """The display name of the PersistentResource."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "display_name", None)

    @property
    def state(self) -> gca_persistent_resource_compat.PersistentResource.State:
        """The state of the PersistentResource.

        Values:
            STATE_UNSPECIFIED (0):
                Not set.
            PROVISIONING (1):
                The PROVISIONING state indicates the
                persistent resources is being created.
            RUNNING (3):
                The RUNNING state indicates the persistent
                resources is healthy and fully usable.
            STOPPING (4):
                The STOPPING state indicates the persistent
                resources is being deleted.
            ERROR (5):
                The ERROR state indicates the persistent resources may be
                unusable. Details can be found in the ``error`` field.
        """
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "state", None)

    @property
    def error(self) -> Optional[status_pb2.Status]:
        """The error status of the PersistentResource.

        Only populated when the resource's state is ``STOPPING`` or ``ERROR``.

        """
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "error", None)

    @property
    def create_time(self) -> Optional[timestamp_pb2.Timestamp]:
        """Time when the PersistentResource was created."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "create_time", None)

    @property
    def start_time(self) -> Optional[timestamp_pb2.Timestamp]:
        """Time when the PersistentResource first entered the ``RUNNING`` state."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "start_time", None)

    @property
    def update_time(self) -> Optional[timestamp_pb2.Timestamp]:
        """Time when the PersistentResource was most recently updated."""
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "update_time", None)

    @property
    def network(self) -> Optional[str]:
        """The network peered with the PersistentResource.

        The full name of the Compute Engine
        `network </compute/docs/networks-and-firewalls#networks>`__ to peered
        with Vertex AI to host the persistent resources.

        For example, ``projects/12345/global/networks/myVPC``.
        `Format </compute/docs/reference/rest/v1/networks/insert>`__ is of the
        form ``projects/{project}/global/networks/{network}``. Where {project}
        is a project number, as in ``12345``, and {network} is a network name.

        To specify this field, you must have already `configured VPC Network
        Peering for Vertex
        AI <https://cloud.google.com/vertex-ai/docs/general/vpc-peering>`__.

        If this field is left unspecified, the resources aren't peered with any
        network.
        """
        self._assert_gca_resource_is_available()
        return getattr(self._gca_resource, "network", None)

    @classmethod
    @base.optional_sync()
    def create(
        cls,
        persistent_resource_id: str,
        resource_pools: Union[
            List[Dict], List[gca_persistent_resource_compat.ResourcePool]
        ],
        display_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        network: Optional[str] = None,
        kms_key_name: Optional[str] = None,
        service_account: Optional[str] = None,
        reserved_ip_ranges: List[str] = None,
        sync: Optional[bool] = True,  # pylint: disable=unused-argument
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "PersistentResource":
        r"""Creates a PersistentResource.

        Args:
            persistent_resource_id (str):
                Required. The ID to use for the PersistentResource,
                which become the final component of the
                PersistentResource's resource name.

                The maximum length is 63 characters, and valid
                characters are ``/^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/``.

                This corresponds to the ``persistent_resource_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            resource_pools (MutableSequence[google.cloud.aiplatform_v1.types.ResourcePool]):
                Required. The list of resource pools to create for the
                PersistentResource.
            display_name (str):
                Optional. The display name of the
                PersistentResource. The name can be up to 128
                characters long and can consist of any UTF-8
                characters.
            labels (MutableMapping[str, str]):
                Optional. The labels with user-defined
                metadata to organize PersistentResource.

                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.

                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            network (str):
                Optional. The full name of the Compute Engine
                `network </compute/docs/networks-and-firewalls#networks>`__
                to peered with Vertex AI to host the persistent resources.
                For example, ``projects/12345/global/networks/myVPC``.
                `Format </compute/docs/reference/rest/v1/networks/insert>`__
                is of the form
                ``projects/{project}/global/networks/{network}``. Where
                {project} is a project number, as in ``12345``, and
                {network} is a network name.

                To specify this field, you must have already `configured VPC
                Network Peering for Vertex
                AI <https://cloud.google.com/vertex-ai/docs/general/vpc-peering>`__.

                If this field is left unspecified, the resources aren't
                peered with any network.
            kms_key_name (str):
                Optional. Customer-managed encryption key for the
                PersistentResource. If set, this PersistentResource and all
                sub-resources of this PersistentResource will be secured by
                this key.
            service_account (str):
                Optional. Default service account that this
                PersistentResource's workloads run as. The workloads
                including

                -  Any runtime specified via ``ResourceRuntimeSpec`` on
                   creation time, for example, Ray.
                -  Jobs submitted to PersistentResource, if no other service
                   account specified in the job specs.

                Only works when custom service account is enabled and users
                have the ``iam.serviceAccounts.actAs`` permission on this
                service account.
            reserved_ip_ranges (MutableSequence[str]):
                Optional. A list of names for the reserved IP ranges under
                the VPC network that can be used for this persistent
                resource.

                If set, we will deploy the persistent resource within the
                provided IP ranges. Otherwise, the persistent resource is
                deployed to any IP ranges under the provided VPC network.

                Example ['vertex-ai-ip-range'].
            sync (bool):
                Whether to execute this method synchonously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed.
            project (str):
                Project to create this PersistentResource in. Overrides project
                set in aiplatform.init.
            location (str):
                Location to create this PersistentResource in. Overrides
                location set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to create this PersistentResource.
                Overrides credentials set in aiplatform.init.

        Returns:
            persistent_resource (PersistentResource):
                The object representation of the newly created
                PersistentResource.
        """

        if labels:
            utils.validate_labels(labels)

        gca_persistent_resource = gca_persistent_resource_compat.PersistentResource(
            name=persistent_resource_id,
            display_name=display_name,
            resource_pools=resource_pools,
            labels=labels,
            network=network,
            reserved_ip_ranges=reserved_ip_ranges,
        )

        if kms_key_name:
            gca_persistent_resource.encryption_spec = (
                gca_encryption_spec_compat.EncryptionSpec(kms_key_name=kms_key_name)
            )

        if service_account:
            service_account_spec = gca_persistent_resource_compat.ServiceAccountSpec(
                enable_custom_service_account=True, service_account=service_account
            )
            gca_persistent_resource.resource_runtime_spec = (
                gca_persistent_resource_compat.ResourceRuntimeSpec(
                    service_account_spec=service_account_spec
                )
            )

        api_client = cls._instantiate_client(location, credentials).select_version(
            "v1beta1"
        )
        create_lro = cls._create(
            api_client=api_client,
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            persistent_resource=gca_persistent_resource,
            persistent_resource_id=persistent_resource_id,
        )

        _LOGGER.log_create_with_lro(cls, create_lro)

        create_lro.result(timeout=None)
        persistent_resource_result = cls(
            persistent_resource_id=persistent_resource_id,
            project=project,
            location=location,
            credentials=credentials,
        )

        _LOGGER.log_create_complete(
            cls, persistent_resource_result._gca_resource, "persistent resource"
        )

        return persistent_resource_result

    @classmethod
    def _create(
        cls,
        api_client: (
            persistent_resource_service_client_compat.PersistentResourceServiceClient
        ),
        parent: str,
        persistent_resource: gca_persistent_resource_compat.PersistentResource,
        persistent_resource_id: str,
        create_request_timeout: Optional[float] = None,
    ) -> operation.Operation:
        """Creates a PersistentResource directly calling the API client.

        Args:
            api_client (PersistentResourceServiceClient):
                An instance of PersistentResourceServiceClient with the correct
                api_endpoint already set based on user's preferences.
            parent (str):
                Required. Also known as common location path, that usually contains the
                project and location that the user provided to the upstream method.
                IE "projects/my-project/locations/us-central1"
            persistent_resource (gca_persistent_resource_compat.PersistentResource):
                Required. The PersistentResource object to use for the create request.
            persistent_resource_id (str):
                Required. The ID to use for the PersistentResource,
                which become the final component of the
                PersistentResource's resource name.

                The maximum length is 63 characters, and valid
                characters are ``/^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/``.

                This corresponds to the ``persistent_resource_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.

        Returns:
            operation (Operation):
                The long-running operation returned by the Persistent Resource
                create call.
        """
        return api_client.create_persistent_resource(
            parent=parent,
            persistent_resource_id=persistent_resource_id,
            persistent_resource=persistent_resource,
            timeout=create_request_timeout,
        )

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["PersistentResource"]:
        """Lists a Persistent Resources on the provided project and region.

        Args:
            filter (str):
                Optional. An expression for filtering the results of the request.
                For field names both snake_case and camelCase are supported.
            order_by (str):
                Optional. A comma-separated list of fields to order by, sorted in
                ascending order. Use "desc" after a field name for descending.
                Supported fields: `display_name`, `create_time`, `update_time`
            project (str):
                Optional. Project to retrieve list from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve list from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve list. Overrides
                credentials set in aiplatform.init.

        Returns:
            List[PersistentResource]
                A list of PersistentResource objects.
        """
        return cls._list_with_local_order(
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
        )
