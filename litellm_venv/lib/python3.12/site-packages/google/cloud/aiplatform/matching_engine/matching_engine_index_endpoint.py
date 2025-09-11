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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import matching_engine
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    machine_resources as gca_machine_resources_compat,
    matching_engine_index_endpoint as gca_matching_engine_index_endpoint,
    match_service_v1beta1 as gca_match_service_v1beta1,
    index_v1beta1 as gca_index_v1beta1,
    service_networking as gca_service_networking,
    encryption_spec as gca_encryption_spec,
)
from google.cloud.aiplatform.matching_engine._protos import match_service_pb2
from google.cloud.aiplatform.matching_engine._protos import (
    match_service_pb2_grpc,
)
from google.protobuf import field_mask_pb2

import grpc

_LOGGER = base.Logger(__name__)


@dataclass
class Namespace:
    """Namespace specifies the rules for determining the datapoints that are eligible for each matching query, overall query is an AND across namespaces.
    Args:
        name (str):
            Required. The name of this Namespace.
        allow_tokens (List(str)):
            Optional. The allowed tokens in the namespace.
        deny_tokens (List(str)):
            Optional. The denied tokens in the namespace. When a token is denied, then matches will be excluded whenever the other datapoint has that token.
            For example, if a query specifies [Namespace("color", ["red","blue"], ["purple"])], then that query will match datapoints that are red or blue,
            but if those points are also purple, then they will be excluded even if they are red/blue.
    """

    name: str
    allow_tokens: list = field(default_factory=list)
    deny_tokens: list = field(default_factory=list)


@dataclass
class NumericNamespace:
    """NumericNamespace specifies the rules for determining the datapoints that
    are eligible for each matching query, overall query is an AND across namespaces.
    This uses numeric comparisons.

    Args:
        name (str):
            Required. The name of this numeric namespace.
        value_int (int):
            Optional. 64 bit integer value for comparison. Must choose one among
            `value_int`, `value_float` and `value_double` for intended
            precision.
        value_float (float):
            Optional. 32 bit float value for comparison. Must choose one among
            `value_int`, `value_float` and `value_double` for
            intended precision.
        value_double (float):
            Optional. 64b bit float value for comparison. Must choose one among
            `value_int`, `value_float` and `value_double` for
            intended precision.
        operator (str):
            Optional. Should be specified for query only, not for a datapoints.
            Specify one operator to use for comparison. Datapoints for which
            comparisons with query's values are true for the operator and value
            combination will be allowlisted. Choose among:
                "LESS" for datapoints' values < query's value;
                "LESS_EQUAL" for datapoints' values <= query's value;
                "EQUAL" for datapoints' values = query's value;
                "GREATER_EQUAL" for datapoints' values >= query's value;
                "GREATER" for datapoints' values > query's value;
    """

    name: str
    value_int: Optional[int] = None
    value_float: Optional[float] = None
    value_double: Optional[float] = None
    op: Optional[str] = None

    def __post_init__(self):
        """Check NumericNamespace values are of correct types and values are
        not all none.
        Args:
            None.

        Raises:
            ValueError: Numeric Namespace provided values must be of correct
            types and one of value_int, value_float, value_double must exist.
        """
        # Check one of
        if (
            self.value_int is None
            and self.value_float is None
            and self.value_double is None
        ):
            raise ValueError(
                "Must choose one among `value_int`,"
                "`value_float` and `value_double` for "
                "intended precision."
            )

        # Check value type
        if self.value_int is not None and not isinstance(self.value_int, int):
            raise ValueError(
                "value_int must be of type int, got" f" { type(self.value_int)}."
            )
        if self.value_float is not None and not isinstance(self.value_float, float):
            raise ValueError(
                "value_float must be of type float, got " f"{ type(self.value_float)}."
            )
        if self.value_double is not None and not isinstance(self.value_double, float):
            raise ValueError(
                "value_double must be of type float, got "
                f"{ type(self.value_double)}."
            )
        # Check operator validity
        if (
            self.op is not None
            and self.op
            not in gca_index_v1beta1.IndexDatapoint.NumericRestriction.Operator._member_names_
        ):
            raise ValueError(
                f"Invalid operator '{self.op}'," " must be one of the valid operators."
            )


@dataclass
class MatchNeighbor:
    """The id and distance of a nearest neighbor match for a given query embedding.

    Args:
        id (str):
            Required. The id of the neighbor.
        distance (float):
            Required. The distance to the query embedding.
        feature_vector (List(float)):
            Optional. The feature vector of the matching datapoint.
        crowding_tag (Optional[str]):
            Optional. Crowding tag of the datapoint, the
            number of neighbors to return in each crowding,
            can be configured during query.
        restricts (List[Namespace]):
            Optional. The restricts of the matching datapoint.
        numeric_restricts:
            Optional. The numeric restricts of the matching datapoint.

    """

    id: str
    distance: float
    feature_vector: Optional[List[float]] = None
    crowding_tag: Optional[str] = None
    restricts: Optional[List[Namespace]] = None
    numeric_restricts: Optional[List[NumericNamespace]] = None

    def from_index_datapoint(
        self, index_datapoint: gca_index_v1beta1.IndexDatapoint
    ) -> "MatchNeighbor":
        """Copies MatchNeighbor fields from an IndexDatapoint.

        Args:
            index_datapoint (gca_index_v1beta1.IndexDatapoint):
                Required. An index datapoint.

        Returns:
            MatchNeighbor
        """
        if not index_datapoint:
            return self
        self.feature_vector = index_datapoint.feature_vector
        if (
            index_datapoint.crowding_tag is not None
            and index_datapoint.crowding_tag.crowding_attribute is not None
        ):
            self.crowding_tag = index_datapoint.crowding_tag.crowding_attribute
        self.restricts = [
            Namespace(
                name=restrict.namespace,
                allow_tokens=restrict.allow_list,
                deny_tokens=restrict.deny_list,
            )
            for restrict in index_datapoint.restricts
        ]
        if index_datapoint.numeric_restricts is not None:
            self.numeric_restricts = []
        for restrict in index_datapoint.numeric_restricts:
            numeric_namespace = None
            restrict_value_type = restrict._pb.WhichOneof("Value")
            if restrict_value_type == "value_int":
                numeric_namespace = NumericNamespace(
                    name=restrict.namespace, value_int=restrict.value_int
                )
            elif restrict_value_type == "value_float":
                numeric_namespace = NumericNamespace(
                    name=restrict.namespace, value_float=restrict.value_float
                )
            elif restrict_value_type == "value_double":
                numeric_namespace = NumericNamespace(
                    name=restrict.namespace, value_double=restrict.value_double
                )
            self.numeric_restricts.append(numeric_namespace)
        return self

    def from_embedding(self, embedding: match_service_pb2.Embedding) -> "MatchNeighbor":
        """Copies MatchNeighbor fields from an Embedding.

        Args:
            embedding (gca_index_v1beta1.Embedding):
                Required. An embedding.

        Returns:
            MatchNeighbor
        """
        if not embedding:
            return self
        self.feature_vector = embedding.float_val
        if not self.crowding_tag and embedding.crowding_attribute is not None:
            self.crowding_tag = str(embedding.crowding_attribute)
        self.restricts = [
            Namespace(
                name=restrict.name,
                allow_tokens=restrict.allow_tokens,
                deny_tokens=restrict.deny_tokens,
            )
            for restrict in embedding.restricts
        ]
        if embedding.numeric_restricts:
            self.numeric_restricts = []
        for restrict in embedding.numeric_restricts:
            numeric_namespace = None
            restrict_value_type = restrict.WhichOneof("Value")
            if restrict_value_type == "value_int":
                numeric_namespace = NumericNamespace(
                    name=restrict.name, value_int=restrict.value_int
                )
            elif restrict_value_type == "value_float":
                numeric_namespace = NumericNamespace(
                    name=restrict.name, value_float=restrict.value_float
                )
            elif restrict_value_type == "value_double":
                numeric_namespace = NumericNamespace(
                    name=restrict.name, value_double=restrict.value_double
                )
            self.numeric_restricts.append(numeric_namespace)
        return self


class MatchingEngineIndexEndpoint(base.VertexAiResourceNounWithFutureManager):
    """Matching Engine index endpoint resource for Vertex AI."""

    client_class = utils.IndexEndpointClientWithOverride

    _resource_noun = "indexEndpoints"
    _getter_method = "get_index_endpoint"
    _list_method = "list_index_endpoints"
    _delete_method = "delete_index_endpoint"
    _parse_resource_name_method = "parse_index_endpoint_path"
    _format_resource_name_method = "index_endpoint_path"

    def __init__(
        self,
        index_endpoint_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing index endpoint given a name or ID.

        Example Usage:

            my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name='projects/123/locations/us-central1/index_endpoint/my_index_id'
            )
            or
            my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name='my_index_endpoint_id'
            )

        Args:
            index_endpoint_name (str):
                Required. A fully-qualified index endpoint resource name or a index ID.
                Example: "projects/123/locations/us-central1/index_endpoints/my_index_id"
                or "my_index_id" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve index endpoint from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve index endpoint from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this IndexEndpoint. Overrides
                credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=index_endpoint_name,
        )
        self._gca_resource = self._get_gca_resource(resource_name=index_endpoint_name)

        self._public_match_client = None
        if self.public_endpoint_domain_name:
            self._public_match_client = self._instantiate_public_match_client()

        self._match_grpc_stub_cache = {}
        self._private_service_connect_ip_address = None

    @classmethod
    def create(
        cls,
        display_name: str,
        network: Optional[str] = None,
        public_endpoint_enabled: bool = False,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync: bool = True,
        enable_private_service_connect: bool = False,
        project_allowlist: Optional[Sequence[str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
    ) -> "MatchingEngineIndexEndpoint":
        """Creates a MatchingEngineIndexEndpoint resource.

        Example Usage:

            my_index_endpoint = aiplatform.IndexEndpoint.create(
                display_name='my_endpoint',
            )

        Args:
            display_name (str):
                Required. The display name of the IndexEndpoint.
                The name can be up to 128 characters long and
                can be consist of any UTF-8 characters.
            network (str):
                Optional. The full name of the Google Compute Engine
                `network <https://cloud.google.com/compute/docs/networks-and-firewalls#networks>`__
                to which the IndexEndpoint should be peered.

                Private services access must already be configured for the network.
                If left unspecified, the network set with aiplatform.init will be used.

                `Format <https://cloud.google.com/compute/docs/reference/rest/v1/networks/insert>`__:
                projects/{project}/global/networks/{network}. Where
                {project} is a project number, as in '12345', and {network}
                is network name.
            public_endpoint_enabled (bool):
                Optional. If true, the deployed index will be
                accessible through public endpoint.
            description (str):
                Optional. The description of the IndexEndpoint.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your IndexEndpoint.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one
                IndexEndpoint (System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            project (str):
                Optional. Project to create IndexEndpoint in. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to create IndexEndpoint in. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create IndexEndpoints. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            sync (bool):
                Optional. Whether to execute this creation synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            enable_private_service_connect (bool):
                If true, expose the index endpoint via private service connect.
            project_allowlist (Sequence[str]):
                Optional. List of projects from which the forwarding rule will
                target the service attachment.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the index endpoint.
                Has the form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this index endpoint and all sub-resources of this
                index endpoint will be secured by this key.
                The key needs to be in the same region as where the index
                endpoint is created.
            create_request_timeout (float):
                Optional. The timeout for the request in seconds.

        Returns:
            MatchingEngineIndexEndpoint - IndexEndpoint resource object

        Raises:
            ValueError: A network must be instantiated when creating a IndexEndpoint.
        """
        network = network or initializer.global_config.network

        if not (network or public_endpoint_enabled or enable_private_service_connect):
            raise ValueError(
                "Please provide `network` argument for Private Service Access endpoint,"
                "or provide `enable_private_service_connect` for Private Service"
                "Connect endpoint, or provide `public_endpoint_enabled` to"
                "deploy to a public endpoint"
            )

        if (
            sum(
                bool(network_setting)
                for network_setting in [
                    network,
                    public_endpoint_enabled,
                    enable_private_service_connect,
                ]
            )
            > 1
        ):
            raise ValueError(
                "One and only one among network, public_endpoint_enabled and enable_private_service_connect should be set."
            )

        return cls._create(
            display_name=display_name,
            network=network,
            public_endpoint_enabled=public_endpoint_enabled,
            description=description,
            labels=labels,
            project=project,
            location=location,
            credentials=credentials,
            request_metadata=request_metadata,
            sync=sync,
            enable_private_service_connect=enable_private_service_connect,
            project_allowlist=project_allowlist,
            encryption_spec_key_name=encryption_spec_key_name,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    @base.optional_sync()
    def _create(
        cls,
        display_name: str,
        network: Optional[str] = None,
        public_endpoint_enabled: bool = False,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync: bool = True,
        enable_private_service_connect: bool = False,
        project_allowlist: Optional[Sequence[str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
    ) -> "MatchingEngineIndexEndpoint":
        """Helper method to ensure network synchronization and to
        create a MatchingEngineIndexEndpoint resource.

        Args:
            display_name (str):
                Required. The display name of the IndexEndpoint.
                The name can be up to 128 characters long and
                can be consist of any UTF-8 characters.
            network (str):
                Optional. The full name of the Google Compute Engine
                `network <https://cloud.google.com/compute/docs/networks-and-firewalls#networks>`__
                to which the IndexEndpoint should be peered.
                Private services access must already be configured for the network.

                `Format <https://cloud.google.com/compute/docs/reference/rest/v1/networks/insert>`__:
                projects/{project}/global/networks/{network}. Where
                {project} is a project number, as in '12345', and {network}
                is network name.
            public_endpoint_enabled (bool):
                Optional. If true, the deployed index will be
                accessible through public endpoint.
            description (str):
                Optional. The description of the IndexEndpoint.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your IndexEndpoint.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one
                IndexEndpoint (System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            project (str):
                Optional. Project to create IndexEndpoint in. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to create IndexEndpoint in. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create IndexEndpoints. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            sync (bool):
                Optional. Whether to execute this creation synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            encryption_spec_key_name (str):
                Immutable. Customer-managed encryption key
                spec for an IndexEndpoint. If set, this
                IndexEndpoint and all sub-resources of this
                IndexEndpoint will be secured by this key.
            enable_private_service_connect (bool):
                Required. If true, expose the IndexEndpoint
                via private service connect.
            project_allowlist (MutableSequence[str]):
                A list of Projects from which the forwarding
                rule will target the service attachment.
            create_request_timeout (float):
                Optional. The timeout for the request in seconds.

        Returns:
            MatchingEngineIndexEndpoint - IndexEndpoint resource object
        """
        # public
        if public_endpoint_enabled:
            gapic_index_endpoint = gca_matching_engine_index_endpoint.IndexEndpoint(
                display_name=display_name,
                description=description,
                public_endpoint_enabled=public_endpoint_enabled,
            )
        # PSA
        elif network:
            gapic_index_endpoint = gca_matching_engine_index_endpoint.IndexEndpoint(
                display_name=display_name, description=description, network=network
            )
        # PSC
        else:
            gapic_index_endpoint = gca_matching_engine_index_endpoint.IndexEndpoint(
                display_name=display_name,
                description=description,
                private_service_connect_config=gca_service_networking.PrivateServiceConnectConfig(
                    project_allowlist=project_allowlist,
                    enable_private_service_connect=enable_private_service_connect,
                ),
            )

        if encryption_spec_key_name:
            gapic_index_endpoint.encryption_spec = gca_encryption_spec.EncryptionSpec(
                kms_key_name=encryption_spec_key_name
            )

        if labels:
            utils.validate_labels(labels)
            gapic_index_endpoint.labels = labels

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        create_lro = api_client.create_index_endpoint(
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            index_endpoint=gapic_index_endpoint,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, create_lro)

        created_index = create_lro.result()

        _LOGGER.log_create_complete(cls, created_index, "index_endpoint")

        index_obj = cls(
            index_endpoint_name=created_index.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        return index_obj

    def _instantiate_public_match_client(
        self,
    ) -> utils.MatchClientWithOverride:
        """Helper method to instantiates match client with optional
        overrides for this endpoint.
        Returns:
            match_client (match_service_client.MatchServiceClient):
                Initialized match client with optional overrides.
        """
        return initializer.global_config.create_client(
            client_class=utils.MatchClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
            api_path_override=self.public_endpoint_domain_name,
        )

    def _instantiate_private_match_service_stub(
        self,
        deployed_index_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> match_service_pb2_grpc.MatchServiceStub:
        """Helper method to instantiate private match service stub.
        Args:
            deployed_index_id (str):
                Optional. Required for private service access endpoint.
                The user specified ID of the DeployedIndex.
            ip_address (str):
                Optional. Required for private service connect. The ip address
                the forwarding rule makes use of.
        Returns:
            stub (match_service_pb2_grpc.MatchServiceStub):
                Initialized match service stub.
        Raises:
            RuntimeError: No deployed index with id deployed_index_id found
            ValueError: Should not set ip address for networks other than
                        private service connect.
        """
        if ip_address:
            # Should only set for Private Service Connect
            if self.public_endpoint_domain_name:
                raise ValueError(
                    "MatchingEngineIndexEndpoint is set to use ",
                    "public network. Could not establish connection using "
                    "provided ip address",
                )
            elif self.private_service_access_network:
                raise ValueError(
                    "MatchingEngineIndexEndpoint is set to use ",
                    "private service access network. Could not establish "
                    "connection using provided ip address",
                )
        else:
            # Private Service Access, find server ip for deployed index
            deployed_indexes = [
                deployed_index
                for deployed_index in self.deployed_indexes
                if deployed_index.id == deployed_index_id
            ]

            if not deployed_indexes:
                raise RuntimeError(
                    f"No deployed index with id '{deployed_index_id}' found"
                )

            # Retrieve server ip from deployed index
            ip_address = deployed_indexes[0].private_endpoints.match_grpc_address

        if ip_address not in self._match_grpc_stub_cache:
            # Set up channel and stub
            channel = grpc.insecure_channel("{}:10000".format(ip_address))
            self._match_grpc_stub_cache[
                ip_address
            ] = match_service_pb2_grpc.MatchServiceStub(channel)
        return self._match_grpc_stub_cache[ip_address]

    @property
    def public_endpoint_domain_name(self) -> Optional[str]:
        """Public endpoint DNS name."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.public_endpoint_domain_name

    @property
    def private_service_access_network(self) -> Optional[str]:
        """ "Private service access network."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.network

    @property
    def private_service_connect_ip_address(self) -> Optional[str]:
        """ "Private service connect ip address."""
        return self._private_service_connect_ip_address

    @private_service_connect_ip_address.setter
    def private_service_connect_ip_address(self, ip_address: str) -> Optional[str]:
        """ "Setter for private service connect ip address."""
        self._private_service_connect_ip_address = ip_address

    def update(
        self,
        display_name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        update_request_timeout: Optional[float] = None,
    ) -> "MatchingEngineIndexEndpoint":
        """Updates an existing index endpoint resource.

        Args:
            display_name (str):
                Required. The display name of the IndexEndpoint.
                The name can be up to 128 characters long and
                can be consist of any UTF-8 characters.
            description (str):
                Optional. The description of the IndexEndpoint.
            labels (Dict[str, str]):
                Optional. The labels with user-defined
                metadata to organize your Indexs.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                on and examples of labels. No more than 64 user
                labels can be associated with one IndexEndpoint
                (System labels are excluded)."
                System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            update_request_timeout (float):
                Optional. The timeout for the request in seconds.

        Returns:
            MatchingEngineIndexEndpoint - The updated index endpoint resource object.
        """

        self.wait()

        update_mask = list()

        if labels:
            utils.validate_labels(labels)
            update_mask.append("labels")

        if display_name is not None:
            update_mask.append("display_name")

        if description is not None:
            update_mask.append("description")

        update_mask = field_mask_pb2.FieldMask(paths=update_mask)

        gapic_index_endpoint = gca_matching_engine_index_endpoint.IndexEndpoint(
            name=self.resource_name,
            display_name=display_name,
            description=description,
            labels=labels,
        )

        self._gca_resource = self.api_client.update_index_endpoint(
            index_endpoint=gapic_index_endpoint,
            update_mask=update_mask,
            metadata=request_metadata,
            timeout=update_request_timeout,
        )

        return self

    @staticmethod
    def _build_deployed_index(
        deployed_index_id: str,
        index_resource_name: Optional[str] = None,
        display_name: Optional[str] = None,
        machine_type: Optional[str] = None,
        min_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
        enable_access_logging: Optional[bool] = None,
        reserved_ip_ranges: Optional[Sequence[str]] = None,
        deployment_group: Optional[str] = None,
        auth_config_audiences: Optional[Sequence[str]] = None,
        auth_config_allowed_issuers: Optional[Sequence[str]] = None,
    ) -> gca_matching_engine_index_endpoint.DeployedIndex:
        """Builds a DeployedIndex.

        Args:
            deployed_index_id (str):
                Required. The user specified ID of the
                DeployedIndex. The ID can be up to 128
                characters long and must start with a letter and
                only contain letters, numbers, and underscores.
                The ID must be unique within the project it is
                created in.
            index_resource_name (str):
                Optional. A fully-qualified index endpoint resource name or a index ID.
                Example: "projects/123/locations/us-central1/index_endpoints/my_index_id"
            display_name (str):
                Optional. The display name of the DeployedIndex. If not provided upon
                creation, the Index's display_name is used.
            machine_type (str):
                Optional. The type of machine. Not specifying machine type will
                result in model to be deployed with automatic resources.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.

                If this value is not provided, the value of 2 will be used.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the larger value of min_replica_count or 2 will
                be used. If value provided is smaller than min_replica_count, it
                will automatically be increased to be min_replica_count.
            enable_access_logging (bool):
                Optional. If true, private endpoint's access
                logs are sent to StackDriver Logging.
                These logs are like standard server access logs,
                containing information like timestamp and
                latency for each MatchRequest.
                Note that Stackdriver logs may incur a cost,
                especially if the deployed index receives a high
                queries per second rate (QPS). Estimate your
                costs before enabling this option.
            deployed_index_auth_config (google.cloud.aiplatform_v1.types.DeployedIndexAuthConfig):
                Optional. If set, the authentication is
                enabled for the private endpoint.
            reserved_ip_ranges (Sequence[str]):
                Optional. A list of reserved ip ranges under
                the VPC network that can be used for this
                DeployedIndex.
                If set, we will deploy the index within the
                provided ip ranges. Otherwise, the index might
                be deployed to any ip ranges under the provided
                VPC network.

                The value sohuld be the name of the address
                (https://cloud.google.com/compute/docs/reference/rest/v1/addresses)
                Example: 'vertex-ai-ip-range'.
            deployment_group (str):
                Optional. The deployment group can be no longer than 64
                characters (eg: 'test', 'prod'). If not set, we will use the
                'default' deployment group.

                Creating ``deployment_groups`` with ``reserved_ip_ranges``
                is a recommended practice when the peered network has
                multiple peering ranges. This creates your deployments from
                predictable IP spaces for easier traffic administration.
                Also, one deployment_group (except 'default') can only be
                used with the same reserved_ip_ranges which means if the
                deployment_group has been used with reserved_ip_ranges: [a,
                b, c], using it with [a, b] or [d, e] is disallowed.

                Note: we only support up to 5 deployment groups(not
                including 'default').
            auth_config_audiences (Sequence[str]):
                Optional. The list of JWT
                `audiences <https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32#section-4.1.3>`__.
                that are allowed to access. A JWT containing any of these
                audiences will be accepted.
            auth_config_allowed_issuers (Sequence[str]):
                Optional. A list of allowed JWT issuers. Each entry must be a valid
                Google service account, in the following format:

                ``service-account-name@project-id.iam.gserviceaccount.com``
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
        """

        deployed_index = gca_matching_engine_index_endpoint.DeployedIndex(
            id=deployed_index_id,
            index=index_resource_name,
            display_name=display_name,
            enable_access_logging=enable_access_logging,
            reserved_ip_ranges=reserved_ip_ranges,
            deployment_group=deployment_group,
        )

        if auth_config_audiences and auth_config_allowed_issuers:
            deployed_index.deployed_index_auth_config = gca_matching_engine_index_endpoint.DeployedIndexAuthConfig(
                auth_provider=gca_matching_engine_index_endpoint.DeployedIndexAuthConfig.AuthProvider(
                    audiences=auth_config_audiences,
                    allowed_issuers=auth_config_allowed_issuers,
                )
            )

        if machine_type:
            machine_spec = gca_machine_resources_compat.MachineSpec(
                machine_type=machine_type
            )

            deployed_index.dedicated_resources = (
                gca_machine_resources_compat.DedicatedResources(
                    machine_spec=machine_spec,
                    min_replica_count=min_replica_count,
                    max_replica_count=max_replica_count,
                )
            )

        else:
            deployed_index.automatic_resources = (
                gca_machine_resources_compat.AutomaticResources(
                    min_replica_count=min_replica_count,
                    max_replica_count=max_replica_count,
                )
            )
        return deployed_index

    def deploy_index(
        self,
        index: matching_engine.MatchingEngineIndex,
        deployed_index_id: str,
        display_name: Optional[str] = None,
        machine_type: Optional[str] = None,
        min_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
        enable_access_logging: Optional[bool] = None,
        reserved_ip_ranges: Optional[Sequence[str]] = None,
        deployment_group: Optional[str] = None,
        auth_config_audiences: Optional[Sequence[str]] = None,
        auth_config_allowed_issuers: Optional[Sequence[str]] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        deploy_request_timeout: Optional[float] = None,
    ) -> "MatchingEngineIndexEndpoint":
        """Deploys an existing index resource to this endpoint resource.

        Args:
            index (MatchingEngineIndex):
                Required. The Index this is the
                deployment of. We may refer to this Index as the
                DeployedIndex's "original" Index.
            deployed_index_id (str):
                Required. The user specified ID of the
                DeployedIndex. The ID can be up to 128
                characters long and must start with a letter and
                only contain letters, numbers, and underscores.
                The ID must be unique within the project it is
                created in.
            display_name (str):
                The display name of the DeployedIndex. If not provided upon
                creation, the Index's display_name is used.
            machine_type (str):
                Optional. The type of machine. Not specifying machine type will
                result in model to be deployed with automatic resources.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.

                If this value is not provided, the value of 2 will be used.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the larger value of min_replica_count or 2 will
                be used. If value provided is smaller than min_replica_count, it
                will automatically be increased to be min_replica_count.
            enable_access_logging (bool):
                Optional. If true, private endpoint's access
                logs are sent to StackDriver Logging.
                These logs are like standard server access logs,
                containing information like timestamp and
                latency for each MatchRequest.
                Note that Stackdriver logs may incur a cost,
                especially if the deployed index receives a high
                queries per second rate (QPS). Estimate your
                costs before enabling this option.
            reserved_ip_ranges (Sequence[str]):
                Optional. A list of reserved ip ranges under
                the VPC network that can be used for this
                DeployedIndex.
                If set, we will deploy the index within the
                provided ip ranges. Otherwise, the index might
                be deployed to any ip ranges under the provided
                VPC network.

                The value sohuld be the name of the address
                (https://cloud.google.com/compute/docs/reference/rest/v1/addresses)
                Example: 'vertex-ai-ip-range'.
            deployment_group (str):
                Optional. The deployment group can be no longer than 64
                characters (eg: 'test', 'prod'). If not set, we will use the
                'default' deployment group.

                Creating ``deployment_groups`` with ``reserved_ip_ranges``
                is a recommended practice when the peered network has
                multiple peering ranges. This creates your deployments from
                predictable IP spaces for easier traffic administration.
                Also, one deployment_group (except 'default') can only be
                used with the same reserved_ip_ranges which means if the
                deployment_group has been used with reserved_ip_ranges: [a,
                b, c], using it with [a, b] or [d, e] is disallowed.

                Note: we only support up to 5 deployment groups(not
                including 'default').
            auth_config_audiences (Sequence[str]):
                The list of JWT
                `audiences <https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32#section-4.1.3>`__.
                that are allowed to access. A JWT containing any of these
                audiences will be accepted.

                auth_config_audiences and auth_config_allowed_issuers must be passed together.
            auth_config_allowed_issuers (Sequence[str]):
                A list of allowed JWT issuers. Each entry must be a valid
                Google service account, in the following format:

                ``service-account-name@project-id.iam.gserviceaccount.com``

                auth_config_audiences and auth_config_allowed_issuers must be passed together.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.

            deploy_request_timeout (float):
                Optional. The timeout for the request in seconds.
        Returns:
            MatchingEngineIndexEndpoint - IndexEndpoint resource object
        """

        self.wait()

        _LOGGER.log_action_start_against_resource(
            "Deploying index",
            "index_endpoint",
            self,
        )

        deployed_index = self._build_deployed_index(
            deployed_index_id=deployed_index_id,
            index_resource_name=index.resource_name,
            display_name=display_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            enable_access_logging=enable_access_logging,
            reserved_ip_ranges=reserved_ip_ranges,
            deployment_group=deployment_group,
            auth_config_audiences=auth_config_audiences,
            auth_config_allowed_issuers=auth_config_allowed_issuers,
        )

        deploy_lro = self.api_client.deploy_index(
            index_endpoint=self.resource_name,
            deployed_index=deployed_index,
            metadata=request_metadata,
            timeout=deploy_request_timeout,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Deploy index", "index_endpoint", self.__class__, deploy_lro
        )

        deploy_lro.result(timeout=None)

        _LOGGER.log_action_completed_against_resource(
            "index_endpoint", "Deployed index", self
        )

        # update local resource
        self._sync_gca_resource()

        return self

    def undeploy_index(
        self,
        deployed_index_id: str,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        undeploy_request_timeout: Optional[float] = None,
    ) -> "MatchingEngineIndexEndpoint":
        """Undeploy a deployed index endpoint resource.

        Args:
            deployed_index_id (str):
                Required. The ID of the DeployedIndex
                to be undeployed from the IndexEndpoint.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            undeploy_request_timeout (float):
                Optional. The timeout for the request in seconds.
        Returns:
            MatchingEngineIndexEndpoint - IndexEndpoint resource object
        """

        self.wait()

        _LOGGER.log_action_start_against_resource(
            "Undeploying index",
            "index_endpoint",
            self,
        )

        undeploy_lro = self.api_client.undeploy_index(
            index_endpoint=self.resource_name,
            deployed_index_id=deployed_index_id,
            metadata=request_metadata,
            timeout=undeploy_request_timeout,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Undeploy index", "index_endpoint", self.__class__, undeploy_lro
        )

        undeploy_lro.result()

        _LOGGER.log_action_completed_against_resource(
            "index_endpoint", "Undeployed index", self
        )

        return self

    def mutate_deployed_index(
        self,
        deployed_index_id: str,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        mutate_request_timeout: Optional[float] = None,
    ):
        """Updates an existing deployed index under this endpoint resource.

        Args:
            index_id (str):
                Required. The ID of the MatchingEnginIndex associated with the DeployedIndex.
            deployed_index_id (str):
                Required. The user specified ID of the
                DeployedIndex. The ID can be up to 128
                characters long and must start with a letter and
                only contain letters, numbers, and underscores.
                The ID must be unique within the project it is
                created in.
            min_replica_count (int):
                Optional. The minimum number of machine replicas this deployed
                model will be always deployed on. If traffic against it increases,
                it may dynamically be deployed onto more replicas, and as traffic
                decreases, some of these extra replicas may be freed.
            max_replica_count (int):
                Optional. The maximum number of replicas this deployed model may
                be deployed on when the traffic against it increases. If requested
                value is too large, the deployment will error, but if deployment
                succeeds then the ability to scale the model to that many replicas
                is guaranteed (barring service outages). If traffic against the
                deployed model increases beyond what its replicas at maximum may
                handle, a portion of the traffic will be dropped. If this value
                is not provided, the larger value of min_replica_count or 1 will
                be used. If value provided is smaller than min_replica_count, it
                will automatically be increased to be min_replica_count.
            request_metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as metadata.
            timeout (float):
                Optional. The timeout for the request in seconds.
        """

        self.wait()

        _LOGGER.log_action_start_against_resource(
            "Mutating index",
            "index_endpoint",
            self,
        )

        deployed_index = self._build_deployed_index(
            index_resource_name=None,
            deployed_index_id=deployed_index_id,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
        )

        deploy_lro = self.api_client.mutate_deployed_index(
            index_endpoint=self.resource_name,
            deployed_index=deployed_index,
            metadata=request_metadata,
            timeout=mutate_request_timeout,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Mutate index", "index_endpoint", self.__class__, deploy_lro
        )

        deploy_lro.result()

        # update local resource
        self._sync_gca_resource()

        _LOGGER.log_action_completed_against_resource("index_endpoint", "Mutated", self)

        return self

    @property
    def deployed_indexes(
        self,
    ) -> List[gca_matching_engine_index_endpoint.DeployedIndex]:
        """Returns a list of deployed indexes on this endpoint.

        Returns:
            List[gca_matching_engine_index_endpoint.DeployedIndex] - Deployed indexes
        """
        self._assert_gca_resource_is_available()
        return self._gca_resource.deployed_indexes

    @base.optional_sync()
    def _undeploy(
        self,
        deployed_index_id: str,
        metadata: Optional[Sequence[Tuple[str, str]]] = (),
        sync=True,
        undeploy_request_timeout: Optional[float] = None,
    ) -> None:
        """Undeploys a deployed index.

        Args:
            deployed_index_id (str):
                Required. The ID of the DeployedIndex to be undeployed from the
                Endpoint.
            metadata (Sequence[Tuple[str, str]]):
                Optional. Strings which should be sent along with the request as
                metadata.
            timeout (float):
                Optional. The timeout for the request in seconds.
        """
        self._sync_gca_resource()

        _LOGGER.log_action_start_against_resource("Undeploying", "index_endpoint", self)

        operation_future = self.api_client.undeploy_index(
            index_endpoint=self.resource_name,
            deployed_index_id=deployed_index_id,
            metadata=metadata,
            timeout=undeploy_request_timeout,
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Undeploy", "index_endpoint", self.__class__, operation_future
        )

        # block before returning
        operation_future.result()

        # update local resource
        self._sync_gca_resource()

        _LOGGER.log_action_completed_against_resource(
            "index_endpoint", "undeployed", self
        )

    def undeploy_all(self, sync: bool = True) -> "MatchingEngineIndexEndpoint":
        """Undeploys every index deployed to this MatchingEngineIndexEndpoint.

        Args:
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
        """
        self._sync_gca_resource()

        for deployed_index in self.deployed_indexes:
            self._undeploy(deployed_index_id=deployed_index.id, sync=sync)

        return self

    def delete(self, force: bool = False, sync: bool = True) -> None:
        """Deletes this MatchingEngineIndexEndpoint resource. If force is set to True,
        all indexes on this endpoint will be undeployed prior to deletion.

        Args:
            force (bool):
                Required. If force is set to True, all deployed indexes on this
                endpoint will be undeployed first. Default is False.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
        Raises:
            FailedPrecondition: If indexes are deployed on this MatchingEngineIndexEndpoint and force = False.
        """
        if force:
            self.undeploy_all(sync=sync)

        super().delete(sync=sync)

    @property
    def description(self) -> str:
        """Description of the index endpoint."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.description

    def find_neighbors(
        self,
        *,
        deployed_index_id: str,
        queries: Optional[List[List[float]]] = None,
        num_neighbors: int = 10,
        filter: Optional[List[Namespace]] = None,
        per_crowding_attribute_neighbor_count: Optional[int] = None,
        approx_num_neighbors: Optional[int] = None,
        fraction_leaf_nodes_to_search_override: Optional[float] = None,
        return_full_datapoint: bool = False,
        numeric_filter: Optional[List[NumericNamespace]] = None,
        embedding_ids: Optional[List[str]] = None,
    ) -> List[List[MatchNeighbor]]:
        """Retrieves nearest neighbors for the given embedding queries on the
        specified deployed index which is deployed to either public or private
        endpoint.

        ```
        Example usage:
            my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name='projects/123/locations/us-central1/index_endpoint/my_index_endpoint_id'
            )
            my_index_endpoint.find_neighbors(deployed_index_id="deployed_index_id", queries= [[1, 1]],)
        ```
        Args:
            deployed_index_id (str):
                Required. The ID of the DeployedIndex to match the queries against.
            queries (List[List[float]]):
                Required. A list of queries. Each query is a list of floats, representing a single embedding.
            num_neighbors (int):
                Required. The number of nearest neighbors to be retrieved from database for
                each query.
            filter (List[Namespace]):
                Optional. A list of Namespaces for filtering the matching results.
                For example, [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])] will match datapoints
                that satisfy "red color" but not include datapoints with "squared shape".
                Please refer to https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json for more detail.

            per_crowding_attribute_neighbor_count (int):
                Optional. Crowding is a constraint on a neighbor list produced
                by nearest neighbor search requiring that no more than some
                value k' of the k neighbors returned have the same value of
                crowding_attribute. It's used for improving result diversity.
                This field is the maximum number of matches with the same crowding tag.

            approx_num_neighbors (int):
                Optional. The number of neighbors to find via approximate search
                before exact reordering is performed. If not set, the default
                value from scam config is used; if set, this value must be > 0.

            fraction_leaf_nodes_to_search_override (float):
                Optional. The fraction of the number of leaves to search, set at
                query time allows user to tune search performance. This value
                increase result in both search accuracy and latency increase.
                The value should be between 0.0 and 1.0.

            return_full_datapoint (bool):
                Optional. If set to true, the full datapoints (including all
                vector values and of the nearest neighbors are returned.
                Note that returning full datapoint will significantly increase the
                latency and cost of the query.

            numeric_filter (list[NumericNamespace]):
                Optional. A list of NumericNamespaces for filtering the matching
                results. For example:
                [NumericNamespace(name="cost", value_int=5, op="GREATER")]
                will match datapoints that its cost is greater than 5.

            embedding_ids (str):
               Optional. If `queries` is set, will use `queries` to do nearest
               neighbor search. If `queries` isn't set, will first use
               `embedding_ids` to lookup embedding values from dataset, if embedding
               with `embedding_ids` exists in the dataset, do nearest neighbor search.

        Returns:
            List[List[MatchNeighbor]] - A list of nearest neighbors for each query.
        """

        if not self._public_match_client:
            # Private endpoint
            return self.match(
                deployed_index_id=deployed_index_id,
                queries=queries,
                num_neighbors=num_neighbors,
                filter=filter,
                per_crowding_attribute_num_neighbors=per_crowding_attribute_neighbor_count,
                approx_num_neighbors=approx_num_neighbors,
                fraction_leaf_nodes_to_search_override=fraction_leaf_nodes_to_search_override,
                numeric_filter=numeric_filter,
            )

        # Create the FindNeighbors request
        find_neighbors_request = gca_match_service_v1beta1.FindNeighborsRequest()
        find_neighbors_request.index_endpoint = self.resource_name
        find_neighbors_request.deployed_index_id = deployed_index_id
        find_neighbors_request.return_full_datapoint = return_full_datapoint

        # Token restricts
        restricts = []
        if filter:
            for namespace in filter:
                restrict = gca_index_v1beta1.IndexDatapoint.Restriction()
                restrict.namespace = namespace.name
                restrict.allow_list.extend(namespace.allow_tokens)
                restrict.deny_list.extend(namespace.deny_tokens)
                restricts.append(restrict)
        # Numeric restricts
        numeric_restricts = []
        if numeric_filter:
            for numeric_namespace in numeric_filter:
                numeric_restrict = gca_index_v1beta1.IndexDatapoint.NumericRestriction()
                numeric_restrict.namespace = numeric_namespace.name
                numeric_restrict.op = numeric_namespace.op
                numeric_restrict.value_int = numeric_namespace.value_int
                numeric_restrict.value_float = numeric_namespace.value_float
                numeric_restrict.value_double = numeric_namespace.value_double
                numeric_restricts.append(numeric_restrict)
        # Queries
        query_by_id = False if queries else True
        queries = queries if queries else embedding_ids
        if queries:
            for query in queries:
                find_neighbors_query = gca_match_service_v1beta1.FindNeighborsRequest.Query(
                    neighbor_count=num_neighbors,
                    per_crowding_attribute_neighbor_count=per_crowding_attribute_neighbor_count,
                    approximate_neighbor_count=approx_num_neighbors,
                    fraction_leaf_nodes_to_search_override=fraction_leaf_nodes_to_search_override,
                )
                datapoint = gca_index_v1beta1.IndexDatapoint(
                    datapoint_id=query if query_by_id else None,
                    feature_vector=None if query_by_id else query,
                )
                datapoint.restricts.extend(restricts)
                datapoint.numeric_restricts.extend(numeric_restricts)
                find_neighbors_query.datapoint = datapoint
                find_neighbors_request.queries.append(find_neighbors_query)
        else:
            raise ValueError(
                "To find neighbors using matching engine,"
                "please specify `queries` or `embedding_ids`"
            )

        response = self._public_match_client.find_neighbors(find_neighbors_request)

        # Wrap the results in MatchNeighbor objects and return
        return [
            [
                MatchNeighbor(
                    id=neighbor.datapoint.datapoint_id, distance=neighbor.distance
                ).from_index_datapoint(index_datapoint=neighbor.datapoint)
                for neighbor in embedding_neighbors.neighbors
            ]
            for embedding_neighbors in response.nearest_neighbors
        ]

    def read_index_datapoints(
        self,
        *,
        deployed_index_id: str,
        ids: List[str] = [],
    ) -> List[gca_index_v1beta1.IndexDatapoint]:
        """Reads the datapoints/vectors of the given IDs on the specified
        deployed index which is deployed to public or private endpoint.

        ```
        Example Usage:
            my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name='projects/123/locations/us-central1/index_endpoint/my_index_id'
            )
            my_index_endpoint.read_index_datapoints(deployed_index_id="public_test1", ids= ["606431", "896688"],)
        ```

        Args:
            deployed_index_id (str):
                Required. The ID of the DeployedIndex to match the queries against.
            ids (List[str]):
                Required. IDs of the datapoints to be searched for.
        Returns:
            List[gca_index_v1beta1.IndexDatapoint] - A list of datapoints/vectors of the given IDs.
        """
        if not self._public_match_client:
            # Call private match service stub with BatchGetEmbeddings request
            embeddings = self._batch_get_embeddings(
                deployed_index_id=deployed_index_id,
                ids=ids,
            )

            response = []
            for embedding in embeddings:
                index_datapoint = gca_index_v1beta1.IndexDatapoint(
                    datapoint_id=embedding.id,
                    feature_vector=embedding.float_val,
                    restricts=[
                        gca_index_v1beta1.IndexDatapoint.Restriction(
                            namespace=restrict.name,
                            allow_list=restrict.allow_tokens,
                            deny_list=restrict.deny_tokens,
                        )
                        for restrict in embedding.restricts
                    ],
                )
                if embedding.crowding_attribute:
                    index_datapoint.crowding_tag = (
                        gca_index_v1beta1.IndexDatapoint.CrowdingTag(
                            crowding_attribute=str(embedding.crowding_attribute)
                        )
                    )
                response.append(index_datapoint)
            return response

        # Create the ReadIndexDatapoints request
        read_index_datapoints_request = (
            gca_match_service_v1beta1.ReadIndexDatapointsRequest()
        )
        read_index_datapoints_request.index_endpoint = self.resource_name
        read_index_datapoints_request.deployed_index_id = deployed_index_id

        for id in ids:
            read_index_datapoints_request.ids.append(id)

        response = self._public_match_client.read_index_datapoints(
            read_index_datapoints_request
        )

        # Wrap the results and return
        return response.datapoints

    def _batch_get_embeddings(
        self,
        *,
        deployed_index_id: str,
        ids: List[str] = [],
    ) -> List[match_service_pb2.Embedding]:
        """
        Reads the datapoints/vectors of the given IDs on the specified index
        which is deployed to private endpoint.

        Args:
            deployed_index_id (str):
                Required. The ID of the DeployedIndex to match the queries against.
            ids (List[str]):
                Required. IDs of the datapoints to be searched for.
        Returns:
            List[match_service_pb2.Embedding] - A list of datapoints/vectors of the given IDs.
        """
        stub = self._instantiate_private_match_service_stub(
            deployed_index_id=deployed_index_id,
            ip_address=self._private_service_connect_ip_address,
        )

        # Create the batch get embeddings request
        batch_request = match_service_pb2.BatchGetEmbeddingsRequest()
        batch_request.deployed_index_id = deployed_index_id

        for id in ids:
            batch_request.id.append(id)
        response = stub.BatchGetEmbeddings(batch_request)

        return response.embeddings

    def match(
        self,
        deployed_index_id: str,
        queries: List[List[float]] = None,
        num_neighbors: int = 1,
        filter: Optional[List[Namespace]] = None,
        per_crowding_attribute_num_neighbors: Optional[int] = None,
        approx_num_neighbors: Optional[int] = None,
        fraction_leaf_nodes_to_search_override: Optional[float] = None,
        low_level_batch_size: int = 0,
        numeric_filter: Optional[List[NumericNamespace]] = None,
    ) -> List[List[MatchNeighbor]]:
        """Retrieves nearest neighbors for the given embedding queries on the
        specified deployed index for private endpoint only.

        Args:
            deployed_index_id (str):
                Required. The ID of the DeployedIndex to match the queries against.
            queries (List[List[float]]):
                Optional. A list of queries. Each query is a list of floats, representing a single embedding.
            num_neighbors (int):
                Required. The number of nearest neighbors to be retrieved from database for
                each query.
            filter (List[Namespace]):
                Optional. A list of Namespaces for filtering the matching results.
                For example, [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])] will match datapoints
                that satisfy "red color" but not include datapoints with "squared shape".
                Please refer to https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json for more detail.
            per_crowding_attribute_num_neighbors (int):
                Optional. Crowding is a constraint on a neighbor list produced by nearest neighbor
                search requiring that no more than some value k' of the k neighbors
                returned have the same value of crowding_attribute.
                It's used for improving result diversity.
                This field is the maximum number of matches with the same crowding tag.
            approx_num_neighbors (int):
                The number of neighbors to find via approximate search before exact reordering is performed.
                If not set, the default value from scam config is used; if set, this value must be > 0.
            fraction_leaf_nodes_to_search_override (float):
                Optional. The fraction of the number of leaves to search, set at
                query time allows user to tune search performance. This value
                increase result in both search accuracy and latency increase.
                The value should be between 0.0 and 1.0.
            low_level_batch_size (int):
                Optional. Selects the optimal batch size to use for low-level
                batching. Queries within each low level batch are executed
                sequentially while low level batches are executed in parallel.
                This field is optional, defaults to 0 if not set. A non-positive
                number disables low level batching, i.e. all queries are
                executed sequentially.
            numeric_filter (Optional[list[NumericNamespace]]):
                Optional. A list of NumericNamespaces for filtering the matching
                results. For example:
                [NumericNamespace(name="cost", value_int=5, op="GREATER")]
                will match datapoints that its cost is greater than 5.

        Returns:
            List[List[MatchNeighbor]] - A list of nearest neighbors for each query.
        """
        stub = self._instantiate_private_match_service_stub(
            deployed_index_id=deployed_index_id,
            ip_address=self._private_service_connect_ip_address,
        )

        # Create the batch match request
        batch_request = match_service_pb2.BatchMatchRequest()
        batch_request_for_index = (
            match_service_pb2.BatchMatchRequest.BatchMatchRequestPerIndex()
        )
        batch_request_for_index.deployed_index_id = deployed_index_id
        batch_request_for_index.low_level_batch_size = low_level_batch_size

        # Preprocess restricts to be used for each request
        restricts = []
        # Token restricts
        if filter:
            for namespace in filter:
                restrict = match_service_pb2.Namespace()
                restrict.name = namespace.name
                restrict.allow_tokens.extend(namespace.allow_tokens)
                restrict.deny_tokens.extend(namespace.deny_tokens)
                restricts.append(restrict)
        numeric_restricts = []
        # Numeric restricts
        if numeric_filter:
            for numeric_namespace in numeric_filter:
                numeric_restrict = match_service_pb2.NumericNamespace()
                numeric_restrict.name = numeric_namespace.name
                numeric_restrict.op = match_service_pb2.NumericNamespace.Operator.Value(
                    numeric_namespace.op
                )
                if numeric_namespace.value_int is not None:
                    numeric_restrict.value_int = numeric_namespace.value_int
                if numeric_namespace.value_float is not None:
                    numeric_restrict.value_float = numeric_namespace.value_float
                if numeric_namespace.value_double is not None:
                    numeric_restrict.value_double = numeric_namespace.value_double
                numeric_restricts.append(numeric_restrict)

        requests = []
        if queries:
            for query in queries:
                request = match_service_pb2.MatchRequest(
                    deployed_index_id=deployed_index_id,
                    float_val=query,
                    num_neighbors=num_neighbors,
                    restricts=restricts,
                    per_crowding_attribute_num_neighbors=per_crowding_attribute_num_neighbors,
                    approx_num_neighbors=approx_num_neighbors,
                    fraction_leaf_nodes_to_search_override=fraction_leaf_nodes_to_search_override,
                    numeric_restricts=numeric_restricts,
                )
                requests.append(request)
        else:
            raise ValueError(
                "To find neighbors using matching engine,"
                "please specify `queries` or `embedding_ids`"
            )

        batch_request_for_index.requests.extend(requests)
        batch_request.requests.append(batch_request_for_index)

        # Perform the request
        response = stub.BatchMatch(batch_request)

        # Wrap the results in MatchNeighbor objects and return
        match_neighbors_response = []
        for resp in response.responses[0].responses:
            match_neighbors_id_map = {}
            for neighbor in resp.neighbor:
                match_neighbors_id_map[neighbor.id] = MatchNeighbor(
                    id=neighbor.id, distance=neighbor.distance
                )
            for embedding in resp.embeddings:
                if embedding.id in match_neighbors_id_map:
                    match_neighbors_id_map[embedding.id] = match_neighbors_id_map[
                        embedding.id
                    ].from_embedding(embedding=embedding)
            match_neighbors_response.append(list(match_neighbors_id_map.values()))
        return match_neighbors_response
