# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1beta1.types import machine_resources
from google.cloud.aiplatform_v1beta1.types import service_networking
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "IndexEndpoint",
        "DeployedIndex",
        "DeployedIndexAuthConfig",
        "IndexPrivateEndpoints",
    },
)


class IndexEndpoint(proto.Message):
    r"""Indexes are deployed into it. An IndexEndpoint can have
    multiple DeployedIndexes.

    Attributes:
        name (str):
            Output only. The resource name of the
            IndexEndpoint.
        display_name (str):
            Required. The display name of the
            IndexEndpoint. The name can be up to 128
            characters long and can consist of any UTF-8
            characters.
        description (str):
            The description of the IndexEndpoint.
        deployed_indexes (MutableSequence[google.cloud.aiplatform_v1beta1.types.DeployedIndex]):
            Output only. The indexes deployed in this
            endpoint.
        etag (str):
            Used to perform consistent read-modify-write
            updates. If not set, a blind "overwrite" update
            happens.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize your IndexEndpoints.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            IndexEndpoint was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            IndexEndpoint was last updated. This timestamp
            is not updated when the endpoint's
            DeployedIndexes are updated, e.g. due to updates
            of the original Indexes they are the deployments
            of.
        network (str):
            Optional. The full name of the Google Compute Engine
            `network <https://cloud.google.com/compute/docs/networks-and-firewalls#networks>`__
            to which the IndexEndpoint should be peered.

            Private services access must already be configured for the
            network. If left unspecified, the Endpoint is not peered
            with any network.

            [network][google.cloud.aiplatform.v1beta1.IndexEndpoint.network]
            and
            [private_service_connect_config][google.cloud.aiplatform.v1beta1.IndexEndpoint.private_service_connect_config]
            are mutually exclusive.

            `Format <https://cloud.google.com/compute/docs/reference/rest/v1/networks/insert>`__:
            ``projects/{project}/global/networks/{network}``. Where
            {project} is a project number, as in '12345', and {network}
            is network name.
        enable_private_service_connect (bool):
            Optional. Deprecated: If true, expose the IndexEndpoint via
            private service connect.

            Only one of the fields,
            [network][google.cloud.aiplatform.v1beta1.IndexEndpoint.network]
            or
            [enable_private_service_connect][google.cloud.aiplatform.v1beta1.IndexEndpoint.enable_private_service_connect],
            can be set.
        private_service_connect_config (google.cloud.aiplatform_v1beta1.types.PrivateServiceConnectConfig):
            Optional. Configuration for private service connect.

            [network][google.cloud.aiplatform.v1beta1.IndexEndpoint.network]
            and
            [private_service_connect_config][google.cloud.aiplatform.v1beta1.IndexEndpoint.private_service_connect_config]
            are mutually exclusive.
        public_endpoint_enabled (bool):
            Optional. If true, the deployed index will be
            accessible through public endpoint.
        public_endpoint_domain_name (str):
            Output only. If
            [public_endpoint_enabled][google.cloud.aiplatform.v1beta1.IndexEndpoint.public_endpoint_enabled]
            is true, this field will be populated with the domain name
            to use for this index endpoint.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Immutable. Customer-managed encryption key
            spec for an IndexEndpoint. If set, this
            IndexEndpoint and all sub-resources of this
            IndexEndpoint will be secured by this key.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    deployed_indexes: MutableSequence["DeployedIndex"] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message="DeployedIndex",
    )
    etag: str = proto.Field(
        proto.STRING,
        number=5,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=6,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    network: str = proto.Field(
        proto.STRING,
        number=9,
    )
    enable_private_service_connect: bool = proto.Field(
        proto.BOOL,
        number=10,
    )
    private_service_connect_config: service_networking.PrivateServiceConnectConfig = (
        proto.Field(
            proto.MESSAGE,
            number=12,
            message=service_networking.PrivateServiceConnectConfig,
        )
    )
    public_endpoint_enabled: bool = proto.Field(
        proto.BOOL,
        number=13,
    )
    public_endpoint_domain_name: str = proto.Field(
        proto.STRING,
        number=14,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=15,
        message=gca_encryption_spec.EncryptionSpec,
    )


class DeployedIndex(proto.Message):
    r"""A deployment of an Index. IndexEndpoints contain one or more
    DeployedIndexes.

    Attributes:
        id (str):
            Required. The user specified ID of the
            DeployedIndex. The ID can be up to 128
            characters long and must start with a letter and
            only contain letters, numbers, and underscores.
            The ID must be unique within the project it is
            created in.
        index (str):
            Required. The name of the Index this is the
            deployment of. We may refer to this Index as the
            DeployedIndex's "original" Index.
        display_name (str):
            The display name of the DeployedIndex. If not provided upon
            creation, the Index's display_name is used.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the DeployedIndex
            was created.
        private_endpoints (google.cloud.aiplatform_v1beta1.types.IndexPrivateEndpoints):
            Output only. Provides paths for users to send requests
            directly to the deployed index services running on Cloud via
            private services access. This field is populated if
            [network][google.cloud.aiplatform.v1beta1.IndexEndpoint.network]
            is configured.
        index_sync_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The DeployedIndex may depend on various data on
            its original Index. Additionally when certain changes to the
            original Index are being done (e.g. when what the Index
            contains is being changed) the DeployedIndex may be
            asynchronously updated in the background to reflect these
            changes. If this timestamp's value is at least the
            [Index.update_time][google.cloud.aiplatform.v1beta1.Index.update_time]
            of the original Index, it means that this DeployedIndex and
            the original Index are in sync. If this timestamp is older,
            then to see which updates this DeployedIndex already
            contains (and which it does not), one must
            [list][google.longrunning.Operations.ListOperations] the
            operations that are running on the original Index. Only the
            successfully completed Operations with
            [update_time][google.cloud.aiplatform.v1beta1.GenericOperationMetadata.update_time]
            equal or before this sync time are contained in this
            DeployedIndex.
        automatic_resources (google.cloud.aiplatform_v1beta1.types.AutomaticResources):
            Optional. A description of resources that the DeployedIndex
            uses, which to large degree are decided by Vertex AI, and
            optionally allows only a modest additional configuration. If
            min_replica_count is not set, the default value is 2 (we
            don't provide SLA when min_replica_count=1). If
            max_replica_count is not set, the default value is
            min_replica_count. The max allowed replica count is 1000.
        dedicated_resources (google.cloud.aiplatform_v1beta1.types.DedicatedResources):
            Optional. A description of resources that are dedicated to
            the DeployedIndex, and that need a higher degree of manual
            configuration. The field min_replica_count must be set to a
            value strictly greater than 0, or else validation will fail.
            We don't provide SLA when min_replica_count=1. If
            max_replica_count is not set, the default value is
            min_replica_count. The max allowed replica count is 1000.

            Available machine types for SMALL shard: e2-standard-2 and
            all machine types available for MEDIUM and LARGE shard.

            Available machine types for MEDIUM shard: e2-standard-16 and
            all machine types available for LARGE shard.

            Available machine types for LARGE shard: e2-highmem-16,
            n2d-standard-32.

            n1-standard-16 and n1-standard-32 are still available, but
            we recommend e2-standard-16 and e2-highmem-16 for cost
            efficiency.
        enable_access_logging (bool):
            Optional. If true, private endpoint's access
            logs are sent to Cloud Logging.

            These logs are like standard server access logs,
            containing information like timestamp and
            latency for each MatchRequest.

            Note that logs may incur a cost, especially if
            the deployed index receives a high queries per
            second rate (QPS). Estimate your costs before
            enabling this option.
        deployed_index_auth_config (google.cloud.aiplatform_v1beta1.types.DeployedIndexAuthConfig):
            Optional. If set, the authentication is
            enabled for the private endpoint.
        reserved_ip_ranges (MutableSequence[str]):
            Optional. A list of reserved ip ranges under the VPC network
            that can be used for this DeployedIndex.

            If set, we will deploy the index within the provided ip
            ranges. Otherwise, the index might be deployed to any ip
            ranges under the provided VPC network.

            The value should be the name of the address
            (https://cloud.google.com/compute/docs/reference/rest/v1/addresses)
            Example: ['vertex-ai-ip-range'].

            For more information about subnets and network IP ranges,
            please see
            https://cloud.google.com/vpc/docs/subnets#manually_created_subnet_ip_ranges.
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
    """

    id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    index: str = proto.Field(
        proto.STRING,
        number=2,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    private_endpoints: "IndexPrivateEndpoints" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="IndexPrivateEndpoints",
    )
    index_sync_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    automatic_resources: machine_resources.AutomaticResources = proto.Field(
        proto.MESSAGE,
        number=7,
        message=machine_resources.AutomaticResources,
    )
    dedicated_resources: machine_resources.DedicatedResources = proto.Field(
        proto.MESSAGE,
        number=16,
        message=machine_resources.DedicatedResources,
    )
    enable_access_logging: bool = proto.Field(
        proto.BOOL,
        number=8,
    )
    deployed_index_auth_config: "DeployedIndexAuthConfig" = proto.Field(
        proto.MESSAGE,
        number=9,
        message="DeployedIndexAuthConfig",
    )
    reserved_ip_ranges: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=10,
    )
    deployment_group: str = proto.Field(
        proto.STRING,
        number=11,
    )


class DeployedIndexAuthConfig(proto.Message):
    r"""Used to set up the auth on the DeployedIndex's private
    endpoint.

    Attributes:
        auth_provider (google.cloud.aiplatform_v1beta1.types.DeployedIndexAuthConfig.AuthProvider):
            Defines the authentication provider that the
            DeployedIndex uses.
    """

    class AuthProvider(proto.Message):
        r"""Configuration for an authentication provider, including support for
        `JSON Web Token
        (JWT) <https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32>`__.

        Attributes:
            audiences (MutableSequence[str]):
                The list of JWT
                `audiences <https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32#section-4.1.3>`__.
                that are allowed to access. A JWT containing any of these
                audiences will be accepted.
            allowed_issuers (MutableSequence[str]):
                A list of allowed JWT issuers. Each entry must be a valid
                Google service account, in the following format:

                ``service-account-name@project-id.iam.gserviceaccount.com``
        """

        audiences: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )
        allowed_issuers: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )

    auth_provider: AuthProvider = proto.Field(
        proto.MESSAGE,
        number=1,
        message=AuthProvider,
    )


class IndexPrivateEndpoints(proto.Message):
    r"""IndexPrivateEndpoints proto is used to provide paths for users to
    send requests via private endpoints (e.g. private service access,
    private service connect). To send request via private service
    access, use match_grpc_address. To send request via private service
    connect, use service_attachment.

    Attributes:
        match_grpc_address (str):
            Output only. The ip address used to send
            match gRPC requests.
        service_attachment (str):
            Output only. The name of the service
            attachment resource. Populated if private
            service connect is enabled.
        psc_automated_endpoints (MutableSequence[google.cloud.aiplatform_v1beta1.types.PscAutomatedEndpoints]):
            Output only. PscAutomatedEndpoints is
            populated if private service connect is enabled
            if PscAutomatedConfig is set.
    """

    match_grpc_address: str = proto.Field(
        proto.STRING,
        number=1,
    )
    service_attachment: str = proto.Field(
        proto.STRING,
        number=2,
    )
    psc_automated_endpoints: MutableSequence[
        service_networking.PscAutomatedEndpoints
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=service_networking.PscAutomatedEndpoints,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
