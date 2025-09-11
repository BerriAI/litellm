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

from google.cloud.aiplatform_v1beta1.types import tool
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "HttpElementLocation",
        "AuthType",
        "Extension",
        "ExtensionManifest",
        "ExtensionOperation",
        "AuthConfig",
        "RuntimeConfig",
        "ExtensionPrivateServiceConnectConfig",
    },
)


class HttpElementLocation(proto.Enum):
    r"""Enum of location an HTTP element can be.

    Values:
        HTTP_IN_UNSPECIFIED (0):
            No description available.
        HTTP_IN_QUERY (1):
            Element is in the HTTP request query.
        HTTP_IN_HEADER (2):
            Element is in the HTTP request header.
        HTTP_IN_PATH (3):
            Element is in the HTTP request path.
        HTTP_IN_BODY (4):
            Element is in the HTTP request body.
        HTTP_IN_COOKIE (5):
            Element is in the HTTP request cookie.
    """
    HTTP_IN_UNSPECIFIED = 0
    HTTP_IN_QUERY = 1
    HTTP_IN_HEADER = 2
    HTTP_IN_PATH = 3
    HTTP_IN_BODY = 4
    HTTP_IN_COOKIE = 5


class AuthType(proto.Enum):
    r"""Type of Auth.

    Values:
        AUTH_TYPE_UNSPECIFIED (0):
            No description available.
        NO_AUTH (1):
            No Auth.
        API_KEY_AUTH (2):
            API Key Auth.
        HTTP_BASIC_AUTH (3):
            HTTP Basic Auth.
        GOOGLE_SERVICE_ACCOUNT_AUTH (4):
            Google Service Account Auth.
        OAUTH (6):
            OAuth auth.
        OIDC_AUTH (8):
            OpenID Connect (OIDC) Auth.
    """
    AUTH_TYPE_UNSPECIFIED = 0
    NO_AUTH = 1
    API_KEY_AUTH = 2
    HTTP_BASIC_AUTH = 3
    GOOGLE_SERVICE_ACCOUNT_AUTH = 4
    OAUTH = 6
    OIDC_AUTH = 8


class Extension(proto.Message):
    r"""Extensions are tools for large language models to access
    external data, run computations, etc.

    Attributes:
        name (str):
            Identifier. The resource name of the
            Extension.
        display_name (str):
            Required. The display name of the Extension.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        description (str):
            Optional. The description of the Extension.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Extension
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Extension
            was most recently updated.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        manifest (google.cloud.aiplatform_v1beta1.types.ExtensionManifest):
            Required. Manifest of the Extension.
        extension_operations (MutableSequence[google.cloud.aiplatform_v1beta1.types.ExtensionOperation]):
            Output only. Supported operations.
        runtime_config (google.cloud.aiplatform_v1beta1.types.RuntimeConfig):
            Optional. Runtime config controlling the
            runtime behavior of this Extension.
        tool_use_examples (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolUseExample]):
            Optional. Examples to illustrate the usage of
            the extension as a tool.
        private_service_connect_config (google.cloud.aiplatform_v1beta1.types.ExtensionPrivateServiceConnectConfig):
            Optional. The PrivateServiceConnect config
            for the extension. If specified, the service
            endpoints associated with the Extension should
            be registered with private network access in the
            provided Service Directory
            (https://cloud.google.com/service-directory/docs/configuring-private-network-access).

            If the service contains more than one endpoint
            with a network, the service will arbitrarilty
            choose one of the endpoints to use for extension
            execution.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    description: str = proto.Field(
        proto.STRING,
        number=4,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=7,
    )
    manifest: "ExtensionManifest" = proto.Field(
        proto.MESSAGE,
        number=9,
        message="ExtensionManifest",
    )
    extension_operations: MutableSequence["ExtensionOperation"] = proto.RepeatedField(
        proto.MESSAGE,
        number=11,
        message="ExtensionOperation",
    )
    runtime_config: "RuntimeConfig" = proto.Field(
        proto.MESSAGE,
        number=13,
        message="RuntimeConfig",
    )
    tool_use_examples: MutableSequence[tool.ToolUseExample] = proto.RepeatedField(
        proto.MESSAGE,
        number=15,
        message=tool.ToolUseExample,
    )
    private_service_connect_config: "ExtensionPrivateServiceConnectConfig" = (
        proto.Field(
            proto.MESSAGE,
            number=16,
            message="ExtensionPrivateServiceConnectConfig",
        )
    )


class ExtensionManifest(proto.Message):
    r"""Manifest spec of an Extension needed for runtime execution.

    Attributes:
        name (str):
            Required. Extension name shown to the LLM.
            The name can be up to 128 characters long.
        description (str):
            Required. The natural language description
            shown to the LLM. It should describe the usage
            of the extension, and is essential for the LLM
            to perform reasoning.
        api_spec (google.cloud.aiplatform_v1beta1.types.ExtensionManifest.ApiSpec):
            Required. Immutable. The API specification
            shown to the LLM.
        auth_config (google.cloud.aiplatform_v1beta1.types.AuthConfig):
            Required. Immutable. Type of auth supported
            by this extension.
    """

    class ApiSpec(proto.Message):
        r"""The API specification shown to the LLM.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            open_api_yaml (str):
                The API spec in Open API standard and YAML
                format.

                This field is a member of `oneof`_ ``api_spec``.
            open_api_gcs_uri (str):
                Cloud Storage URI pointing to the OpenAPI
                spec.

                This field is a member of `oneof`_ ``api_spec``.
        """

        open_api_yaml: str = proto.Field(
            proto.STRING,
            number=1,
            oneof="api_spec",
        )
        open_api_gcs_uri: str = proto.Field(
            proto.STRING,
            number=2,
            oneof="api_spec",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    description: str = proto.Field(
        proto.STRING,
        number=2,
    )
    api_spec: ApiSpec = proto.Field(
        proto.MESSAGE,
        number=3,
        message=ApiSpec,
    )
    auth_config: "AuthConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="AuthConfig",
    )


class ExtensionOperation(proto.Message):
    r"""Operation of an extension.

    Attributes:
        operation_id (str):
            Operation ID that uniquely identifies the
            operations among the extension. See: "Operation
            Object" in https://swagger.io/specification/.

            This field is parsed from the OpenAPI spec. For
            HTTP extensions, if it does not exist in the
            spec, we will generate one from the HTTP method
            and path.
        function_declaration (google.cloud.aiplatform_v1beta1.types.FunctionDeclaration):
            Output only. Structured representation of a
            function declaration as defined by the OpenAPI
            Spec.
    """

    operation_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    function_declaration: tool.FunctionDeclaration = proto.Field(
        proto.MESSAGE,
        number=3,
        message=tool.FunctionDeclaration,
    )


class AuthConfig(proto.Message):
    r"""Auth configuration to run the extension.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        api_key_config (google.cloud.aiplatform_v1beta1.types.AuthConfig.ApiKeyConfig):
            Config for API key auth.

            This field is a member of `oneof`_ ``auth_config``.
        http_basic_auth_config (google.cloud.aiplatform_v1beta1.types.AuthConfig.HttpBasicAuthConfig):
            Config for HTTP Basic auth.

            This field is a member of `oneof`_ ``auth_config``.
        google_service_account_config (google.cloud.aiplatform_v1beta1.types.AuthConfig.GoogleServiceAccountConfig):
            Config for Google Service Account auth.

            This field is a member of `oneof`_ ``auth_config``.
        oauth_config (google.cloud.aiplatform_v1beta1.types.AuthConfig.OauthConfig):
            Config for user oauth.

            This field is a member of `oneof`_ ``auth_config``.
        oidc_config (google.cloud.aiplatform_v1beta1.types.AuthConfig.OidcConfig):
            Config for user OIDC auth.

            This field is a member of `oneof`_ ``auth_config``.
        auth_type (google.cloud.aiplatform_v1beta1.types.AuthType):
            Type of auth scheme.
    """

    class ApiKeyConfig(proto.Message):
        r"""Config for authentication with API key.

        Attributes:
            name (str):
                Required. The parameter name of the API key. E.g. If the API
                request is "https://example.com/act?api_key=", "api_key"
                would be the parameter name.
            api_key_secret (str):
                Required. The name of the SecretManager secret version
                resource storing the API key. Format:
                ``projects/{project}/secrets/{secrete}/versions/{version}``

                -  If specified, the ``secretmanager.versions.access``
                   permission should be granted to Vertex AI Extension
                   Service Agent
                   (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents)
                   on the specified resource.
            http_element_location (google.cloud.aiplatform_v1beta1.types.HttpElementLocation):
                Required. The location of the API key.
        """

        name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        api_key_secret: str = proto.Field(
            proto.STRING,
            number=2,
        )
        http_element_location: "HttpElementLocation" = proto.Field(
            proto.ENUM,
            number=3,
            enum="HttpElementLocation",
        )

    class HttpBasicAuthConfig(proto.Message):
        r"""Config for HTTP Basic Authentication.

        Attributes:
            credential_secret (str):
                Required. The name of the SecretManager secret version
                resource storing the base64 encoded credentials. Format:
                ``projects/{project}/secrets/{secrete}/versions/{version}``

                -  If specified, the ``secretmanager.versions.access``
                   permission should be granted to Vertex AI Extension
                   Service Agent
                   (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents)
                   on the specified resource.
        """

        credential_secret: str = proto.Field(
            proto.STRING,
            number=2,
        )

    class GoogleServiceAccountConfig(proto.Message):
        r"""Config for Google Service Account Authentication.

        Attributes:
            service_account (str):
                Optional. The service account that the extension execution
                service runs as.

                -  If the service account is specified, the
                   ``iam.serviceAccounts.getAccessToken`` permission should
                   be granted to Vertex AI Extension Service Agent
                   (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents)
                   on the specified service account.

                -  If not specified, the Vertex AI Extension Service Agent
                   will be used to execute the Extension.
        """

        service_account: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class OauthConfig(proto.Message):
        r"""Config for user oauth.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            access_token (str):
                Access token for extension endpoint. Only used to propagate
                token from [[ExecuteExtensionRequest.runtime_auth_config]]
                at request time.

                This field is a member of `oneof`_ ``oauth_config``.
            service_account (str):
                The service account used to generate access tokens for
                executing the Extension.

                -  If the service account is specified, the
                   ``iam.serviceAccounts.getAccessToken`` permission should
                   be granted to Vertex AI Extension Service Agent
                   (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents)
                   on the provided service account.

                This field is a member of `oneof`_ ``oauth_config``.
        """

        access_token: str = proto.Field(
            proto.STRING,
            number=1,
            oneof="oauth_config",
        )
        service_account: str = proto.Field(
            proto.STRING,
            number=2,
            oneof="oauth_config",
        )

    class OidcConfig(proto.Message):
        r"""Config for user OIDC auth.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            id_token (str):
                OpenID Connect formatted ID token for extension endpoint.
                Only used to propagate token from
                [[ExecuteExtensionRequest.runtime_auth_config]] at request
                time.

                This field is a member of `oneof`_ ``oidc_config``.
            service_account (str):
                The service account used to generate an OpenID Connect
                (OIDC)-compatible JWT token signed by the Google OIDC
                Provider (accounts.google.com) for extension endpoint
                (https://cloud.google.com/iam/docs/create-short-lived-credentials-direct#sa-credentials-oidc).

                -  The audience for the token will be set to the URL in the
                   server url defined in the OpenApi spec.

                -  If the service account is provided, the service account
                   should grant ``iam.serviceAccounts.getOpenIdToken``
                   permission to Vertex AI Extension Service Agent
                   (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents).

                This field is a member of `oneof`_ ``oidc_config``.
        """

        id_token: str = proto.Field(
            proto.STRING,
            number=1,
            oneof="oidc_config",
        )
        service_account: str = proto.Field(
            proto.STRING,
            number=2,
            oneof="oidc_config",
        )

    api_key_config: ApiKeyConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="auth_config",
        message=ApiKeyConfig,
    )
    http_basic_auth_config: HttpBasicAuthConfig = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="auth_config",
        message=HttpBasicAuthConfig,
    )
    google_service_account_config: GoogleServiceAccountConfig = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="auth_config",
        message=GoogleServiceAccountConfig,
    )
    oauth_config: OauthConfig = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="auth_config",
        message=OauthConfig,
    )
    oidc_config: OidcConfig = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="auth_config",
        message=OidcConfig,
    )
    auth_type: "AuthType" = proto.Field(
        proto.ENUM,
        number=101,
        enum="AuthType",
    )


class RuntimeConfig(proto.Message):
    r"""Runtime configuration to run the extension.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        code_interpreter_runtime_config (google.cloud.aiplatform_v1beta1.types.RuntimeConfig.CodeInterpreterRuntimeConfig):
            Code execution runtime configurations for
            code interpreter extension.

            This field is a member of `oneof`_ ``GoogleFirstPartyExtensionConfig``.
        vertex_ai_search_runtime_config (google.cloud.aiplatform_v1beta1.types.RuntimeConfig.VertexAISearchRuntimeConfig):
            Runtime configuration for Vertext AI Search
            extension.

            This field is a member of `oneof`_ ``GoogleFirstPartyExtensionConfig``.
        default_params (google.protobuf.struct_pb2.Struct):
            Optional. Default parameters that will be set for all the
            execution of this extension. If specified, the parameter
            values can be overridden by values in
            [[ExecuteExtensionRequest.operation_params]] at request
            time.

            The struct should be in a form of map with param name as the
            key and actual param value as the value. E.g. If this
            operation requires a param "name" to be set to "abc". you
            can set this to something like {"name": "abc"}.
    """

    class CodeInterpreterRuntimeConfig(proto.Message):
        r"""

        Attributes:
            file_input_gcs_bucket (str):
                Optional. The GCS bucket for file input of
                this Extension. If specified, support input from
                the GCS bucket. Vertex Extension Custom Code
                Service Agent should be granted file reader to
                this bucket.
                If not specified, the extension will only accept
                file contents from request body and reject GCS
                file inputs.
            file_output_gcs_bucket (str):
                Optional. The GCS bucket for file output of
                this Extension. If specified, write all output
                files to the GCS bucket. Vertex Extension Custom
                Code Service Agent should be granted file writer
                to this bucket.
                If not specified, the file content will be
                output in response body.
        """

        file_input_gcs_bucket: str = proto.Field(
            proto.STRING,
            number=1,
        )
        file_output_gcs_bucket: str = proto.Field(
            proto.STRING,
            number=2,
        )

    class VertexAISearchRuntimeConfig(proto.Message):
        r"""

        Attributes:
            serving_config_name (str):
                Required. Vertext AI Search serving config name. Format:
                ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/servingConfigs/{serving_config}``
                or
                ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/servingConfigs/{serving_config}``
        """

        serving_config_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    code_interpreter_runtime_config: CodeInterpreterRuntimeConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="GoogleFirstPartyExtensionConfig",
        message=CodeInterpreterRuntimeConfig,
    )
    vertex_ai_search_runtime_config: VertexAISearchRuntimeConfig = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="GoogleFirstPartyExtensionConfig",
        message=VertexAISearchRuntimeConfig,
    )
    default_params: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=4,
        message=struct_pb2.Struct,
    )


class ExtensionPrivateServiceConnectConfig(proto.Message):
    r"""PrivateExtensionConfig configuration for the extension.

    Attributes:
        service_directory (str):
            Required. The Service Directory resource name in which the
            service endpoints associated to the extension are
            registered. Format:
            ``projects/{project_id}/locations/{location_id}/namespaces/{namespace_id}/services/{service_id}``

            -  The Vertex AI Extension Service Agent
               (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents)
               should be granted ``servicedirectory.viewer`` and
               ``servicedirectory.pscAuthorizedService`` roles on the
               resource.
    """

    service_directory: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
