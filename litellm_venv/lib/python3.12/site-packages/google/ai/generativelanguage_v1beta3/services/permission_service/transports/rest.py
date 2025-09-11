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

import dataclasses
import json  # type: ignore
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from google.api_core import gapic_v1, path_template, rest_helpers, rest_streaming
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore


from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore

from google.ai.generativelanguage_v1beta3.types import permission as gag_permission
from google.ai.generativelanguage_v1beta3.types import permission
from google.ai.generativelanguage_v1beta3.types import permission_service

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import PermissionServiceTransport

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=requests_version,
)


class PermissionServiceRestInterceptor:
    """Interceptor for PermissionService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the PermissionServiceRestTransport.

    .. code-block:: python
        class MyCustomPermissionServiceInterceptor(PermissionServiceRestInterceptor):
            def pre_create_permission(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_permission(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_permission(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_permission(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_permission(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_permissions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_permissions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_transfer_ownership(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_transfer_ownership(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_permission(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_permission(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = PermissionServiceRestTransport(interceptor=MyCustomPermissionServiceInterceptor())
        client = PermissionServiceClient(transport=transport)


    """

    def pre_create_permission(
        self,
        request: permission_service.CreatePermissionRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[permission_service.CreatePermissionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_permission

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PermissionService server.
        """
        return request, metadata

    def post_create_permission(
        self, response: gag_permission.Permission
    ) -> gag_permission.Permission:
        """Post-rpc interceptor for create_permission

        Override in a subclass to manipulate the response
        after it is returned by the PermissionService server but before
        it is returned to user code.
        """
        return response

    def pre_delete_permission(
        self,
        request: permission_service.DeletePermissionRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[permission_service.DeletePermissionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_permission

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PermissionService server.
        """
        return request, metadata

    def pre_get_permission(
        self,
        request: permission_service.GetPermissionRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[permission_service.GetPermissionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_permission

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PermissionService server.
        """
        return request, metadata

    def post_get_permission(
        self, response: permission.Permission
    ) -> permission.Permission:
        """Post-rpc interceptor for get_permission

        Override in a subclass to manipulate the response
        after it is returned by the PermissionService server but before
        it is returned to user code.
        """
        return response

    def pre_list_permissions(
        self,
        request: permission_service.ListPermissionsRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[permission_service.ListPermissionsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_permissions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PermissionService server.
        """
        return request, metadata

    def post_list_permissions(
        self, response: permission_service.ListPermissionsResponse
    ) -> permission_service.ListPermissionsResponse:
        """Post-rpc interceptor for list_permissions

        Override in a subclass to manipulate the response
        after it is returned by the PermissionService server but before
        it is returned to user code.
        """
        return response

    def pre_transfer_ownership(
        self,
        request: permission_service.TransferOwnershipRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[permission_service.TransferOwnershipRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for transfer_ownership

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PermissionService server.
        """
        return request, metadata

    def post_transfer_ownership(
        self, response: permission_service.TransferOwnershipResponse
    ) -> permission_service.TransferOwnershipResponse:
        """Post-rpc interceptor for transfer_ownership

        Override in a subclass to manipulate the response
        after it is returned by the PermissionService server but before
        it is returned to user code.
        """
        return response

    def pre_update_permission(
        self,
        request: permission_service.UpdatePermissionRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[permission_service.UpdatePermissionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_permission

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PermissionService server.
        """
        return request, metadata

    def post_update_permission(
        self, response: gag_permission.Permission
    ) -> gag_permission.Permission:
        """Post-rpc interceptor for update_permission

        Override in a subclass to manipulate the response
        after it is returned by the PermissionService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class PermissionServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: PermissionServiceRestInterceptor


class PermissionServiceRestTransport(PermissionServiceTransport):
    """REST backend transport for PermissionService.

    Provides methods for managing permissions to PaLM API
    resources.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    """

    def __init__(
        self,
        *,
        host: str = "generativelanguage.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[PermissionServiceRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'generativelanguage.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        # Run the base constructor
        # TODO(yon-mg): resolve other ctor params i.e. scopes, quota, etc.
        # TODO: When custom host (api_endpoint) is set, `scopes` must *also* be set on the
        # credentials object
        maybe_url_match = re.match("^(?P<scheme>http(?:s)?://)?(?P<host>.*)$", host)
        if maybe_url_match is None:
            raise ValueError(
                f"Unexpected hostname structure: {host}"
            )  # pragma: NO COVER

        url_match_items = maybe_url_match.groupdict()

        host = f"{url_scheme}://{host}" if not url_match_items["scheme"] else host

        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            api_audience=api_audience,
        )
        self._session = AuthorizedSession(
            self._credentials, default_host=self.DEFAULT_HOST
        )
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or PermissionServiceRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CreatePermission(PermissionServiceRestStub):
        def __hash__(self):
            return hash("CreatePermission")

        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {
                k: v
                for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items()
                if k not in message_dict
            }

        def __call__(
            self,
            request: permission_service.CreatePermissionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> gag_permission.Permission:
            r"""Call the create permission method over HTTP.

            Args:
                request (~.permission_service.CreatePermissionRequest):
                    The request object. Request to create a ``Permission``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.gag_permission.Permission:
                    Permission resource grants user,
                group or the rest of the world access to
                the PaLM API resource (e.g. a tuned
                model, file).

                A role is a collection of permitted
                operations that allows users to perform
                specific actions on PaLM API resources.
                To make them available to users, groups,
                or service accounts, you assign roles.
                When you assign a role, you grant
                permissions that the role contains.

                There are three concentric roles. Each
                role is a superset of the previous
                role's permitted operations:

                - reader can use the resource (e.g.
                  tuned model) for inference
                - writer has reader's permissions and
                  additionally can edit and share
                - owner has writer's permissions and
                  additionally can delete

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta3/{parent=tunedModels/*}/permissions",
                    "body": "permission",
                },
            ]
            request, metadata = self._interceptor.pre_create_permission(
                request, metadata
            )
            pb_request = permission_service.CreatePermissionRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request["body"], use_integers_for_enums=True
            )
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]

            # Jsonify the query params
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request["query_params"],
                    use_integers_for_enums=True,
                )
            )
            query_params.update(self._get_unset_required_fields(query_params))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = gag_permission.Permission()
            pb_resp = gag_permission.Permission.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_permission(resp)
            return resp

    class _DeletePermission(PermissionServiceRestStub):
        def __hash__(self):
            return hash("DeletePermission")

        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {
                k: v
                for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items()
                if k not in message_dict
            }

        def __call__(
            self,
            request: permission_service.DeletePermissionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ):
            r"""Call the delete permission method over HTTP.

            Args:
                request (~.permission_service.DeletePermissionRequest):
                    The request object. Request to delete the ``Permission``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "delete",
                    "uri": "/v1beta3/{name=tunedModels/*/permissions/*}",
                },
            ]
            request, metadata = self._interceptor.pre_delete_permission(
                request, metadata
            )
            pb_request = permission_service.DeletePermissionRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request["uri"]
            method = transcoded_request["method"]

            # Jsonify the query params
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request["query_params"],
                    use_integers_for_enums=True,
                )
            )
            query_params.update(self._get_unset_required_fields(query_params))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _GetPermission(PermissionServiceRestStub):
        def __hash__(self):
            return hash("GetPermission")

        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {
                k: v
                for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items()
                if k not in message_dict
            }

        def __call__(
            self,
            request: permission_service.GetPermissionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> permission.Permission:
            r"""Call the get permission method over HTTP.

            Args:
                request (~.permission_service.GetPermissionRequest):
                    The request object. Request for getting information about a specific
                ``Permission``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.permission.Permission:
                    Permission resource grants user,
                group or the rest of the world access to
                the PaLM API resource (e.g. a tuned
                model, file).

                A role is a collection of permitted
                operations that allows users to perform
                specific actions on PaLM API resources.
                To make them available to users, groups,
                or service accounts, you assign roles.
                When you assign a role, you grant
                permissions that the role contains.

                There are three concentric roles. Each
                role is a superset of the previous
                role's permitted operations:

                - reader can use the resource (e.g.
                  tuned model) for inference
                - writer has reader's permissions and
                  additionally can edit and share
                - owner has writer's permissions and
                  additionally can delete

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "get",
                    "uri": "/v1beta3/{name=tunedModels/*/permissions/*}",
                },
            ]
            request, metadata = self._interceptor.pre_get_permission(request, metadata)
            pb_request = permission_service.GetPermissionRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request["uri"]
            method = transcoded_request["method"]

            # Jsonify the query params
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request["query_params"],
                    use_integers_for_enums=True,
                )
            )
            query_params.update(self._get_unset_required_fields(query_params))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = permission.Permission()
            pb_resp = permission.Permission.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_permission(resp)
            return resp

    class _ListPermissions(PermissionServiceRestStub):
        def __hash__(self):
            return hash("ListPermissions")

        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {
                k: v
                for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items()
                if k not in message_dict
            }

        def __call__(
            self,
            request: permission_service.ListPermissionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> permission_service.ListPermissionsResponse:
            r"""Call the list permissions method over HTTP.

            Args:
                request (~.permission_service.ListPermissionsRequest):
                    The request object. Request for listing permissions.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.permission_service.ListPermissionsResponse:
                    Response from ``ListPermissions`` containing a paginated
                list of permissions.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "get",
                    "uri": "/v1beta3/{parent=tunedModels/*}/permissions",
                },
            ]
            request, metadata = self._interceptor.pre_list_permissions(
                request, metadata
            )
            pb_request = permission_service.ListPermissionsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request["uri"]
            method = transcoded_request["method"]

            # Jsonify the query params
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request["query_params"],
                    use_integers_for_enums=True,
                )
            )
            query_params.update(self._get_unset_required_fields(query_params))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = permission_service.ListPermissionsResponse()
            pb_resp = permission_service.ListPermissionsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_permissions(resp)
            return resp

    class _TransferOwnership(PermissionServiceRestStub):
        def __hash__(self):
            return hash("TransferOwnership")

        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {
                k: v
                for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items()
                if k not in message_dict
            }

        def __call__(
            self,
            request: permission_service.TransferOwnershipRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> permission_service.TransferOwnershipResponse:
            r"""Call the transfer ownership method over HTTP.

            Args:
                request (~.permission_service.TransferOwnershipRequest):
                    The request object. Request to transfer the ownership of
                the tuned model.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.permission_service.TransferOwnershipResponse:
                    Response from ``TransferOwnership``.
            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta3/{name=tunedModels/*}:transferOwnership",
                    "body": "*",
                },
            ]
            request, metadata = self._interceptor.pre_transfer_ownership(
                request, metadata
            )
            pb_request = permission_service.TransferOwnershipRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request["body"], use_integers_for_enums=True
            )
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]

            # Jsonify the query params
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request["query_params"],
                    use_integers_for_enums=True,
                )
            )
            query_params.update(self._get_unset_required_fields(query_params))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = permission_service.TransferOwnershipResponse()
            pb_resp = permission_service.TransferOwnershipResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_transfer_ownership(resp)
            return resp

    class _UpdatePermission(PermissionServiceRestStub):
        def __hash__(self):
            return hash("UpdatePermission")

        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {
            "updateMask": {},
        }

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {
                k: v
                for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items()
                if k not in message_dict
            }

        def __call__(
            self,
            request: permission_service.UpdatePermissionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> gag_permission.Permission:
            r"""Call the update permission method over HTTP.

            Args:
                request (~.permission_service.UpdatePermissionRequest):
                    The request object. Request to update the ``Permission``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.gag_permission.Permission:
                    Permission resource grants user,
                group or the rest of the world access to
                the PaLM API resource (e.g. a tuned
                model, file).

                A role is a collection of permitted
                operations that allows users to perform
                specific actions on PaLM API resources.
                To make them available to users, groups,
                or service accounts, you assign roles.
                When you assign a role, you grant
                permissions that the role contains.

                There are three concentric roles. Each
                role is a superset of the previous
                role's permitted operations:

                - reader can use the resource (e.g.
                  tuned model) for inference
                - writer has reader's permissions and
                  additionally can edit and share
                - owner has writer's permissions and
                  additionally can delete

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "patch",
                    "uri": "/v1beta3/{permission.name=tunedModels/*/permissions/*}",
                    "body": "permission",
                },
            ]
            request, metadata = self._interceptor.pre_update_permission(
                request, metadata
            )
            pb_request = permission_service.UpdatePermissionRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request["body"], use_integers_for_enums=True
            )
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]

            # Jsonify the query params
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request["query_params"],
                    use_integers_for_enums=True,
                )
            )
            query_params.update(self._get_unset_required_fields(query_params))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = gag_permission.Permission()
            pb_resp = gag_permission.Permission.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_permission(resp)
            return resp

    @property
    def create_permission(
        self,
    ) -> Callable[
        [permission_service.CreatePermissionRequest], gag_permission.Permission
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreatePermission(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_permission(
        self,
    ) -> Callable[[permission_service.DeletePermissionRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeletePermission(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_permission(
        self,
    ) -> Callable[[permission_service.GetPermissionRequest], permission.Permission]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetPermission(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_permissions(
        self,
    ) -> Callable[
        [permission_service.ListPermissionsRequest],
        permission_service.ListPermissionsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListPermissions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def transfer_ownership(
        self,
    ) -> Callable[
        [permission_service.TransferOwnershipRequest],
        permission_service.TransferOwnershipResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._TransferOwnership(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_permission(
        self,
    ) -> Callable[
        [permission_service.UpdatePermissionRequest], gag_permission.Permission
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdatePermission(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("PermissionServiceRestTransport",)
