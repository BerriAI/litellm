# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
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
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1, rest_helpers, rest_streaming
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
import google.protobuf
from google.protobuf import json_format
from requests import __version__ as requests_version

from google.cloud.iam_credentials_v1.types import common

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseIAMCredentialsRestTransport

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = logging.getLogger(__name__)

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=f"requests@{requests_version}",
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class IAMCredentialsRestInterceptor:
    """Interceptor for IAMCredentials.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the IAMCredentialsRestTransport.

    .. code-block:: python
        class MyCustomIAMCredentialsInterceptor(IAMCredentialsRestInterceptor):
            def pre_generate_access_token(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_generate_access_token(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_generate_id_token(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_generate_id_token(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_sign_blob(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_sign_blob(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_sign_jwt(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_sign_jwt(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = IAMCredentialsRestTransport(interceptor=MyCustomIAMCredentialsInterceptor())
        client = IAMCredentialsClient(transport=transport)


    """

    def pre_generate_access_token(
        self,
        request: common.GenerateAccessTokenRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        common.GenerateAccessTokenRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for generate_access_token

        Override in a subclass to manipulate the request or metadata
        before they are sent to the IAMCredentials server.
        """
        return request, metadata

    def post_generate_access_token(
        self, response: common.GenerateAccessTokenResponse
    ) -> common.GenerateAccessTokenResponse:
        """Post-rpc interceptor for generate_access_token

        DEPRECATED. Please use the `post_generate_access_token_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the IAMCredentials server but before
        it is returned to user code. This `post_generate_access_token` interceptor runs
        before the `post_generate_access_token_with_metadata` interceptor.
        """
        return response

    def post_generate_access_token_with_metadata(
        self,
        response: common.GenerateAccessTokenResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        common.GenerateAccessTokenResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for generate_access_token

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the IAMCredentials server but before it is returned to user code.

        We recommend only using this `post_generate_access_token_with_metadata`
        interceptor in new development instead of the `post_generate_access_token` interceptor.
        When both interceptors are used, this `post_generate_access_token_with_metadata` interceptor runs after the
        `post_generate_access_token` interceptor. The (possibly modified) response returned by
        `post_generate_access_token` will be passed to
        `post_generate_access_token_with_metadata`.
        """
        return response, metadata

    def pre_generate_id_token(
        self,
        request: common.GenerateIdTokenRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[common.GenerateIdTokenRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for generate_id_token

        Override in a subclass to manipulate the request or metadata
        before they are sent to the IAMCredentials server.
        """
        return request, metadata

    def post_generate_id_token(
        self, response: common.GenerateIdTokenResponse
    ) -> common.GenerateIdTokenResponse:
        """Post-rpc interceptor for generate_id_token

        DEPRECATED. Please use the `post_generate_id_token_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the IAMCredentials server but before
        it is returned to user code. This `post_generate_id_token` interceptor runs
        before the `post_generate_id_token_with_metadata` interceptor.
        """
        return response

    def post_generate_id_token_with_metadata(
        self,
        response: common.GenerateIdTokenResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[common.GenerateIdTokenResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for generate_id_token

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the IAMCredentials server but before it is returned to user code.

        We recommend only using this `post_generate_id_token_with_metadata`
        interceptor in new development instead of the `post_generate_id_token` interceptor.
        When both interceptors are used, this `post_generate_id_token_with_metadata` interceptor runs after the
        `post_generate_id_token` interceptor. The (possibly modified) response returned by
        `post_generate_id_token` will be passed to
        `post_generate_id_token_with_metadata`.
        """
        return response, metadata

    def pre_sign_blob(
        self,
        request: common.SignBlobRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[common.SignBlobRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for sign_blob

        Override in a subclass to manipulate the request or metadata
        before they are sent to the IAMCredentials server.
        """
        return request, metadata

    def post_sign_blob(
        self, response: common.SignBlobResponse
    ) -> common.SignBlobResponse:
        """Post-rpc interceptor for sign_blob

        DEPRECATED. Please use the `post_sign_blob_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the IAMCredentials server but before
        it is returned to user code. This `post_sign_blob` interceptor runs
        before the `post_sign_blob_with_metadata` interceptor.
        """
        return response

    def post_sign_blob_with_metadata(
        self,
        response: common.SignBlobResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[common.SignBlobResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for sign_blob

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the IAMCredentials server but before it is returned to user code.

        We recommend only using this `post_sign_blob_with_metadata`
        interceptor in new development instead of the `post_sign_blob` interceptor.
        When both interceptors are used, this `post_sign_blob_with_metadata` interceptor runs after the
        `post_sign_blob` interceptor. The (possibly modified) response returned by
        `post_sign_blob` will be passed to
        `post_sign_blob_with_metadata`.
        """
        return response, metadata

    def pre_sign_jwt(
        self,
        request: common.SignJwtRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[common.SignJwtRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for sign_jwt

        Override in a subclass to manipulate the request or metadata
        before they are sent to the IAMCredentials server.
        """
        return request, metadata

    def post_sign_jwt(self, response: common.SignJwtResponse) -> common.SignJwtResponse:
        """Post-rpc interceptor for sign_jwt

        DEPRECATED. Please use the `post_sign_jwt_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the IAMCredentials server but before
        it is returned to user code. This `post_sign_jwt` interceptor runs
        before the `post_sign_jwt_with_metadata` interceptor.
        """
        return response

    def post_sign_jwt_with_metadata(
        self,
        response: common.SignJwtResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[common.SignJwtResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for sign_jwt

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the IAMCredentials server but before it is returned to user code.

        We recommend only using this `post_sign_jwt_with_metadata`
        interceptor in new development instead of the `post_sign_jwt` interceptor.
        When both interceptors are used, this `post_sign_jwt_with_metadata` interceptor runs after the
        `post_sign_jwt` interceptor. The (possibly modified) response returned by
        `post_sign_jwt` will be passed to
        `post_sign_jwt_with_metadata`.
        """
        return response, metadata


@dataclasses.dataclass
class IAMCredentialsRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: IAMCredentialsRestInterceptor


class IAMCredentialsRestTransport(_BaseIAMCredentialsRestTransport):
    """REST backend synchronous transport for IAMCredentials.

    A service account is a special type of Google account that
    belongs to your application or a virtual machine (VM), instead
    of to an individual end user. Your application assumes the
    identity of the service account to call Google APIs, so that the
    users aren't directly involved.

    Service account credentials are used to temporarily assume the
    identity of the service account. Supported credential types
    include OAuth 2.0 access tokens, OpenID Connect ID tokens,
    self-signed JSON Web Tokens (JWTs), and more.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "iamcredentials.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[IAMCredentialsRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'iamcredentials.googleapis.com').
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
        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            url_scheme=url_scheme,
            api_audience=api_audience,
        )
        self._session = AuthorizedSession(
            self._credentials, default_host=self.DEFAULT_HOST
        )
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or IAMCredentialsRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _GenerateAccessToken(
        _BaseIAMCredentialsRestTransport._BaseGenerateAccessToken,
        IAMCredentialsRestStub,
    ):
        def __hash__(self):
            return hash("IAMCredentialsRestTransport.GenerateAccessToken")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: common.GenerateAccessTokenRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> common.GenerateAccessTokenResponse:
            r"""Call the generate access token method over HTTP.

            Args:
                request (~.common.GenerateAccessTokenRequest):
                    The request object.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.common.GenerateAccessTokenResponse:

            """

            http_options = (
                _BaseIAMCredentialsRestTransport._BaseGenerateAccessToken._get_http_options()
            )

            request, metadata = self._interceptor.pre_generate_access_token(
                request, metadata
            )
            transcoded_request = _BaseIAMCredentialsRestTransport._BaseGenerateAccessToken._get_transcoded_request(
                http_options, request
            )

            body = _BaseIAMCredentialsRestTransport._BaseGenerateAccessToken._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseIAMCredentialsRestTransport._BaseGenerateAccessToken._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.iam.credentials_v1.IAMCredentialsClient.GenerateAccessToken",
                    extra={
                        "serviceName": "google.iam.credentials.v1.IAMCredentials",
                        "rpcName": "GenerateAccessToken",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = IAMCredentialsRestTransport._GenerateAccessToken._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = common.GenerateAccessTokenResponse()
            pb_resp = common.GenerateAccessTokenResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_generate_access_token(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_generate_access_token_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = common.GenerateAccessTokenResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.iam.credentials_v1.IAMCredentialsClient.generate_access_token",
                    extra={
                        "serviceName": "google.iam.credentials.v1.IAMCredentials",
                        "rpcName": "GenerateAccessToken",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GenerateIdToken(
        _BaseIAMCredentialsRestTransport._BaseGenerateIdToken, IAMCredentialsRestStub
    ):
        def __hash__(self):
            return hash("IAMCredentialsRestTransport.GenerateIdToken")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: common.GenerateIdTokenRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> common.GenerateIdTokenResponse:
            r"""Call the generate id token method over HTTP.

            Args:
                request (~.common.GenerateIdTokenRequest):
                    The request object.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.common.GenerateIdTokenResponse:

            """

            http_options = (
                _BaseIAMCredentialsRestTransport._BaseGenerateIdToken._get_http_options()
            )

            request, metadata = self._interceptor.pre_generate_id_token(
                request, metadata
            )
            transcoded_request = _BaseIAMCredentialsRestTransport._BaseGenerateIdToken._get_transcoded_request(
                http_options, request
            )

            body = _BaseIAMCredentialsRestTransport._BaseGenerateIdToken._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseIAMCredentialsRestTransport._BaseGenerateIdToken._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.iam.credentials_v1.IAMCredentialsClient.GenerateIdToken",
                    extra={
                        "serviceName": "google.iam.credentials.v1.IAMCredentials",
                        "rpcName": "GenerateIdToken",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = IAMCredentialsRestTransport._GenerateIdToken._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = common.GenerateIdTokenResponse()
            pb_resp = common.GenerateIdTokenResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_generate_id_token(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_generate_id_token_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = common.GenerateIdTokenResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.iam.credentials_v1.IAMCredentialsClient.generate_id_token",
                    extra={
                        "serviceName": "google.iam.credentials.v1.IAMCredentials",
                        "rpcName": "GenerateIdToken",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _SignBlob(
        _BaseIAMCredentialsRestTransport._BaseSignBlob, IAMCredentialsRestStub
    ):
        def __hash__(self):
            return hash("IAMCredentialsRestTransport.SignBlob")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: common.SignBlobRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> common.SignBlobResponse:
            r"""Call the sign blob method over HTTP.

            Args:
                request (~.common.SignBlobRequest):
                    The request object.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.common.SignBlobResponse:

            """

            http_options = (
                _BaseIAMCredentialsRestTransport._BaseSignBlob._get_http_options()
            )

            request, metadata = self._interceptor.pre_sign_blob(request, metadata)
            transcoded_request = (
                _BaseIAMCredentialsRestTransport._BaseSignBlob._get_transcoded_request(
                    http_options, request
                )
            )

            body = (
                _BaseIAMCredentialsRestTransport._BaseSignBlob._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseIAMCredentialsRestTransport._BaseSignBlob._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.iam.credentials_v1.IAMCredentialsClient.SignBlob",
                    extra={
                        "serviceName": "google.iam.credentials.v1.IAMCredentials",
                        "rpcName": "SignBlob",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = IAMCredentialsRestTransport._SignBlob._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = common.SignBlobResponse()
            pb_resp = common.SignBlobResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_sign_blob(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_sign_blob_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = common.SignBlobResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.iam.credentials_v1.IAMCredentialsClient.sign_blob",
                    extra={
                        "serviceName": "google.iam.credentials.v1.IAMCredentials",
                        "rpcName": "SignBlob",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _SignJwt(
        _BaseIAMCredentialsRestTransport._BaseSignJwt, IAMCredentialsRestStub
    ):
        def __hash__(self):
            return hash("IAMCredentialsRestTransport.SignJwt")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: common.SignJwtRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> common.SignJwtResponse:
            r"""Call the sign jwt method over HTTP.

            Args:
                request (~.common.SignJwtRequest):
                    The request object.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.common.SignJwtResponse:

            """

            http_options = (
                _BaseIAMCredentialsRestTransport._BaseSignJwt._get_http_options()
            )

            request, metadata = self._interceptor.pre_sign_jwt(request, metadata)
            transcoded_request = (
                _BaseIAMCredentialsRestTransport._BaseSignJwt._get_transcoded_request(
                    http_options, request
                )
            )

            body = _BaseIAMCredentialsRestTransport._BaseSignJwt._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = (
                _BaseIAMCredentialsRestTransport._BaseSignJwt._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.iam.credentials_v1.IAMCredentialsClient.SignJwt",
                    extra={
                        "serviceName": "google.iam.credentials.v1.IAMCredentials",
                        "rpcName": "SignJwt",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = IAMCredentialsRestTransport._SignJwt._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = common.SignJwtResponse()
            pb_resp = common.SignJwtResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_sign_jwt(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_sign_jwt_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = common.SignJwtResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.iam.credentials_v1.IAMCredentialsClient.sign_jwt",
                    extra={
                        "serviceName": "google.iam.credentials.v1.IAMCredentials",
                        "rpcName": "SignJwt",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def generate_access_token(
        self,
    ) -> Callable[
        [common.GenerateAccessTokenRequest], common.GenerateAccessTokenResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GenerateAccessToken(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def generate_id_token(
        self,
    ) -> Callable[[common.GenerateIdTokenRequest], common.GenerateIdTokenResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GenerateIdToken(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def sign_blob(self) -> Callable[[common.SignBlobRequest], common.SignBlobResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SignBlob(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def sign_jwt(self) -> Callable[[common.SignJwtRequest], common.SignJwtResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SignJwt(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("IAMCredentialsRestTransport",)
