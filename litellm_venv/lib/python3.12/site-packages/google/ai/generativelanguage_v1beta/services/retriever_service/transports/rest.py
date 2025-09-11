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

from google.ai.generativelanguage_v1beta.types import retriever, retriever_service

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import RetrieverServiceTransport

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=requests_version,
)


class RetrieverServiceRestInterceptor:
    """Interceptor for RetrieverService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the RetrieverServiceRestTransport.

    .. code-block:: python
        class MyCustomRetrieverServiceInterceptor(RetrieverServiceRestInterceptor):
            def pre_batch_create_chunks(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_batch_create_chunks(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_batch_delete_chunks(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_batch_update_chunks(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_batch_update_chunks(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_chunk(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_chunk(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_corpus(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_corpus(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_document(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_chunk(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_corpus(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_chunk(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_chunk(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_corpus(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_corpus(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_document(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_chunks(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_chunks(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_corpora(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_corpora(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_documents(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_documents(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_query_corpus(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_query_corpus(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_query_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_query_document(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_chunk(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_chunk(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_corpus(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_corpus(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_document(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = RetrieverServiceRestTransport(interceptor=MyCustomRetrieverServiceInterceptor())
        client = RetrieverServiceClient(transport=transport)


    """

    def pre_batch_create_chunks(
        self,
        request: retriever_service.BatchCreateChunksRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.BatchCreateChunksRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for batch_create_chunks

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_batch_create_chunks(
        self, response: retriever_service.BatchCreateChunksResponse
    ) -> retriever_service.BatchCreateChunksResponse:
        """Post-rpc interceptor for batch_create_chunks

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_batch_delete_chunks(
        self,
        request: retriever_service.BatchDeleteChunksRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.BatchDeleteChunksRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for batch_delete_chunks

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def pre_batch_update_chunks(
        self,
        request: retriever_service.BatchUpdateChunksRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.BatchUpdateChunksRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for batch_update_chunks

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_batch_update_chunks(
        self, response: retriever_service.BatchUpdateChunksResponse
    ) -> retriever_service.BatchUpdateChunksResponse:
        """Post-rpc interceptor for batch_update_chunks

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_create_chunk(
        self,
        request: retriever_service.CreateChunkRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.CreateChunkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_chunk

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_create_chunk(self, response: retriever.Chunk) -> retriever.Chunk:
        """Post-rpc interceptor for create_chunk

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_create_corpus(
        self,
        request: retriever_service.CreateCorpusRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.CreateCorpusRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_corpus

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_create_corpus(self, response: retriever.Corpus) -> retriever.Corpus:
        """Post-rpc interceptor for create_corpus

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_create_document(
        self,
        request: retriever_service.CreateDocumentRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.CreateDocumentRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_create_document(self, response: retriever.Document) -> retriever.Document:
        """Post-rpc interceptor for create_document

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_delete_chunk(
        self,
        request: retriever_service.DeleteChunkRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.DeleteChunkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_chunk

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def pre_delete_corpus(
        self,
        request: retriever_service.DeleteCorpusRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.DeleteCorpusRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_corpus

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def pre_delete_document(
        self,
        request: retriever_service.DeleteDocumentRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.DeleteDocumentRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def pre_get_chunk(
        self,
        request: retriever_service.GetChunkRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.GetChunkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_chunk

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_get_chunk(self, response: retriever.Chunk) -> retriever.Chunk:
        """Post-rpc interceptor for get_chunk

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_get_corpus(
        self,
        request: retriever_service.GetCorpusRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.GetCorpusRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_corpus

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_get_corpus(self, response: retriever.Corpus) -> retriever.Corpus:
        """Post-rpc interceptor for get_corpus

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_get_document(
        self,
        request: retriever_service.GetDocumentRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.GetDocumentRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_get_document(self, response: retriever.Document) -> retriever.Document:
        """Post-rpc interceptor for get_document

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_list_chunks(
        self,
        request: retriever_service.ListChunksRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.ListChunksRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_chunks

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_list_chunks(
        self, response: retriever_service.ListChunksResponse
    ) -> retriever_service.ListChunksResponse:
        """Post-rpc interceptor for list_chunks

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_list_corpora(
        self,
        request: retriever_service.ListCorporaRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.ListCorporaRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_corpora

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_list_corpora(
        self, response: retriever_service.ListCorporaResponse
    ) -> retriever_service.ListCorporaResponse:
        """Post-rpc interceptor for list_corpora

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_list_documents(
        self,
        request: retriever_service.ListDocumentsRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.ListDocumentsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_documents

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_list_documents(
        self, response: retriever_service.ListDocumentsResponse
    ) -> retriever_service.ListDocumentsResponse:
        """Post-rpc interceptor for list_documents

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_query_corpus(
        self,
        request: retriever_service.QueryCorpusRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.QueryCorpusRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for query_corpus

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_query_corpus(
        self, response: retriever_service.QueryCorpusResponse
    ) -> retriever_service.QueryCorpusResponse:
        """Post-rpc interceptor for query_corpus

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_query_document(
        self,
        request: retriever_service.QueryDocumentRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.QueryDocumentRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for query_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_query_document(
        self, response: retriever_service.QueryDocumentResponse
    ) -> retriever_service.QueryDocumentResponse:
        """Post-rpc interceptor for query_document

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_update_chunk(
        self,
        request: retriever_service.UpdateChunkRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.UpdateChunkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_chunk

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_update_chunk(self, response: retriever.Chunk) -> retriever.Chunk:
        """Post-rpc interceptor for update_chunk

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_update_corpus(
        self,
        request: retriever_service.UpdateCorpusRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.UpdateCorpusRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_corpus

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_update_corpus(self, response: retriever.Corpus) -> retriever.Corpus:
        """Post-rpc interceptor for update_corpus

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response

    def pre_update_document(
        self,
        request: retriever_service.UpdateDocumentRequest,
        metadata: Sequence[Tuple[str, str]],
    ) -> Tuple[retriever_service.UpdateDocumentRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the RetrieverService server.
        """
        return request, metadata

    def post_update_document(self, response: retriever.Document) -> retriever.Document:
        """Post-rpc interceptor for update_document

        Override in a subclass to manipulate the response
        after it is returned by the RetrieverService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class RetrieverServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: RetrieverServiceRestInterceptor


class RetrieverServiceRestTransport(RetrieverServiceTransport):
    """REST backend transport for RetrieverService.

    An API for semantic search over a corpus of user uploaded
    content.

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
        interceptor: Optional[RetrieverServiceRestInterceptor] = None,
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
        self._interceptor = interceptor or RetrieverServiceRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _BatchCreateChunks(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("BatchCreateChunks")

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
            request: retriever_service.BatchCreateChunksRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever_service.BatchCreateChunksResponse:
            r"""Call the batch create chunks method over HTTP.

            Args:
                request (~.retriever_service.BatchCreateChunksRequest):
                    The request object. Request to batch create ``Chunk``\ s.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever_service.BatchCreateChunksResponse:
                    Response from ``BatchCreateChunks`` containing a list of
                created ``Chunk``\ s.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta/{parent=corpora/*/documents/*}/chunks:batchCreate",
                    "body": "*",
                },
            ]
            request, metadata = self._interceptor.pre_batch_create_chunks(
                request, metadata
            )
            pb_request = retriever_service.BatchCreateChunksRequest.pb(request)
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
            resp = retriever_service.BatchCreateChunksResponse()
            pb_resp = retriever_service.BatchCreateChunksResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_batch_create_chunks(resp)
            return resp

    class _BatchDeleteChunks(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("BatchDeleteChunks")

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
            request: retriever_service.BatchDeleteChunksRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ):
            r"""Call the batch delete chunks method over HTTP.

            Args:
                request (~.retriever_service.BatchDeleteChunksRequest):
                    The request object. Request to batch delete ``Chunk``\ s.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta/{parent=corpora/*/documents/*}/chunks:batchDelete",
                    "body": "*",
                },
            ]
            request, metadata = self._interceptor.pre_batch_delete_chunks(
                request, metadata
            )
            pb_request = retriever_service.BatchDeleteChunksRequest.pb(request)
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

    class _BatchUpdateChunks(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("BatchUpdateChunks")

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
            request: retriever_service.BatchUpdateChunksRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever_service.BatchUpdateChunksResponse:
            r"""Call the batch update chunks method over HTTP.

            Args:
                request (~.retriever_service.BatchUpdateChunksRequest):
                    The request object. Request to batch update ``Chunk``\ s.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever_service.BatchUpdateChunksResponse:
                    Response from ``BatchUpdateChunks`` containing a list of
                updated ``Chunk``\ s.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta/{parent=corpora/*/documents/*}/chunks:batchUpdate",
                    "body": "*",
                },
            ]
            request, metadata = self._interceptor.pre_batch_update_chunks(
                request, metadata
            )
            pb_request = retriever_service.BatchUpdateChunksRequest.pb(request)
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
            resp = retriever_service.BatchUpdateChunksResponse()
            pb_resp = retriever_service.BatchUpdateChunksResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_batch_update_chunks(resp)
            return resp

    class _CreateChunk(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("CreateChunk")

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
            request: retriever_service.CreateChunkRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Chunk:
            r"""Call the create chunk method over HTTP.

            Args:
                request (~.retriever_service.CreateChunkRequest):
                    The request object. Request to create a ``Chunk``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Chunk:
                    A ``Chunk`` is a subpart of a ``Document`` that is
                treated as an independent unit for the purposes of
                vector representation and storage. A ``Corpus`` can have
                a maximum of 1 million ``Chunk``\ s.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta/{parent=corpora/*/documents/*}/chunks",
                    "body": "chunk",
                },
            ]
            request, metadata = self._interceptor.pre_create_chunk(request, metadata)
            pb_request = retriever_service.CreateChunkRequest.pb(request)
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
            resp = retriever.Chunk()
            pb_resp = retriever.Chunk.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_chunk(resp)
            return resp

    class _CreateCorpus(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("CreateCorpus")

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
            request: retriever_service.CreateCorpusRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Corpus:
            r"""Call the create corpus method over HTTP.

            Args:
                request (~.retriever_service.CreateCorpusRequest):
                    The request object. Request to create a ``Corpus``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Corpus:
                    A ``Corpus`` is a collection of ``Document``\ s. A
                project can create up to 5 corpora.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta/corpora",
                    "body": "corpus",
                },
            ]
            request, metadata = self._interceptor.pre_create_corpus(request, metadata)
            pb_request = retriever_service.CreateCorpusRequest.pb(request)
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
            resp = retriever.Corpus()
            pb_resp = retriever.Corpus.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_corpus(resp)
            return resp

    class _CreateDocument(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("CreateDocument")

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
            request: retriever_service.CreateDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Document:
            r"""Call the create document method over HTTP.

            Args:
                request (~.retriever_service.CreateDocumentRequest):
                    The request object. Request to create a ``Document``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Document:
                    A ``Document`` is a collection of ``Chunk``\ s. A
                ``Corpus`` can have a maximum of 10,000 ``Document``\ s.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta/{parent=corpora/*}/documents",
                    "body": "document",
                },
            ]
            request, metadata = self._interceptor.pre_create_document(request, metadata)
            pb_request = retriever_service.CreateDocumentRequest.pb(request)
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
            resp = retriever.Document()
            pb_resp = retriever.Document.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_document(resp)
            return resp

    class _DeleteChunk(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("DeleteChunk")

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
            request: retriever_service.DeleteChunkRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ):
            r"""Call the delete chunk method over HTTP.

            Args:
                request (~.retriever_service.DeleteChunkRequest):
                    The request object. Request to delete a ``Chunk``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "delete",
                    "uri": "/v1beta/{name=corpora/*/documents/*/chunks/*}",
                },
            ]
            request, metadata = self._interceptor.pre_delete_chunk(request, metadata)
            pb_request = retriever_service.DeleteChunkRequest.pb(request)
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

    class _DeleteCorpus(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("DeleteCorpus")

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
            request: retriever_service.DeleteCorpusRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ):
            r"""Call the delete corpus method over HTTP.

            Args:
                request (~.retriever_service.DeleteCorpusRequest):
                    The request object. Request to delete a ``Corpus``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "delete",
                    "uri": "/v1beta/{name=corpora/*}",
                },
            ]
            request, metadata = self._interceptor.pre_delete_corpus(request, metadata)
            pb_request = retriever_service.DeleteCorpusRequest.pb(request)
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

    class _DeleteDocument(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("DeleteDocument")

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
            request: retriever_service.DeleteDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ):
            r"""Call the delete document method over HTTP.

            Args:
                request (~.retriever_service.DeleteDocumentRequest):
                    The request object. Request to delete a ``Document``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "delete",
                    "uri": "/v1beta/{name=corpora/*/documents/*}",
                },
            ]
            request, metadata = self._interceptor.pre_delete_document(request, metadata)
            pb_request = retriever_service.DeleteDocumentRequest.pb(request)
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

    class _GetChunk(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("GetChunk")

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
            request: retriever_service.GetChunkRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Chunk:
            r"""Call the get chunk method over HTTP.

            Args:
                request (~.retriever_service.GetChunkRequest):
                    The request object. Request for getting information about a specific
                ``Chunk``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Chunk:
                    A ``Chunk`` is a subpart of a ``Document`` that is
                treated as an independent unit for the purposes of
                vector representation and storage. A ``Corpus`` can have
                a maximum of 1 million ``Chunk``\ s.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "get",
                    "uri": "/v1beta/{name=corpora/*/documents/*/chunks/*}",
                },
            ]
            request, metadata = self._interceptor.pre_get_chunk(request, metadata)
            pb_request = retriever_service.GetChunkRequest.pb(request)
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
            resp = retriever.Chunk()
            pb_resp = retriever.Chunk.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_chunk(resp)
            return resp

    class _GetCorpus(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("GetCorpus")

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
            request: retriever_service.GetCorpusRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Corpus:
            r"""Call the get corpus method over HTTP.

            Args:
                request (~.retriever_service.GetCorpusRequest):
                    The request object. Request for getting information about a specific
                ``Corpus``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Corpus:
                    A ``Corpus`` is a collection of ``Document``\ s. A
                project can create up to 5 corpora.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "get",
                    "uri": "/v1beta/{name=corpora/*}",
                },
            ]
            request, metadata = self._interceptor.pre_get_corpus(request, metadata)
            pb_request = retriever_service.GetCorpusRequest.pb(request)
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
            resp = retriever.Corpus()
            pb_resp = retriever.Corpus.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_corpus(resp)
            return resp

    class _GetDocument(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("GetDocument")

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
            request: retriever_service.GetDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Document:
            r"""Call the get document method over HTTP.

            Args:
                request (~.retriever_service.GetDocumentRequest):
                    The request object. Request for getting information about a specific
                ``Document``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Document:
                    A ``Document`` is a collection of ``Chunk``\ s. A
                ``Corpus`` can have a maximum of 10,000 ``Document``\ s.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "get",
                    "uri": "/v1beta/{name=corpora/*/documents/*}",
                },
            ]
            request, metadata = self._interceptor.pre_get_document(request, metadata)
            pb_request = retriever_service.GetDocumentRequest.pb(request)
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
            resp = retriever.Document()
            pb_resp = retriever.Document.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_document(resp)
            return resp

    class _ListChunks(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("ListChunks")

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
            request: retriever_service.ListChunksRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever_service.ListChunksResponse:
            r"""Call the list chunks method over HTTP.

            Args:
                request (~.retriever_service.ListChunksRequest):
                    The request object. Request for listing ``Chunk``\ s.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever_service.ListChunksResponse:
                    Response from ``ListChunks`` containing a paginated list
                of ``Chunk``\ s. The ``Chunk``\ s are sorted by
                ascending ``chunk.create_time``.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "get",
                    "uri": "/v1beta/{parent=corpora/*/documents/*}/chunks",
                },
            ]
            request, metadata = self._interceptor.pre_list_chunks(request, metadata)
            pb_request = retriever_service.ListChunksRequest.pb(request)
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
            resp = retriever_service.ListChunksResponse()
            pb_resp = retriever_service.ListChunksResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_chunks(resp)
            return resp

    class _ListCorpora(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("ListCorpora")

        def __call__(
            self,
            request: retriever_service.ListCorporaRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever_service.ListCorporaResponse:
            r"""Call the list corpora method over HTTP.

            Args:
                request (~.retriever_service.ListCorporaRequest):
                    The request object. Request for listing ``Corpora``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever_service.ListCorporaResponse:
                    Response from ``ListCorpora`` containing a paginated
                list of ``Corpora``. The results are sorted by ascending
                ``corpus.create_time``.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "get",
                    "uri": "/v1beta/corpora",
                },
            ]
            request, metadata = self._interceptor.pre_list_corpora(request, metadata)
            pb_request = retriever_service.ListCorporaRequest.pb(request)
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
            resp = retriever_service.ListCorporaResponse()
            pb_resp = retriever_service.ListCorporaResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_corpora(resp)
            return resp

    class _ListDocuments(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("ListDocuments")

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
            request: retriever_service.ListDocumentsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever_service.ListDocumentsResponse:
            r"""Call the list documents method over HTTP.

            Args:
                request (~.retriever_service.ListDocumentsRequest):
                    The request object. Request for listing ``Document``\ s.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever_service.ListDocumentsResponse:
                    Response from ``ListDocuments`` containing a paginated
                list of ``Document``\ s. The ``Document``\ s are sorted
                by ascending ``document.create_time``.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "get",
                    "uri": "/v1beta/{parent=corpora/*}/documents",
                },
            ]
            request, metadata = self._interceptor.pre_list_documents(request, metadata)
            pb_request = retriever_service.ListDocumentsRequest.pb(request)
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
            resp = retriever_service.ListDocumentsResponse()
            pb_resp = retriever_service.ListDocumentsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_documents(resp)
            return resp

    class _QueryCorpus(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("QueryCorpus")

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
            request: retriever_service.QueryCorpusRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever_service.QueryCorpusResponse:
            r"""Call the query corpus method over HTTP.

            Args:
                request (~.retriever_service.QueryCorpusRequest):
                    The request object. Request for querying a ``Corpus``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever_service.QueryCorpusResponse:
                    Response from ``QueryCorpus`` containing a list of
                relevant chunks.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta/{name=corpora/*}:query",
                    "body": "*",
                },
            ]
            request, metadata = self._interceptor.pre_query_corpus(request, metadata)
            pb_request = retriever_service.QueryCorpusRequest.pb(request)
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
            resp = retriever_service.QueryCorpusResponse()
            pb_resp = retriever_service.QueryCorpusResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_query_corpus(resp)
            return resp

    class _QueryDocument(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("QueryDocument")

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
            request: retriever_service.QueryDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever_service.QueryDocumentResponse:
            r"""Call the query document method over HTTP.

            Args:
                request (~.retriever_service.QueryDocumentRequest):
                    The request object. Request for querying a ``Document``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever_service.QueryDocumentResponse:
                    Response from ``QueryDocument`` containing a list of
                relevant chunks.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "post",
                    "uri": "/v1beta/{name=corpora/*/documents/*}:query",
                    "body": "*",
                },
            ]
            request, metadata = self._interceptor.pre_query_document(request, metadata)
            pb_request = retriever_service.QueryDocumentRequest.pb(request)
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
            resp = retriever_service.QueryDocumentResponse()
            pb_resp = retriever_service.QueryDocumentResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_query_document(resp)
            return resp

    class _UpdateChunk(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("UpdateChunk")

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
            request: retriever_service.UpdateChunkRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Chunk:
            r"""Call the update chunk method over HTTP.

            Args:
                request (~.retriever_service.UpdateChunkRequest):
                    The request object. Request to update a ``Chunk``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Chunk:
                    A ``Chunk`` is a subpart of a ``Document`` that is
                treated as an independent unit for the purposes of
                vector representation and storage. A ``Corpus`` can have
                a maximum of 1 million ``Chunk``\ s.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "patch",
                    "uri": "/v1beta/{chunk.name=corpora/*/documents/*/chunks/*}",
                    "body": "chunk",
                },
            ]
            request, metadata = self._interceptor.pre_update_chunk(request, metadata)
            pb_request = retriever_service.UpdateChunkRequest.pb(request)
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
            resp = retriever.Chunk()
            pb_resp = retriever.Chunk.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_chunk(resp)
            return resp

    class _UpdateCorpus(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("UpdateCorpus")

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
            request: retriever_service.UpdateCorpusRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Corpus:
            r"""Call the update corpus method over HTTP.

            Args:
                request (~.retriever_service.UpdateCorpusRequest):
                    The request object. Request to update a ``Corpus``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Corpus:
                    A ``Corpus`` is a collection of ``Document``\ s. A
                project can create up to 5 corpora.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "patch",
                    "uri": "/v1beta/{corpus.name=corpora/*}",
                    "body": "corpus",
                },
            ]
            request, metadata = self._interceptor.pre_update_corpus(request, metadata)
            pb_request = retriever_service.UpdateCorpusRequest.pb(request)
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
            resp = retriever.Corpus()
            pb_resp = retriever.Corpus.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_corpus(resp)
            return resp

    class _UpdateDocument(RetrieverServiceRestStub):
        def __hash__(self):
            return hash("UpdateDocument")

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
            request: retriever_service.UpdateDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, str]] = (),
        ) -> retriever.Document:
            r"""Call the update document method over HTTP.

            Args:
                request (~.retriever_service.UpdateDocumentRequest):
                    The request object. Request to update a ``Document``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.retriever.Document:
                    A ``Document`` is a collection of ``Chunk``\ s. A
                ``Corpus`` can have a maximum of 10,000 ``Document``\ s.

            """

            http_options: List[Dict[str, str]] = [
                {
                    "method": "patch",
                    "uri": "/v1beta/{document.name=corpora/*/documents/*}",
                    "body": "document",
                },
            ]
            request, metadata = self._interceptor.pre_update_document(request, metadata)
            pb_request = retriever_service.UpdateDocumentRequest.pb(request)
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
            resp = retriever.Document()
            pb_resp = retriever.Document.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_document(resp)
            return resp

    @property
    def batch_create_chunks(
        self,
    ) -> Callable[
        [retriever_service.BatchCreateChunksRequest],
        retriever_service.BatchCreateChunksResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._BatchCreateChunks(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def batch_delete_chunks(
        self,
    ) -> Callable[[retriever_service.BatchDeleteChunksRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._BatchDeleteChunks(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def batch_update_chunks(
        self,
    ) -> Callable[
        [retriever_service.BatchUpdateChunksRequest],
        retriever_service.BatchUpdateChunksResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._BatchUpdateChunks(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_chunk(
        self,
    ) -> Callable[[retriever_service.CreateChunkRequest], retriever.Chunk]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateChunk(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_corpus(
        self,
    ) -> Callable[[retriever_service.CreateCorpusRequest], retriever.Corpus]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateCorpus(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_document(
        self,
    ) -> Callable[[retriever_service.CreateDocumentRequest], retriever.Document]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_chunk(
        self,
    ) -> Callable[[retriever_service.DeleteChunkRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteChunk(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_corpus(
        self,
    ) -> Callable[[retriever_service.DeleteCorpusRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteCorpus(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_document(
        self,
    ) -> Callable[[retriever_service.DeleteDocumentRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_chunk(
        self,
    ) -> Callable[[retriever_service.GetChunkRequest], retriever.Chunk]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetChunk(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_corpus(
        self,
    ) -> Callable[[retriever_service.GetCorpusRequest], retriever.Corpus]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetCorpus(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_document(
        self,
    ) -> Callable[[retriever_service.GetDocumentRequest], retriever.Document]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_chunks(
        self,
    ) -> Callable[
        [retriever_service.ListChunksRequest], retriever_service.ListChunksResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListChunks(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_corpora(
        self,
    ) -> Callable[
        [retriever_service.ListCorporaRequest], retriever_service.ListCorporaResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListCorpora(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_documents(
        self,
    ) -> Callable[
        [retriever_service.ListDocumentsRequest],
        retriever_service.ListDocumentsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListDocuments(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def query_corpus(
        self,
    ) -> Callable[
        [retriever_service.QueryCorpusRequest], retriever_service.QueryCorpusResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._QueryCorpus(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def query_document(
        self,
    ) -> Callable[
        [retriever_service.QueryDocumentRequest],
        retriever_service.QueryDocumentResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._QueryDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_chunk(
        self,
    ) -> Callable[[retriever_service.UpdateChunkRequest], retriever.Chunk]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateChunk(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_corpus(
        self,
    ) -> Callable[[retriever_service.UpdateCorpusRequest], retriever.Corpus]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateCorpus(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_document(
        self,
    ) -> Callable[[retriever_service.UpdateDocumentRequest], retriever.Document]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("RetrieverServiceRestTransport",)
