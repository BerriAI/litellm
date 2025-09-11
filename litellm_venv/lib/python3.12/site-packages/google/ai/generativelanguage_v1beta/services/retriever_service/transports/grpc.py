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
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
import warnings

from google.api_core import gapic_v1, grpc_helpers
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
import grpc  # type: ignore

from google.ai.generativelanguage_v1beta.types import retriever, retriever_service

from .base import DEFAULT_CLIENT_INFO, RetrieverServiceTransport


class RetrieverServiceGrpcTransport(RetrieverServiceTransport):
    """gRPC backend transport for RetrieverService.

    An API for semantic search over a corpus of user uploaded
    content.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends protocol buffers over the wire using gRPC (which is built on
    top of HTTP/2); the ``grpcio`` package must be installed.
    """

    _stubs: Dict[str, Callable]

    def __init__(
        self,
        *,
        host: str = "generativelanguage.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        channel: Optional[grpc.Channel] = None,
        api_mtls_endpoint: Optional[str] = None,
        client_cert_source: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        ssl_channel_credentials: Optional[grpc.ChannelCredentials] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
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
                This argument is ignored if ``channel`` is provided.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            channel (Optional[grpc.Channel]): A ``Channel`` instance through
                which to make calls.
            api_mtls_endpoint (Optional[str]): Deprecated. The mutual TLS endpoint.
                If provided, it overrides the ``host`` argument and tries to create
                a mutual TLS channel with client SSL credentials from
                ``client_cert_source`` or application default SSL credentials.
            client_cert_source (Optional[Callable[[], Tuple[bytes, bytes]]]):
                Deprecated. A callback to provide client SSL certificate bytes and
                private key bytes, both in PEM format. It is ignored if
                ``api_mtls_endpoint`` is None.
            ssl_channel_credentials (grpc.ChannelCredentials): SSL credentials
                for the grpc channel. It is ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Optional[Callable[[], Tuple[bytes, bytes]]]):
                A callback to provide client certificate bytes and private key bytes,
                both in PEM format. It is used to configure a mutual TLS channel. It is
                ignored if ``channel`` or ``ssl_channel_credentials`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.

        Raises:
          google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
              creation failed for any reason.
          google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """
        self._grpc_channel = None
        self._ssl_channel_credentials = ssl_channel_credentials
        self._stubs: Dict[str, Callable] = {}

        if api_mtls_endpoint:
            warnings.warn("api_mtls_endpoint is deprecated", DeprecationWarning)
        if client_cert_source:
            warnings.warn("client_cert_source is deprecated", DeprecationWarning)

        if channel:
            # Ignore credentials if a channel was passed.
            credentials = False
            # If a channel was explicitly provided, set it.
            self._grpc_channel = channel
            self._ssl_channel_credentials = None

        else:
            if api_mtls_endpoint:
                host = api_mtls_endpoint

                # Create SSL credentials with client_cert_source or application
                # default SSL credentials.
                if client_cert_source:
                    cert, key = client_cert_source()
                    self._ssl_channel_credentials = grpc.ssl_channel_credentials(
                        certificate_chain=cert, private_key=key
                    )
                else:
                    self._ssl_channel_credentials = SslCredentials().ssl_credentials

            else:
                if client_cert_source_for_mtls and not ssl_channel_credentials:
                    cert, key = client_cert_source_for_mtls()
                    self._ssl_channel_credentials = grpc.ssl_channel_credentials(
                        certificate_chain=cert, private_key=key
                    )

        # The base transport sets the host, credentials and scopes
        super().__init__(
            host=host,
            credentials=credentials,
            credentials_file=credentials_file,
            scopes=scopes,
            quota_project_id=quota_project_id,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            api_audience=api_audience,
        )

        if not self._grpc_channel:
            self._grpc_channel = type(self).create_channel(
                self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                # Set ``credentials_file`` to ``None`` here as
                # the credentials that we saved earlier should be used.
                credentials_file=None,
                scopes=self._scopes,
                ssl_credentials=self._ssl_channel_credentials,
                quota_project_id=quota_project_id,
                options=[
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                ],
            )

        # Wrap messages. This must be done after self._grpc_channel exists
        self._prep_wrapped_messages(client_info)

    @classmethod
    def create_channel(
        cls,
        host: str = "generativelanguage.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        **kwargs,
    ) -> grpc.Channel:
        """Create and return a gRPC channel object.
        Args:
            host (Optional[str]): The host for the channel to use.
            credentials (Optional[~.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If
                none are specified, the client will attempt to ascertain
                the credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            kwargs (Optional[dict]): Keyword arguments, which are passed to the
                channel creation.
        Returns:
            grpc.Channel: A gRPC channel object.

        Raises:
            google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """

        return grpc_helpers.create_channel(
            host,
            credentials=credentials,
            credentials_file=credentials_file,
            quota_project_id=quota_project_id,
            default_scopes=cls.AUTH_SCOPES,
            scopes=scopes,
            default_host=cls.DEFAULT_HOST,
            **kwargs,
        )

    @property
    def grpc_channel(self) -> grpc.Channel:
        """Return the channel designed to connect to this service."""
        return self._grpc_channel

    @property
    def create_corpus(
        self,
    ) -> Callable[[retriever_service.CreateCorpusRequest], retriever.Corpus]:
        r"""Return a callable for the create corpus method over gRPC.

        Creates an empty ``Corpus``.

        Returns:
            Callable[[~.CreateCorpusRequest],
                    ~.Corpus]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_corpus" not in self._stubs:
            self._stubs["create_corpus"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/CreateCorpus",
                request_serializer=retriever_service.CreateCorpusRequest.serialize,
                response_deserializer=retriever.Corpus.deserialize,
            )
        return self._stubs["create_corpus"]

    @property
    def get_corpus(
        self,
    ) -> Callable[[retriever_service.GetCorpusRequest], retriever.Corpus]:
        r"""Return a callable for the get corpus method over gRPC.

        Gets information about a specific ``Corpus``.

        Returns:
            Callable[[~.GetCorpusRequest],
                    ~.Corpus]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_corpus" not in self._stubs:
            self._stubs["get_corpus"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/GetCorpus",
                request_serializer=retriever_service.GetCorpusRequest.serialize,
                response_deserializer=retriever.Corpus.deserialize,
            )
        return self._stubs["get_corpus"]

    @property
    def update_corpus(
        self,
    ) -> Callable[[retriever_service.UpdateCorpusRequest], retriever.Corpus]:
        r"""Return a callable for the update corpus method over gRPC.

        Updates a ``Corpus``.

        Returns:
            Callable[[~.UpdateCorpusRequest],
                    ~.Corpus]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_corpus" not in self._stubs:
            self._stubs["update_corpus"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/UpdateCorpus",
                request_serializer=retriever_service.UpdateCorpusRequest.serialize,
                response_deserializer=retriever.Corpus.deserialize,
            )
        return self._stubs["update_corpus"]

    @property
    def delete_corpus(
        self,
    ) -> Callable[[retriever_service.DeleteCorpusRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete corpus method over gRPC.

        Deletes a ``Corpus``.

        Returns:
            Callable[[~.DeleteCorpusRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_corpus" not in self._stubs:
            self._stubs["delete_corpus"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/DeleteCorpus",
                request_serializer=retriever_service.DeleteCorpusRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_corpus"]

    @property
    def list_corpora(
        self,
    ) -> Callable[
        [retriever_service.ListCorporaRequest], retriever_service.ListCorporaResponse
    ]:
        r"""Return a callable for the list corpora method over gRPC.

        Lists all ``Corpora`` owned by the user.

        Returns:
            Callable[[~.ListCorporaRequest],
                    ~.ListCorporaResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_corpora" not in self._stubs:
            self._stubs["list_corpora"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/ListCorpora",
                request_serializer=retriever_service.ListCorporaRequest.serialize,
                response_deserializer=retriever_service.ListCorporaResponse.deserialize,
            )
        return self._stubs["list_corpora"]

    @property
    def query_corpus(
        self,
    ) -> Callable[
        [retriever_service.QueryCorpusRequest], retriever_service.QueryCorpusResponse
    ]:
        r"""Return a callable for the query corpus method over gRPC.

        Performs semantic search over a ``Corpus``.

        Returns:
            Callable[[~.QueryCorpusRequest],
                    ~.QueryCorpusResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "query_corpus" not in self._stubs:
            self._stubs["query_corpus"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/QueryCorpus",
                request_serializer=retriever_service.QueryCorpusRequest.serialize,
                response_deserializer=retriever_service.QueryCorpusResponse.deserialize,
            )
        return self._stubs["query_corpus"]

    @property
    def create_document(
        self,
    ) -> Callable[[retriever_service.CreateDocumentRequest], retriever.Document]:
        r"""Return a callable for the create document method over gRPC.

        Creates an empty ``Document``.

        Returns:
            Callable[[~.CreateDocumentRequest],
                    ~.Document]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_document" not in self._stubs:
            self._stubs["create_document"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/CreateDocument",
                request_serializer=retriever_service.CreateDocumentRequest.serialize,
                response_deserializer=retriever.Document.deserialize,
            )
        return self._stubs["create_document"]

    @property
    def get_document(
        self,
    ) -> Callable[[retriever_service.GetDocumentRequest], retriever.Document]:
        r"""Return a callable for the get document method over gRPC.

        Gets information about a specific ``Document``.

        Returns:
            Callable[[~.GetDocumentRequest],
                    ~.Document]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_document" not in self._stubs:
            self._stubs["get_document"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/GetDocument",
                request_serializer=retriever_service.GetDocumentRequest.serialize,
                response_deserializer=retriever.Document.deserialize,
            )
        return self._stubs["get_document"]

    @property
    def update_document(
        self,
    ) -> Callable[[retriever_service.UpdateDocumentRequest], retriever.Document]:
        r"""Return a callable for the update document method over gRPC.

        Updates a ``Document``.

        Returns:
            Callable[[~.UpdateDocumentRequest],
                    ~.Document]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_document" not in self._stubs:
            self._stubs["update_document"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/UpdateDocument",
                request_serializer=retriever_service.UpdateDocumentRequest.serialize,
                response_deserializer=retriever.Document.deserialize,
            )
        return self._stubs["update_document"]

    @property
    def delete_document(
        self,
    ) -> Callable[[retriever_service.DeleteDocumentRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete document method over gRPC.

        Deletes a ``Document``.

        Returns:
            Callable[[~.DeleteDocumentRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_document" not in self._stubs:
            self._stubs["delete_document"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/DeleteDocument",
                request_serializer=retriever_service.DeleteDocumentRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_document"]

    @property
    def list_documents(
        self,
    ) -> Callable[
        [retriever_service.ListDocumentsRequest],
        retriever_service.ListDocumentsResponse,
    ]:
        r"""Return a callable for the list documents method over gRPC.

        Lists all ``Document``\ s in a ``Corpus``.

        Returns:
            Callable[[~.ListDocumentsRequest],
                    ~.ListDocumentsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_documents" not in self._stubs:
            self._stubs["list_documents"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/ListDocuments",
                request_serializer=retriever_service.ListDocumentsRequest.serialize,
                response_deserializer=retriever_service.ListDocumentsResponse.deserialize,
            )
        return self._stubs["list_documents"]

    @property
    def query_document(
        self,
    ) -> Callable[
        [retriever_service.QueryDocumentRequest],
        retriever_service.QueryDocumentResponse,
    ]:
        r"""Return a callable for the query document method over gRPC.

        Performs semantic search over a ``Document``.

        Returns:
            Callable[[~.QueryDocumentRequest],
                    ~.QueryDocumentResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "query_document" not in self._stubs:
            self._stubs["query_document"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/QueryDocument",
                request_serializer=retriever_service.QueryDocumentRequest.serialize,
                response_deserializer=retriever_service.QueryDocumentResponse.deserialize,
            )
        return self._stubs["query_document"]

    @property
    def create_chunk(
        self,
    ) -> Callable[[retriever_service.CreateChunkRequest], retriever.Chunk]:
        r"""Return a callable for the create chunk method over gRPC.

        Creates a ``Chunk``.

        Returns:
            Callable[[~.CreateChunkRequest],
                    ~.Chunk]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_chunk" not in self._stubs:
            self._stubs["create_chunk"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/CreateChunk",
                request_serializer=retriever_service.CreateChunkRequest.serialize,
                response_deserializer=retriever.Chunk.deserialize,
            )
        return self._stubs["create_chunk"]

    @property
    def batch_create_chunks(
        self,
    ) -> Callable[
        [retriever_service.BatchCreateChunksRequest],
        retriever_service.BatchCreateChunksResponse,
    ]:
        r"""Return a callable for the batch create chunks method over gRPC.

        Batch create ``Chunk``\ s.

        Returns:
            Callable[[~.BatchCreateChunksRequest],
                    ~.BatchCreateChunksResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_create_chunks" not in self._stubs:
            self._stubs["batch_create_chunks"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/BatchCreateChunks",
                request_serializer=retriever_service.BatchCreateChunksRequest.serialize,
                response_deserializer=retriever_service.BatchCreateChunksResponse.deserialize,
            )
        return self._stubs["batch_create_chunks"]

    @property
    def get_chunk(
        self,
    ) -> Callable[[retriever_service.GetChunkRequest], retriever.Chunk]:
        r"""Return a callable for the get chunk method over gRPC.

        Gets information about a specific ``Chunk``.

        Returns:
            Callable[[~.GetChunkRequest],
                    ~.Chunk]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_chunk" not in self._stubs:
            self._stubs["get_chunk"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/GetChunk",
                request_serializer=retriever_service.GetChunkRequest.serialize,
                response_deserializer=retriever.Chunk.deserialize,
            )
        return self._stubs["get_chunk"]

    @property
    def update_chunk(
        self,
    ) -> Callable[[retriever_service.UpdateChunkRequest], retriever.Chunk]:
        r"""Return a callable for the update chunk method over gRPC.

        Updates a ``Chunk``.

        Returns:
            Callable[[~.UpdateChunkRequest],
                    ~.Chunk]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_chunk" not in self._stubs:
            self._stubs["update_chunk"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/UpdateChunk",
                request_serializer=retriever_service.UpdateChunkRequest.serialize,
                response_deserializer=retriever.Chunk.deserialize,
            )
        return self._stubs["update_chunk"]

    @property
    def batch_update_chunks(
        self,
    ) -> Callable[
        [retriever_service.BatchUpdateChunksRequest],
        retriever_service.BatchUpdateChunksResponse,
    ]:
        r"""Return a callable for the batch update chunks method over gRPC.

        Batch update ``Chunk``\ s.

        Returns:
            Callable[[~.BatchUpdateChunksRequest],
                    ~.BatchUpdateChunksResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_update_chunks" not in self._stubs:
            self._stubs["batch_update_chunks"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/BatchUpdateChunks",
                request_serializer=retriever_service.BatchUpdateChunksRequest.serialize,
                response_deserializer=retriever_service.BatchUpdateChunksResponse.deserialize,
            )
        return self._stubs["batch_update_chunks"]

    @property
    def delete_chunk(
        self,
    ) -> Callable[[retriever_service.DeleteChunkRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete chunk method over gRPC.

        Deletes a ``Chunk``.

        Returns:
            Callable[[~.DeleteChunkRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_chunk" not in self._stubs:
            self._stubs["delete_chunk"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/DeleteChunk",
                request_serializer=retriever_service.DeleteChunkRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_chunk"]

    @property
    def batch_delete_chunks(
        self,
    ) -> Callable[[retriever_service.BatchDeleteChunksRequest], empty_pb2.Empty]:
        r"""Return a callable for the batch delete chunks method over gRPC.

        Batch delete ``Chunk``\ s.

        Returns:
            Callable[[~.BatchDeleteChunksRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_delete_chunks" not in self._stubs:
            self._stubs["batch_delete_chunks"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/BatchDeleteChunks",
                request_serializer=retriever_service.BatchDeleteChunksRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["batch_delete_chunks"]

    @property
    def list_chunks(
        self,
    ) -> Callable[
        [retriever_service.ListChunksRequest], retriever_service.ListChunksResponse
    ]:
        r"""Return a callable for the list chunks method over gRPC.

        Lists all ``Chunk``\ s in a ``Document``.

        Returns:
            Callable[[~.ListChunksRequest],
                    ~.ListChunksResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_chunks" not in self._stubs:
            self._stubs["list_chunks"] = self.grpc_channel.unary_unary(
                "/google.ai.generativelanguage.v1beta.RetrieverService/ListChunks",
                request_serializer=retriever_service.ListChunksRequest.serialize,
                response_deserializer=retriever_service.ListChunksResponse.deserialize,
            )
        return self._stubs["list_chunks"]

    def close(self):
        self.grpc_channel.close()

    @property
    def kind(self) -> str:
        return "grpc"


__all__ = ("RetrieverServiceGrpcTransport",)
