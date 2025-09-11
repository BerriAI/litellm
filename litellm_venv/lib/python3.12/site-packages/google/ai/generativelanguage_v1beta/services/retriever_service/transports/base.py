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
import abc
from typing import Awaitable, Callable, Dict, Optional, Sequence, Union

import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.protobuf import empty_pb2  # type: ignore

from google.ai.generativelanguage_v1beta import gapic_version as package_version
from google.ai.generativelanguage_v1beta.types import retriever, retriever_service

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)


class RetrieverServiceTransport(abc.ABC):
    """Abstract transport class for RetrieverService."""

    AUTH_SCOPES = ()

    DEFAULT_HOST: str = "generativelanguage.googleapis.com"

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        api_audience: Optional[str] = None,
        **kwargs,
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
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A list of scopes.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
        """

        scopes_kwargs = {"scopes": scopes, "default_scopes": self.AUTH_SCOPES}

        # Save the scopes.
        self._scopes = scopes

        # If no credentials are provided, then determine the appropriate
        # defaults.
        if credentials and credentials_file:
            raise core_exceptions.DuplicateCredentialArgs(
                "'credentials_file' and 'credentials' are mutually exclusive"
            )

        if credentials_file is not None:
            credentials, _ = google.auth.load_credentials_from_file(
                credentials_file, **scopes_kwargs, quota_project_id=quota_project_id
            )
        elif credentials is None:
            credentials, _ = google.auth.default(
                **scopes_kwargs, quota_project_id=quota_project_id
            )
            # Don't apply audience if the credentials file passed from user.
            if hasattr(credentials, "with_gdch_audience"):
                credentials = credentials.with_gdch_audience(
                    api_audience if api_audience else host
                )

        # If the credentials are service account credentials, then always try to use self signed JWT.
        if (
            always_use_jwt_access
            and isinstance(credentials, service_account.Credentials)
            and hasattr(service_account.Credentials, "with_always_use_jwt_access")
        ):
            credentials = credentials.with_always_use_jwt_access(True)

        # Save the credentials.
        self._credentials = credentials

        # Save the hostname. Default to port 443 (HTTPS) if none is specified.
        if ":" not in host:
            host += ":443"
        self._host = host

    @property
    def host(self):
        return self._host

    def _prep_wrapped_messages(self, client_info):
        # Precompute the wrapped methods.
        self._wrapped_methods = {
            self.create_corpus: gapic_v1.method.wrap_method(
                self.create_corpus,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_corpus: gapic_v1.method.wrap_method(
                self.get_corpus,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.update_corpus: gapic_v1.method.wrap_method(
                self.update_corpus,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.delete_corpus: gapic_v1.method.wrap_method(
                self.delete_corpus,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.list_corpora: gapic_v1.method.wrap_method(
                self.list_corpora,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.query_corpus: gapic_v1.method.wrap_method(
                self.query_corpus,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.create_document: gapic_v1.method.wrap_method(
                self.create_document,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.get_document: gapic_v1.method.wrap_method(
                self.get_document,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.update_document: gapic_v1.method.wrap_method(
                self.update_document,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.delete_document: gapic_v1.method.wrap_method(
                self.delete_document,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.list_documents: gapic_v1.method.wrap_method(
                self.list_documents,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.query_document: gapic_v1.method.wrap_method(
                self.query_document,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.create_chunk: gapic_v1.method.wrap_method(
                self.create_chunk,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.batch_create_chunks: gapic_v1.method.wrap_method(
                self.batch_create_chunks,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_chunk: gapic_v1.method.wrap_method(
                self.get_chunk,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.update_chunk: gapic_v1.method.wrap_method(
                self.update_chunk,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.batch_update_chunks: gapic_v1.method.wrap_method(
                self.batch_update_chunks,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_chunk: gapic_v1.method.wrap_method(
                self.delete_chunk,
                default_retry=retries.Retry(
                    initial=1.0,
                    maximum=10.0,
                    multiplier=1.3,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=60.0,
                ),
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.batch_delete_chunks: gapic_v1.method.wrap_method(
                self.batch_delete_chunks,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_chunks: gapic_v1.method.wrap_method(
                self.list_chunks,
                default_timeout=None,
                client_info=client_info,
            ),
        }

    def close(self):
        """Closes resources associated with the transport.

        .. warning::
             Only call this method if the transport is NOT shared
             with other clients - this may cause errors in other clients!
        """
        raise NotImplementedError()

    @property
    def create_corpus(
        self,
    ) -> Callable[
        [retriever_service.CreateCorpusRequest],
        Union[retriever.Corpus, Awaitable[retriever.Corpus]],
    ]:
        raise NotImplementedError()

    @property
    def get_corpus(
        self,
    ) -> Callable[
        [retriever_service.GetCorpusRequest],
        Union[retriever.Corpus, Awaitable[retriever.Corpus]],
    ]:
        raise NotImplementedError()

    @property
    def update_corpus(
        self,
    ) -> Callable[
        [retriever_service.UpdateCorpusRequest],
        Union[retriever.Corpus, Awaitable[retriever.Corpus]],
    ]:
        raise NotImplementedError()

    @property
    def delete_corpus(
        self,
    ) -> Callable[
        [retriever_service.DeleteCorpusRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def list_corpora(
        self,
    ) -> Callable[
        [retriever_service.ListCorporaRequest],
        Union[
            retriever_service.ListCorporaResponse,
            Awaitable[retriever_service.ListCorporaResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def query_corpus(
        self,
    ) -> Callable[
        [retriever_service.QueryCorpusRequest],
        Union[
            retriever_service.QueryCorpusResponse,
            Awaitable[retriever_service.QueryCorpusResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_document(
        self,
    ) -> Callable[
        [retriever_service.CreateDocumentRequest],
        Union[retriever.Document, Awaitable[retriever.Document]],
    ]:
        raise NotImplementedError()

    @property
    def get_document(
        self,
    ) -> Callable[
        [retriever_service.GetDocumentRequest],
        Union[retriever.Document, Awaitable[retriever.Document]],
    ]:
        raise NotImplementedError()

    @property
    def update_document(
        self,
    ) -> Callable[
        [retriever_service.UpdateDocumentRequest],
        Union[retriever.Document, Awaitable[retriever.Document]],
    ]:
        raise NotImplementedError()

    @property
    def delete_document(
        self,
    ) -> Callable[
        [retriever_service.DeleteDocumentRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def list_documents(
        self,
    ) -> Callable[
        [retriever_service.ListDocumentsRequest],
        Union[
            retriever_service.ListDocumentsResponse,
            Awaitable[retriever_service.ListDocumentsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def query_document(
        self,
    ) -> Callable[
        [retriever_service.QueryDocumentRequest],
        Union[
            retriever_service.QueryDocumentResponse,
            Awaitable[retriever_service.QueryDocumentResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_chunk(
        self,
    ) -> Callable[
        [retriever_service.CreateChunkRequest],
        Union[retriever.Chunk, Awaitable[retriever.Chunk]],
    ]:
        raise NotImplementedError()

    @property
    def batch_create_chunks(
        self,
    ) -> Callable[
        [retriever_service.BatchCreateChunksRequest],
        Union[
            retriever_service.BatchCreateChunksResponse,
            Awaitable[retriever_service.BatchCreateChunksResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_chunk(
        self,
    ) -> Callable[
        [retriever_service.GetChunkRequest],
        Union[retriever.Chunk, Awaitable[retriever.Chunk]],
    ]:
        raise NotImplementedError()

    @property
    def update_chunk(
        self,
    ) -> Callable[
        [retriever_service.UpdateChunkRequest],
        Union[retriever.Chunk, Awaitable[retriever.Chunk]],
    ]:
        raise NotImplementedError()

    @property
    def batch_update_chunks(
        self,
    ) -> Callable[
        [retriever_service.BatchUpdateChunksRequest],
        Union[
            retriever_service.BatchUpdateChunksResponse,
            Awaitable[retriever_service.BatchUpdateChunksResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_chunk(
        self,
    ) -> Callable[
        [retriever_service.DeleteChunkRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def batch_delete_chunks(
        self,
    ) -> Callable[
        [retriever_service.BatchDeleteChunksRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def list_chunks(
        self,
    ) -> Callable[
        [retriever_service.ListChunksRequest],
        Union[
            retriever_service.ListChunksResponse,
            Awaitable[retriever_service.ListChunksResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def kind(self) -> str:
        raise NotImplementedError()


__all__ = ("RetrieverServiceTransport",)
