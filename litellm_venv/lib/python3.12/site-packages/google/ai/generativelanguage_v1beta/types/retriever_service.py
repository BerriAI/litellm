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

from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore

from google.ai.generativelanguage_v1beta.types import retriever

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta",
    manifest={
        "CreateCorpusRequest",
        "GetCorpusRequest",
        "UpdateCorpusRequest",
        "DeleteCorpusRequest",
        "ListCorporaRequest",
        "ListCorporaResponse",
        "QueryCorpusRequest",
        "QueryCorpusResponse",
        "RelevantChunk",
        "CreateDocumentRequest",
        "GetDocumentRequest",
        "UpdateDocumentRequest",
        "DeleteDocumentRequest",
        "ListDocumentsRequest",
        "ListDocumentsResponse",
        "QueryDocumentRequest",
        "QueryDocumentResponse",
        "CreateChunkRequest",
        "BatchCreateChunksRequest",
        "BatchCreateChunksResponse",
        "GetChunkRequest",
        "UpdateChunkRequest",
        "BatchUpdateChunksRequest",
        "BatchUpdateChunksResponse",
        "DeleteChunkRequest",
        "BatchDeleteChunksRequest",
        "ListChunksRequest",
        "ListChunksResponse",
    },
)


class CreateCorpusRequest(proto.Message):
    r"""Request to create a ``Corpus``.

    Attributes:
        corpus (google.ai.generativelanguage_v1beta.types.Corpus):
            Required. The ``Corpus`` to create.
    """

    corpus: retriever.Corpus = proto.Field(
        proto.MESSAGE,
        number=1,
        message=retriever.Corpus,
    )


class GetCorpusRequest(proto.Message):
    r"""Request for getting information about a specific ``Corpus``.

    Attributes:
        name (str):
            Required. The name of the ``Corpus``. Example:
            ``corpora/my-corpus-123``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateCorpusRequest(proto.Message):
    r"""Request to update a ``Corpus``.

    Attributes:
        corpus (google.ai.generativelanguage_v1beta.types.Corpus):
            Required. The ``Corpus`` to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The list of fields to update. Currently, this only
            supports updating ``display_name``.
    """

    corpus: retriever.Corpus = proto.Field(
        proto.MESSAGE,
        number=1,
        message=retriever.Corpus,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteCorpusRequest(proto.Message):
    r"""Request to delete a ``Corpus``.

    Attributes:
        name (str):
            Required. The resource name of the ``Corpus``. Example:
            ``corpora/my-corpus-123``
        force (bool):
            Optional. If set to true, any ``Document``\ s and objects
            related to this ``Corpus`` will also be deleted.

            If false (the default), a ``FAILED_PRECONDITION`` error will
            be returned if ``Corpus`` contains any ``Document``\ s.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class ListCorporaRequest(proto.Message):
    r"""Request for listing ``Corpora``.

    Attributes:
        page_size (int):
            Optional. The maximum number of ``Corpora`` to return (per
            page). The service may return fewer ``Corpora``.

            If unspecified, at most 10 ``Corpora`` will be returned. The
            maximum size limit is 20 ``Corpora`` per page.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListCorpora`` call.

            Provide the ``next_page_token`` returned in the response as
            an argument to the next request to retrieve the next page.

            When paginating, all other parameters provided to
            ``ListCorpora`` must match the call that provided the page
            token.
    """

    page_size: int = proto.Field(
        proto.INT32,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ListCorporaResponse(proto.Message):
    r"""Response from ``ListCorpora`` containing a paginated list of
    ``Corpora``. The results are sorted by ascending
    ``corpus.create_time``.

    Attributes:
        corpora (MutableSequence[google.ai.generativelanguage_v1beta.types.Corpus]):
            The returned corpora.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no more
            pages.
    """

    @property
    def raw_page(self):
        return self

    corpora: MutableSequence[retriever.Corpus] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=retriever.Corpus,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class QueryCorpusRequest(proto.Message):
    r"""Request for querying a ``Corpus``.

    Attributes:
        name (str):
            Required. The name of the ``Corpus`` to query. Example:
            ``corpora/my-corpus-123``
        query (str):
            Required. Query string to perform semantic
            search.
        metadata_filters (MutableSequence[google.ai.generativelanguage_v1beta.types.MetadataFilter]):
            Optional. Filter for ``Chunk`` and ``Document`` metadata.
            Each ``MetadataFilter`` object should correspond to a unique
            key. Multiple ``MetadataFilter`` objects are joined by
            logical "AND"s.

            Example query at document level: (year >= 2020 OR year <
            2010) AND (genre = drama OR genre = action)

            ``MetadataFilter`` object list: metadata_filters = [ {key =
            "document.custom_metadata.year" conditions = [{int_value =
            2020, operation = GREATER_EQUAL}, {int_value = 2010,
            operation = LESS}]}, {key = "document.custom_metadata.year"
            conditions = [{int_value = 2020, operation = GREATER_EQUAL},
            {int_value = 2010, operation = LESS}]}, {key =
            "document.custom_metadata.genre" conditions = [{string_value
            = "drama", operation = EQUAL}, {string_value = "action",
            operation = EQUAL}]}]

            Example query at chunk level for a numeric range of values:
            (year > 2015 AND year <= 2020)

            ``MetadataFilter`` object list: metadata_filters = [ {key =
            "chunk.custom_metadata.year" conditions = [{int_value =
            2015, operation = GREATER}]}, {key =
            "chunk.custom_metadata.year" conditions = [{int_value =
            2020, operation = LESS_EQUAL}]}]

            Note: "AND"s for the same key are only supported for numeric
            values. String values only support "OR"s for the same key.
        results_count (int):
            Optional. The maximum number of ``Chunk``\ s to return. The
            service may return fewer ``Chunk``\ s.

            If unspecified, at most 10 ``Chunk``\ s will be returned.
            The maximum specified result count is 100.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: str = proto.Field(
        proto.STRING,
        number=2,
    )
    metadata_filters: MutableSequence[retriever.MetadataFilter] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=retriever.MetadataFilter,
    )
    results_count: int = proto.Field(
        proto.INT32,
        number=4,
    )


class QueryCorpusResponse(proto.Message):
    r"""Response from ``QueryCorpus`` containing a list of relevant chunks.

    Attributes:
        relevant_chunks (MutableSequence[google.ai.generativelanguage_v1beta.types.RelevantChunk]):
            The relevant chunks.
    """

    relevant_chunks: MutableSequence["RelevantChunk"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="RelevantChunk",
    )


class RelevantChunk(proto.Message):
    r"""The information for a chunk relevant to a query.

    Attributes:
        chunk_relevance_score (float):
            ``Chunk`` relevance to the query.
        chunk (google.ai.generativelanguage_v1beta.types.Chunk):
            ``Chunk`` associated with the query.
    """

    chunk_relevance_score: float = proto.Field(
        proto.FLOAT,
        number=1,
    )
    chunk: retriever.Chunk = proto.Field(
        proto.MESSAGE,
        number=2,
        message=retriever.Chunk,
    )


class CreateDocumentRequest(proto.Message):
    r"""Request to create a ``Document``.

    Attributes:
        parent (str):
            Required. The name of the ``Corpus`` where this ``Document``
            will be created. Example: ``corpora/my-corpus-123``
        document (google.ai.generativelanguage_v1beta.types.Document):
            Required. The ``Document`` to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    document: retriever.Document = proto.Field(
        proto.MESSAGE,
        number=2,
        message=retriever.Document,
    )


class GetDocumentRequest(proto.Message):
    r"""Request for getting information about a specific ``Document``.

    Attributes:
        name (str):
            Required. The name of the ``Document`` to retrieve. Example:
            ``corpora/my-corpus-123/documents/the-doc-abc``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateDocumentRequest(proto.Message):
    r"""Request to update a ``Document``.

    Attributes:
        document (google.ai.generativelanguage_v1beta.types.Document):
            Required. The ``Document`` to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The list of fields to update. Currently, this only
            supports updating ``display_name`` and ``custom_metadata``.
    """

    document: retriever.Document = proto.Field(
        proto.MESSAGE,
        number=1,
        message=retriever.Document,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteDocumentRequest(proto.Message):
    r"""Request to delete a ``Document``.

    Attributes:
        name (str):
            Required. The resource name of the ``Document`` to delete.
            Example: ``corpora/my-corpus-123/documents/the-doc-abc``
        force (bool):
            Optional. If set to true, any ``Chunk``\ s and objects
            related to this ``Document`` will also be deleted.

            If false (the default), a ``FAILED_PRECONDITION`` error will
            be returned if ``Document`` contains any ``Chunk``\ s.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class ListDocumentsRequest(proto.Message):
    r"""Request for listing ``Document``\ s.

    Attributes:
        parent (str):
            Required. The name of the ``Corpus`` containing
            ``Document``\ s. Example: ``corpora/my-corpus-123``
        page_size (int):
            Optional. The maximum number of ``Document``\ s to return
            (per page). The service may return fewer ``Document``\ s.

            If unspecified, at most 10 ``Document``\ s will be returned.
            The maximum size limit is 20 ``Document``\ s per page.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListDocuments`` call.

            Provide the ``next_page_token`` returned in the response as
            an argument to the next request to retrieve the next page.

            When paginating, all other parameters provided to
            ``ListDocuments`` must match the call that provided the page
            token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListDocumentsResponse(proto.Message):
    r"""Response from ``ListDocuments`` containing a paginated list of
    ``Document``\ s. The ``Document``\ s are sorted by ascending
    ``document.create_time``.

    Attributes:
        documents (MutableSequence[google.ai.generativelanguage_v1beta.types.Document]):
            The returned ``Document``\ s.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no more
            pages.
    """

    @property
    def raw_page(self):
        return self

    documents: MutableSequence[retriever.Document] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=retriever.Document,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class QueryDocumentRequest(proto.Message):
    r"""Request for querying a ``Document``.

    Attributes:
        name (str):
            Required. The name of the ``Document`` to query. Example:
            ``corpora/my-corpus-123/documents/the-doc-abc``
        query (str):
            Required. Query string to perform semantic
            search.
        results_count (int):
            Optional. The maximum number of ``Chunk``\ s to return. The
            service may return fewer ``Chunk``\ s.

            If unspecified, at most 10 ``Chunk``\ s will be returned.
            The maximum specified result count is 100.
        metadata_filters (MutableSequence[google.ai.generativelanguage_v1beta.types.MetadataFilter]):
            Optional. Filter for ``Chunk`` metadata. Each
            ``MetadataFilter`` object should correspond to a unique key.
            Multiple ``MetadataFilter`` objects are joined by logical
            "AND"s.

            Note: ``Document``-level filtering is not supported for this
            request because a ``Document`` name is already specified.

            Example query: (year >= 2020 OR year < 2010) AND (genre =
            drama OR genre = action)

            ``MetadataFilter`` object list: metadata_filters = [ {key =
            "chunk.custom_metadata.year" conditions = [{int_value =
            2020, operation = GREATER_EQUAL}, {int_value = 2010,
            operation = LESS}}, {key = "chunk.custom_metadata.genre"
            conditions = [{string_value = "drama", operation = EQUAL},
            {string_value = "action", operation = EQUAL}}]

            Example query for a numeric range of values: (year > 2015
            AND year <= 2020)

            ``MetadataFilter`` object list: metadata_filters = [ {key =
            "chunk.custom_metadata.year" conditions = [{int_value =
            2015, operation = GREATER}]}, {key =
            "chunk.custom_metadata.year" conditions = [{int_value =
            2020, operation = LESS_EQUAL}]}]

            Note: "AND"s for the same key are only supported for numeric
            values. String values only support "OR"s for the same key.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: str = proto.Field(
        proto.STRING,
        number=2,
    )
    results_count: int = proto.Field(
        proto.INT32,
        number=3,
    )
    metadata_filters: MutableSequence[retriever.MetadataFilter] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=retriever.MetadataFilter,
    )


class QueryDocumentResponse(proto.Message):
    r"""Response from ``QueryDocument`` containing a list of relevant
    chunks.

    Attributes:
        relevant_chunks (MutableSequence[google.ai.generativelanguage_v1beta.types.RelevantChunk]):
            The returned relevant chunks.
    """

    relevant_chunks: MutableSequence["RelevantChunk"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="RelevantChunk",
    )


class CreateChunkRequest(proto.Message):
    r"""Request to create a ``Chunk``.

    Attributes:
        parent (str):
            Required. The name of the ``Document`` where this ``Chunk``
            will be created. Example:
            ``corpora/my-corpus-123/documents/the-doc-abc``
        chunk (google.ai.generativelanguage_v1beta.types.Chunk):
            Required. The ``Chunk`` to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    chunk: retriever.Chunk = proto.Field(
        proto.MESSAGE,
        number=2,
        message=retriever.Chunk,
    )


class BatchCreateChunksRequest(proto.Message):
    r"""Request to batch create ``Chunk``\ s.

    Attributes:
        parent (str):
            Optional. The name of the ``Document`` where this batch of
            ``Chunk``\ s will be created. The parent field in every
            ``CreateChunkRequest`` must match this value. Example:
            ``corpora/my-corpus-123/documents/the-doc-abc``
        requests (MutableSequence[google.ai.generativelanguage_v1beta.types.CreateChunkRequest]):
            Required. The request messages specifying the ``Chunk``\ s
            to create. A maximum of 100 ``Chunk``\ s can be created in a
            batch.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    requests: MutableSequence["CreateChunkRequest"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="CreateChunkRequest",
    )


class BatchCreateChunksResponse(proto.Message):
    r"""Response from ``BatchCreateChunks`` containing a list of created
    ``Chunk``\ s.

    Attributes:
        chunks (MutableSequence[google.ai.generativelanguage_v1beta.types.Chunk]):
            ``Chunk``\ s created.
    """

    chunks: MutableSequence[retriever.Chunk] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=retriever.Chunk,
    )


class GetChunkRequest(proto.Message):
    r"""Request for getting information about a specific ``Chunk``.

    Attributes:
        name (str):
            Required. The name of the ``Chunk`` to retrieve. Example:
            ``corpora/my-corpus-123/documents/the-doc-abc/chunks/some-chunk``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateChunkRequest(proto.Message):
    r"""Request to update a ``Chunk``.

    Attributes:
        chunk (google.ai.generativelanguage_v1beta.types.Chunk):
            Required. The ``Chunk`` to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The list of fields to update. Currently, this only
            supports updating ``custom_metadata`` and ``data``.
    """

    chunk: retriever.Chunk = proto.Field(
        proto.MESSAGE,
        number=1,
        message=retriever.Chunk,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class BatchUpdateChunksRequest(proto.Message):
    r"""Request to batch update ``Chunk``\ s.

    Attributes:
        parent (str):
            Optional. The name of the ``Document`` containing the
            ``Chunk``\ s to update. The parent field in every
            ``UpdateChunkRequest`` must match this value. Example:
            ``corpora/my-corpus-123/documents/the-doc-abc``
        requests (MutableSequence[google.ai.generativelanguage_v1beta.types.UpdateChunkRequest]):
            Required. The request messages specifying the ``Chunk``\ s
            to update. A maximum of 100 ``Chunk``\ s can be updated in a
            batch.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    requests: MutableSequence["UpdateChunkRequest"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="UpdateChunkRequest",
    )


class BatchUpdateChunksResponse(proto.Message):
    r"""Response from ``BatchUpdateChunks`` containing a list of updated
    ``Chunk``\ s.

    Attributes:
        chunks (MutableSequence[google.ai.generativelanguage_v1beta.types.Chunk]):
            ``Chunk``\ s updated.
    """

    chunks: MutableSequence[retriever.Chunk] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=retriever.Chunk,
    )


class DeleteChunkRequest(proto.Message):
    r"""Request to delete a ``Chunk``.

    Attributes:
        name (str):
            Required. The resource name of the ``Chunk`` to delete.
            Example:
            ``corpora/my-corpus-123/documents/the-doc-abc/chunks/some-chunk``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class BatchDeleteChunksRequest(proto.Message):
    r"""Request to batch delete ``Chunk``\ s.

    Attributes:
        parent (str):
            Optional. The name of the ``Document`` containing the
            ``Chunk``\ s to delete. The parent field in every
            ``DeleteChunkRequest`` must match this value. Example:
            ``corpora/my-corpus-123/documents/the-doc-abc``
        requests (MutableSequence[google.ai.generativelanguage_v1beta.types.DeleteChunkRequest]):
            Required. The request messages specifying the ``Chunk``\ s
            to delete.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    requests: MutableSequence["DeleteChunkRequest"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="DeleteChunkRequest",
    )


class ListChunksRequest(proto.Message):
    r"""Request for listing ``Chunk``\ s.

    Attributes:
        parent (str):
            Required. The name of the ``Document`` containing
            ``Chunk``\ s. Example:
            ``corpora/my-corpus-123/documents/the-doc-abc``
        page_size (int):
            Optional. The maximum number of ``Chunk``\ s to return (per
            page). The service may return fewer ``Chunk``\ s.

            If unspecified, at most 10 ``Chunk``\ s will be returned.
            The maximum size limit is 100 ``Chunk``\ s per page.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListChunks`` call.

            Provide the ``next_page_token`` returned in the response as
            an argument to the next request to retrieve the next page.

            When paginating, all other parameters provided to
            ``ListChunks`` must match the call that provided the page
            token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListChunksResponse(proto.Message):
    r"""Response from ``ListChunks`` containing a paginated list of
    ``Chunk``\ s. The ``Chunk``\ s are sorted by ascending
    ``chunk.create_time``.

    Attributes:
        chunks (MutableSequence[google.ai.generativelanguage_v1beta.types.Chunk]):
            The returned ``Chunk``\ s.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no more
            pages.
    """

    @property
    def raw_page(self):
        return self

    chunks: MutableSequence[retriever.Chunk] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=retriever.Chunk,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
