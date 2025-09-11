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

from google.cloud.aiplatform_v1beta1.types import operation
from google.cloud.aiplatform_v1beta1.types import vertex_rag_data
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateRagCorpusRequest",
        "GetRagCorpusRequest",
        "ListRagCorporaRequest",
        "ListRagCorporaResponse",
        "DeleteRagCorpusRequest",
        "UploadRagFileRequest",
        "UploadRagFileResponse",
        "ImportRagFilesRequest",
        "ImportRagFilesResponse",
        "GetRagFileRequest",
        "ListRagFilesRequest",
        "ListRagFilesResponse",
        "DeleteRagFileRequest",
        "CreateRagCorpusOperationMetadata",
        "ImportRagFilesOperationMetadata",
    },
)


class CreateRagCorpusRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.CreateRagCorpus][google.cloud.aiplatform.v1beta1.VertexRagDataService.CreateRagCorpus].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            RagCorpus in. Format:
            ``projects/{project}/locations/{location}``
        rag_corpus (google.cloud.aiplatform_v1beta1.types.RagCorpus):
            Required. The RagCorpus to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    rag_corpus: vertex_rag_data.RagCorpus = proto.Field(
        proto.MESSAGE,
        number=2,
        message=vertex_rag_data.RagCorpus,
    )


class GetRagCorpusRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.GetRagCorpus][google.cloud.aiplatform.v1beta1.VertexRagDataService.GetRagCorpus]

    Attributes:
        name (str):
            Required. The name of the RagCorpus resource. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListRagCorporaRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.ListRagCorpora][google.cloud.aiplatform.v1beta1.VertexRagDataService.ListRagCorpora].

    Attributes:
        parent (str):
            Required. The resource name of the Location from which to
            list the RagCorpora. Format:
            ``projects/{project}/locations/{location}``
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token. Typically obtained
            via
            [ListRagCorporaResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListRagCorporaResponse.next_page_token]
            of the previous
            [VertexRagDataService.ListRagCorpora][google.cloud.aiplatform.v1beta1.VertexRagDataService.ListRagCorpora]
            call.
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


class ListRagCorporaResponse(proto.Message):
    r"""Response message for
    [VertexRagDataService.ListRagCorpora][google.cloud.aiplatform.v1beta1.VertexRagDataService.ListRagCorpora].

    Attributes:
        rag_corpora (MutableSequence[google.cloud.aiplatform_v1beta1.types.RagCorpus]):
            List of RagCorpora in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListRagCorporaRequest.page_token][google.cloud.aiplatform.v1beta1.ListRagCorporaRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    rag_corpora: MutableSequence[vertex_rag_data.RagCorpus] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=vertex_rag_data.RagCorpus,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteRagCorpusRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.DeleteRagCorpus][google.cloud.aiplatform.v1beta1.VertexRagDataService.DeleteRagCorpus].

    Attributes:
        name (str):
            Required. The name of the RagCorpus resource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
        force (bool):
            Optional. If set to true, any RagFiles in
            this RagCorpus will also be deleted. Otherwise,
            the request will only work if the RagCorpus has
            no RagFiles.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class UploadRagFileRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.UploadRagFile][google.cloud.aiplatform.v1beta1.VertexRagDataService.UploadRagFile].

    Attributes:
        parent (str):
            Required. The name of the RagCorpus resource into which to
            upload the file. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
        rag_file (google.cloud.aiplatform_v1beta1.types.RagFile):
            Required. The RagFile to upload.
        upload_rag_file_config (google.cloud.aiplatform_v1beta1.types.UploadRagFileConfig):
            Required. The config for the RagFiles to be uploaded into
            the RagCorpus.
            [VertexRagDataService.UploadRagFile][google.cloud.aiplatform.v1beta1.VertexRagDataService.UploadRagFile].
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    rag_file: vertex_rag_data.RagFile = proto.Field(
        proto.MESSAGE,
        number=2,
        message=vertex_rag_data.RagFile,
    )
    upload_rag_file_config: vertex_rag_data.UploadRagFileConfig = proto.Field(
        proto.MESSAGE,
        number=5,
        message=vertex_rag_data.UploadRagFileConfig,
    )


class UploadRagFileResponse(proto.Message):
    r"""Response message for
    [VertexRagDataService.UploadRagFile][google.cloud.aiplatform.v1beta1.VertexRagDataService.UploadRagFile].

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        rag_file (google.cloud.aiplatform_v1beta1.types.RagFile):
            The RagFile that had been uploaded into the
            RagCorpus.

            This field is a member of `oneof`_ ``result``.
        error (google.rpc.status_pb2.Status):
            The error that occurred while processing the
            RagFile.

            This field is a member of `oneof`_ ``result``.
    """

    rag_file: vertex_rag_data.RagFile = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="result",
        message=vertex_rag_data.RagFile,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="result",
        message=status_pb2.Status,
    )


class ImportRagFilesRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.ImportRagFiles][google.cloud.aiplatform.v1beta1.VertexRagDataService.ImportRagFiles].

    Attributes:
        parent (str):
            Required. The name of the RagCorpus resource into which to
            import files. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
        import_rag_files_config (google.cloud.aiplatform_v1beta1.types.ImportRagFilesConfig):
            Required. The config for the RagFiles to be synced and
            imported into the RagCorpus.
            [VertexRagDataService.ImportRagFiles][google.cloud.aiplatform.v1beta1.VertexRagDataService.ImportRagFiles].
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    import_rag_files_config: vertex_rag_data.ImportRagFilesConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        message=vertex_rag_data.ImportRagFilesConfig,
    )


class ImportRagFilesResponse(proto.Message):
    r"""Response message for
    [VertexRagDataService.ImportRagFiles][google.cloud.aiplatform.v1beta1.VertexRagDataService.ImportRagFiles].

    Attributes:
        imported_rag_files_count (int):
            The number of RagFiles that had been imported
            into the RagCorpus.
    """

    imported_rag_files_count: int = proto.Field(
        proto.INT64,
        number=1,
    )


class GetRagFileRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.GetRagFile][google.cloud.aiplatform.v1beta1.VertexRagDataService.GetRagFile]

    Attributes:
        name (str):
            Required. The name of the RagFile resource. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}/ragFiles/{rag_file}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListRagFilesRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.ListRagFiles][google.cloud.aiplatform.v1beta1.VertexRagDataService.ListRagFiles].

    Attributes:
        parent (str):
            Required. The resource name of the RagCorpus from which to
            list the RagFiles. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token. Typically obtained
            via
            [ListRagFilesResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListRagFilesResponse.next_page_token]
            of the previous
            [VertexRagDataService.ListRagFiles][google.cloud.aiplatform.v1beta1.VertexRagDataService.ListRagFiles]
            call.
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


class ListRagFilesResponse(proto.Message):
    r"""Response message for
    [VertexRagDataService.ListRagFiles][google.cloud.aiplatform.v1beta1.VertexRagDataService.ListRagFiles].

    Attributes:
        rag_files (MutableSequence[google.cloud.aiplatform_v1beta1.types.RagFile]):
            List of RagFiles in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListRagFilesRequest.page_token][google.cloud.aiplatform.v1beta1.ListRagFilesRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    rag_files: MutableSequence[vertex_rag_data.RagFile] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=vertex_rag_data.RagFile,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteRagFileRequest(proto.Message):
    r"""Request message for
    [VertexRagDataService.DeleteRagFile][google.cloud.aiplatform.v1beta1.VertexRagDataService.DeleteRagFile].

    Attributes:
        name (str):
            Required. The name of the RagFile resource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}/ragFiles/{rag_file}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateRagCorpusOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [VertexRagDataService.CreateRagCorpus][google.cloud.aiplatform.v1beta1.VertexRagDataService.CreateRagCorpus].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class ImportRagFilesOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [VertexRagDataService.ImportRagFiles][google.cloud.aiplatform.v1beta1.VertexRagDataService.ImportRagFiles].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
        rag_corpus_id (int):
            The resource ID of RagCorpus that this
            operation is executed on.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    rag_corpus_id: int = proto.Field(
        proto.INT64,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
