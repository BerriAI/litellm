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

from google.cloud.aiplatform_v1.types import index as gca_index
from google.cloud.aiplatform_v1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "CreateIndexRequest",
        "CreateIndexOperationMetadata",
        "GetIndexRequest",
        "ListIndexesRequest",
        "ListIndexesResponse",
        "UpdateIndexRequest",
        "UpdateIndexOperationMetadata",
        "DeleteIndexRequest",
        "UpsertDatapointsRequest",
        "UpsertDatapointsResponse",
        "RemoveDatapointsRequest",
        "RemoveDatapointsResponse",
        "NearestNeighborSearchOperationMetadata",
    },
)


class CreateIndexRequest(proto.Message):
    r"""Request message for
    [IndexService.CreateIndex][google.cloud.aiplatform.v1.IndexService.CreateIndex].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            Index in. Format:
            ``projects/{project}/locations/{location}``
        index (google.cloud.aiplatform_v1.types.Index):
            Required. The Index to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    index: gca_index.Index = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_index.Index,
    )


class CreateIndexOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [IndexService.CreateIndex][google.cloud.aiplatform.v1.IndexService.CreateIndex].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            The operation generic information.
        nearest_neighbor_search_operation_metadata (google.cloud.aiplatform_v1.types.NearestNeighborSearchOperationMetadata):
            The operation metadata with regard to
            Matching Engine Index operation.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    nearest_neighbor_search_operation_metadata: "NearestNeighborSearchOperationMetadata" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="NearestNeighborSearchOperationMetadata",
    )


class GetIndexRequest(proto.Message):
    r"""Request message for
    [IndexService.GetIndex][google.cloud.aiplatform.v1.IndexService.GetIndex]

    Attributes:
        name (str):
            Required. The name of the Index resource. Format:
            ``projects/{project}/locations/{location}/indexes/{index}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListIndexesRequest(proto.Message):
    r"""Request message for
    [IndexService.ListIndexes][google.cloud.aiplatform.v1.IndexService.ListIndexes].

    Attributes:
        parent (str):
            Required. The resource name of the Location from which to
            list the Indexes. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter.
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListIndexesResponse.next_page_token][google.cloud.aiplatform.v1.ListIndexesResponse.next_page_token]
            of the previous
            [IndexService.ListIndexes][google.cloud.aiplatform.v1.IndexService.ListIndexes]
            call.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListIndexesResponse(proto.Message):
    r"""Response message for
    [IndexService.ListIndexes][google.cloud.aiplatform.v1.IndexService.ListIndexes].

    Attributes:
        indexes (MutableSequence[google.cloud.aiplatform_v1.types.Index]):
            List of indexes in the requested page.
        next_page_token (str):
            A token to retrieve next page of results. Pass to
            [ListIndexesRequest.page_token][google.cloud.aiplatform.v1.ListIndexesRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    indexes: MutableSequence[gca_index.Index] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_index.Index,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateIndexRequest(proto.Message):
    r"""Request message for
    [IndexService.UpdateIndex][google.cloud.aiplatform.v1.IndexService.UpdateIndex].

    Attributes:
        index (google.cloud.aiplatform_v1.types.Index):
            Required. The Index which updates the
            resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            The update mask applies to the resource. For the
            ``FieldMask`` definition, see
            [google.protobuf.FieldMask][google.protobuf.FieldMask].
    """

    index: gca_index.Index = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_index.Index,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateIndexOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [IndexService.UpdateIndex][google.cloud.aiplatform.v1.IndexService.UpdateIndex].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            The operation generic information.
        nearest_neighbor_search_operation_metadata (google.cloud.aiplatform_v1.types.NearestNeighborSearchOperationMetadata):
            The operation metadata with regard to
            Matching Engine Index operation.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )
    nearest_neighbor_search_operation_metadata: "NearestNeighborSearchOperationMetadata" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="NearestNeighborSearchOperationMetadata",
    )


class DeleteIndexRequest(proto.Message):
    r"""Request message for
    [IndexService.DeleteIndex][google.cloud.aiplatform.v1.IndexService.DeleteIndex].

    Attributes:
        name (str):
            Required. The name of the Index resource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/indexes/{index}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpsertDatapointsRequest(proto.Message):
    r"""Request message for
    [IndexService.UpsertDatapoints][google.cloud.aiplatform.v1.IndexService.UpsertDatapoints]

    Attributes:
        index (str):
            Required. The name of the Index resource to be updated.
            Format:
            ``projects/{project}/locations/{location}/indexes/{index}``
        datapoints (MutableSequence[google.cloud.aiplatform_v1.types.IndexDatapoint]):
            A list of datapoints to be created/updated.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Update mask is used to specify the fields to be
            overwritten in the datapoints by the update. The fields
            specified in the update_mask are relative to each
            IndexDatapoint inside datapoints, not the full request.

            Updatable fields:

            -  Use ``all_restricts`` to update both restricts and
               numeric_restricts.
    """

    index: str = proto.Field(
        proto.STRING,
        number=1,
    )
    datapoints: MutableSequence[gca_index.IndexDatapoint] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=gca_index.IndexDatapoint,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=3,
        message=field_mask_pb2.FieldMask,
    )


class UpsertDatapointsResponse(proto.Message):
    r"""Response message for
    [IndexService.UpsertDatapoints][google.cloud.aiplatform.v1.IndexService.UpsertDatapoints]

    """


class RemoveDatapointsRequest(proto.Message):
    r"""Request message for
    [IndexService.RemoveDatapoints][google.cloud.aiplatform.v1.IndexService.RemoveDatapoints]

    Attributes:
        index (str):
            Required. The name of the Index resource to be updated.
            Format:
            ``projects/{project}/locations/{location}/indexes/{index}``
        datapoint_ids (MutableSequence[str]):
            A list of datapoint ids to be deleted.
    """

    index: str = proto.Field(
        proto.STRING,
        number=1,
    )
    datapoint_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )


class RemoveDatapointsResponse(proto.Message):
    r"""Response message for
    [IndexService.RemoveDatapoints][google.cloud.aiplatform.v1.IndexService.RemoveDatapoints]

    """


class NearestNeighborSearchOperationMetadata(proto.Message):
    r"""Runtime operation metadata with regard to Matching Engine
    Index.

    Attributes:
        content_validation_stats (MutableSequence[google.cloud.aiplatform_v1.types.NearestNeighborSearchOperationMetadata.ContentValidationStats]):
            The validation stats of the content (per file) to be
            inserted or updated on the Matching Engine Index resource.
            Populated if contentsDeltaUri is provided as part of
            [Index.metadata][google.cloud.aiplatform.v1.Index.metadata].
            Please note that, currently for those files that are broken
            or has unsupported file format, we will not have the stats
            for those files.
        data_bytes_count (int):
            The ingested data size in bytes.
    """

    class RecordError(proto.Message):
        r"""

        Attributes:
            error_type (google.cloud.aiplatform_v1.types.NearestNeighborSearchOperationMetadata.RecordError.RecordErrorType):
                The error type of this record.
            error_message (str):
                A human-readable message that is shown to the user to help
                them fix the error. Note that this message may change from
                time to time, your code should check against error_type as
                the source of truth.
            source_gcs_uri (str):
                Cloud Storage URI pointing to the original
                file in user's bucket.
            embedding_id (str):
                Empty if the embedding id is failed to parse.
            raw_record (str):
                The original content of this record.
        """

        class RecordErrorType(proto.Enum):
            r"""

            Values:
                ERROR_TYPE_UNSPECIFIED (0):
                    Default, shall not be used.
                EMPTY_LINE (1):
                    The record is empty.
                INVALID_JSON_SYNTAX (2):
                    Invalid json format.
                INVALID_CSV_SYNTAX (3):
                    Invalid csv format.
                INVALID_AVRO_SYNTAX (4):
                    Invalid avro format.
                INVALID_EMBEDDING_ID (5):
                    The embedding id is not valid.
                EMBEDDING_SIZE_MISMATCH (6):
                    The size of the embedding vectors does not
                    match with the specified dimension.
                NAMESPACE_MISSING (7):
                    The ``namespace`` field is missing.
                PARSING_ERROR (8):
                    Generic catch-all error. Only used for
                    validation failure where the root cause cannot
                    be easily retrieved programmatically.
                DUPLICATE_NAMESPACE (9):
                    There are multiple restricts with the same ``namespace``
                    value.
                OP_IN_DATAPOINT (10):
                    Numeric restrict has operator specified in
                    datapoint.
                MULTIPLE_VALUES (11):
                    Numeric restrict has multiple values
                    specified.
                INVALID_NUMERIC_VALUE (12):
                    Numeric restrict has invalid numeric value
                    specified.
                INVALID_ENCODING (13):
                    File is not in UTF_8 format.
            """
            ERROR_TYPE_UNSPECIFIED = 0
            EMPTY_LINE = 1
            INVALID_JSON_SYNTAX = 2
            INVALID_CSV_SYNTAX = 3
            INVALID_AVRO_SYNTAX = 4
            INVALID_EMBEDDING_ID = 5
            EMBEDDING_SIZE_MISMATCH = 6
            NAMESPACE_MISSING = 7
            PARSING_ERROR = 8
            DUPLICATE_NAMESPACE = 9
            OP_IN_DATAPOINT = 10
            MULTIPLE_VALUES = 11
            INVALID_NUMERIC_VALUE = 12
            INVALID_ENCODING = 13

        error_type: "NearestNeighborSearchOperationMetadata.RecordError.RecordErrorType" = proto.Field(
            proto.ENUM,
            number=1,
            enum="NearestNeighborSearchOperationMetadata.RecordError.RecordErrorType",
        )
        error_message: str = proto.Field(
            proto.STRING,
            number=2,
        )
        source_gcs_uri: str = proto.Field(
            proto.STRING,
            number=3,
        )
        embedding_id: str = proto.Field(
            proto.STRING,
            number=4,
        )
        raw_record: str = proto.Field(
            proto.STRING,
            number=5,
        )

    class ContentValidationStats(proto.Message):
        r"""

        Attributes:
            source_gcs_uri (str):
                Cloud Storage URI pointing to the original
                file in user's bucket.
            valid_record_count (int):
                Number of records in this file that were
                successfully processed.
            invalid_record_count (int):
                Number of records in this file we skipped due
                to validate errors.
            partial_errors (MutableSequence[google.cloud.aiplatform_v1.types.NearestNeighborSearchOperationMetadata.RecordError]):
                The detail information of the partial
                failures encountered for those invalid records
                that couldn't be parsed. Up to 50 partial errors
                will be reported.
        """

        source_gcs_uri: str = proto.Field(
            proto.STRING,
            number=1,
        )
        valid_record_count: int = proto.Field(
            proto.INT64,
            number=2,
        )
        invalid_record_count: int = proto.Field(
            proto.INT64,
            number=3,
        )
        partial_errors: MutableSequence[
            "NearestNeighborSearchOperationMetadata.RecordError"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=4,
            message="NearestNeighborSearchOperationMetadata.RecordError",
        )

    content_validation_stats: MutableSequence[
        ContentValidationStats
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=ContentValidationStats,
    )
    data_bytes_count: int = proto.Field(
        proto.INT64,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
