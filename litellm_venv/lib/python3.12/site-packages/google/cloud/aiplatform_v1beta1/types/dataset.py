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
from google.cloud.aiplatform_v1beta1.types import io
from google.cloud.aiplatform_v1beta1.types import saved_query
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "Dataset",
        "ImportDataConfig",
        "ExportDataConfig",
        "ExportFractionSplit",
    },
)


class Dataset(proto.Message):
    r"""A collection of DataItems and Annotations on them.

    Attributes:
        name (str):
            Output only. The resource name of the
            Dataset.
        display_name (str):
            Required. The user-defined name of the
            Dataset. The name can be up to 128 characters
            long and can consist of any UTF-8 characters.
        description (str):
            The description of the Dataset.
        metadata_schema_uri (str):
            Required. Points to a YAML file stored on
            Google Cloud Storage describing additional
            information about the Dataset. The schema is
            defined as an OpenAPI 3.0.2 Schema Object. The
            schema files that can be used here are found in
            gs://google-cloud-aiplatform/schema/dataset/metadata/.
        metadata (google.protobuf.struct_pb2.Value):
            Required. Additional information about the
            Dataset.
        data_item_count (int):
            Output only. The number of DataItems in this
            Dataset. Only apply for non-structured Dataset.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Dataset was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Dataset was
            last updated.
        etag (str):
            Used to perform consistent read-modify-write
            updates. If not set, a blind "overwrite" update
            happens.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to organize your
            Datasets.

            Label keys and values can be no longer than 64 characters
            (Unicode codepoints), can only contain lowercase letters,
            numeric characters, underscores and dashes. International
            characters are allowed. No more than 64 user labels can be
            associated with one Dataset (System labels are excluded).

            See https://goo.gl/xmQnxf for more information and examples
            of labels. System reserved label keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable. Following
            system labels exist for each Dataset:

            -  "aiplatform.googleapis.com/dataset_metadata_schema":
               output only, its value is the
               [metadata_schema's][google.cloud.aiplatform.v1beta1.Dataset.metadata_schema_uri]
               title.
        saved_queries (MutableSequence[google.cloud.aiplatform_v1beta1.types.SavedQuery]):
            All SavedQueries belong to the Dataset will be returned in
            List/Get Dataset response. The annotation_specs field will
            not be populated except for UI cases which will only use
            [annotation_spec_count][google.cloud.aiplatform.v1beta1.SavedQuery.annotation_spec_count].
            In CreateDataset request, a SavedQuery is created together
            if this field is set, up to one SavedQuery can be set in
            CreateDatasetRequest. The SavedQuery should not contain any
            AnnotationSpec.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Customer-managed encryption key spec for a
            Dataset. If set, this Dataset and all
            sub-resources of this Dataset will be secured by
            this key.
        metadata_artifact (str):
            Output only. The resource name of the Artifact that was
            created in MetadataStore when creating the Dataset. The
            Artifact resource name pattern is
            ``projects/{project}/locations/{location}/metadataStores/{metadata_store}/artifacts/{artifact}``.
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
        number=16,
    )
    metadata_schema_uri: str = proto.Field(
        proto.STRING,
        number=3,
    )
    metadata: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=8,
        message=struct_pb2.Value,
    )
    data_item_count: int = proto.Field(
        proto.INT64,
        number=10,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=6,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=7,
    )
    saved_queries: MutableSequence[saved_query.SavedQuery] = proto.RepeatedField(
        proto.MESSAGE,
        number=9,
        message=saved_query.SavedQuery,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=11,
        message=gca_encryption_spec.EncryptionSpec,
    )
    metadata_artifact: str = proto.Field(
        proto.STRING,
        number=17,
    )


class ImportDataConfig(proto.Message):
    r"""Describes the location from where we import data into a
    Dataset, together with the labels that will be applied to the
    DataItems and the Annotations.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_source (google.cloud.aiplatform_v1beta1.types.GcsSource):
            The Google Cloud Storage location for the
            input content.

            This field is a member of `oneof`_ ``source``.
        data_item_labels (MutableMapping[str, str]):
            Labels that will be applied to newly imported DataItems. If
            an identical DataItem as one being imported already exists
            in the Dataset, then these labels will be appended to these
            of the already existing one, and if labels with identical
            key is imported before, the old label value will be
            overwritten. If two DataItems are identical in the same
            import data operation, the labels will be combined and if
            key collision happens in this case, one of the values will
            be picked randomly. Two DataItems are considered identical
            if their content bytes are identical (e.g. image bytes or
            pdf bytes). These labels will be overridden by Annotation
            labels specified inside index file referenced by
            [import_schema_uri][google.cloud.aiplatform.v1beta1.ImportDataConfig.import_schema_uri],
            e.g. jsonl file.
        annotation_labels (MutableMapping[str, str]):
            Labels that will be applied to newly imported Annotations.
            If two Annotations are identical, one of them will be
            deduped. Two Annotations are considered identical if their
            [payload][google.cloud.aiplatform.v1beta1.Annotation.payload],
            [payload_schema_uri][google.cloud.aiplatform.v1beta1.Annotation.payload_schema_uri]
            and all of their
            [labels][google.cloud.aiplatform.v1beta1.Annotation.labels]
            are the same. These labels will be overridden by Annotation
            labels specified inside index file referenced by
            [import_schema_uri][google.cloud.aiplatform.v1beta1.ImportDataConfig.import_schema_uri],
            e.g. jsonl file.
        import_schema_uri (str):
            Required. Points to a YAML file stored on Google Cloud
            Storage describing the import format. Validation will be
            done against the schema. The schema is defined as an
            `OpenAPI 3.0.2 Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
    """

    gcs_source: io.GcsSource = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="source",
        message=io.GcsSource,
    )
    data_item_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=2,
    )
    annotation_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=3,
    )
    import_schema_uri: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ExportDataConfig(proto.Message):
    r"""Describes what part of the Dataset is to be exported, the
    destination of the export and how to export.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_destination (google.cloud.aiplatform_v1beta1.types.GcsDestination):
            The Google Cloud Storage location where the output is to be
            written to. In the given directory a new directory will be
            created with name:
            ``export-data-<dataset-display-name>-<timestamp-of-export-call>``
            where timestamp is in YYYY-MM-DDThh:mm:ss.sssZ ISO-8601
            format. All export output will be written into that
            directory. Inside that directory, annotations with the same
            schema will be grouped into sub directories which are named
            with the corresponding annotations' schema title. Inside
            these sub directories, a schema.yaml will be created to
            describe the output format.

            This field is a member of `oneof`_ ``destination``.
        fraction_split (google.cloud.aiplatform_v1beta1.types.ExportFractionSplit):
            Split based on fractions defining the size of
            each set.

            This field is a member of `oneof`_ ``split``.
        annotations_filter (str):
            An expression for filtering what part of the Dataset is to
            be exported. Only Annotations that match this filter will be
            exported. The filter syntax is the same as in
            [ListAnnotations][google.cloud.aiplatform.v1beta1.DatasetService.ListAnnotations].
    """

    gcs_destination: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="destination",
        message=io.GcsDestination,
    )
    fraction_split: "ExportFractionSplit" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="split",
        message="ExportFractionSplit",
    )
    annotations_filter: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ExportFractionSplit(proto.Message):
    r"""Assigns the input data to training, validation, and test sets as per
    the given fractions. Any of ``training_fraction``,
    ``validation_fraction`` and ``test_fraction`` may optionally be
    provided, they must sum to up to 1. If the provided ones sum to less
    than 1, the remainder is assigned to sets as decided by Vertex AI.
    If none of the fractions are set, by default roughly 80% of data is
    used for training, 10% for validation, and 10% for test.

    Attributes:
        training_fraction (float):
            The fraction of the input data that is to be
            used to train the Model.
        validation_fraction (float):
            The fraction of the input data that is to be
            used to validate the Model.
        test_fraction (float):
            The fraction of the input data that is to be
            used to evaluate the Model.
    """

    training_fraction: float = proto.Field(
        proto.DOUBLE,
        number=1,
    )
    validation_fraction: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    test_fraction: float = proto.Field(
        proto.DOUBLE,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
