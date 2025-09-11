# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from google.api_core import operation
from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils

from google.cloud.aiplatform.compat.services import dataset_service_client
from google.cloud.aiplatform.compat.types import (
    dataset as gca_dataset,
    dataset_service as gca_dataset_service,
    encryption_spec as gca_encryption_spec,
    io as gca_io,
)
from google.cloud.aiplatform.datasets import _datasources
from google.protobuf import field_mask_pb2
from google.protobuf import json_format

_LOGGER = base.Logger(__name__)


class _Dataset(base.VertexAiResourceNounWithFutureManager):
    """Managed dataset resource for Vertex AI."""

    client_class = utils.DatasetClientWithOverride
    _resource_noun = "datasets"
    _getter_method = "get_dataset"
    _list_method = "list_datasets"
    _delete_method = "delete_dataset"
    _parse_resource_name_method = "parse_dataset_path"
    _format_resource_name_method = "dataset_path"

    _supported_metadata_schema_uris: Tuple[str] = ()

    def __init__(
        self,
        dataset_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing managed dataset given a dataset name or ID.

        Args:
            dataset_name (str):
                Required. A fully-qualified dataset resource name or dataset ID.
                Example: "projects/123/locations/us-central1/datasets/456" or
                "456" when project and location are initialized or passed.
            project (str):
                Optional project to retrieve dataset from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional location to retrieve dataset from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to retrieve this Dataset. Overrides
                credentials set in aiplatform.init.
        """

        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=dataset_name,
        )
        self._gca_resource = self._get_gca_resource(resource_name=dataset_name)
        self._validate_metadata_schema_uri()

    @property
    def metadata_schema_uri(self) -> str:
        """The metadata schema uri of this dataset resource."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.metadata_schema_uri

    def _validate_metadata_schema_uri(self) -> None:
        """Validate the metadata_schema_uri of retrieved dataset resource.

        Raises:
            ValueError: If the dataset type of the retrieved dataset resource is
            not supported by the class.
        """
        if self._supported_metadata_schema_uris and (
            self.metadata_schema_uri not in self._supported_metadata_schema_uris
        ):
            raise ValueError(
                f"{self.__class__.__name__} class can not be used to retrieve "
                f"dataset resource {self.resource_name}, check the dataset type"
            )

    @classmethod
    def create(
        cls,
        # TODO(b/223262536): Make the display_name parameter optional in the next major release
        display_name: str,
        metadata_schema_uri: str,
        gcs_source: Optional[Union[str, Sequence[str]]] = None,
        bq_source: Optional[str] = None,
        import_schema_uri: Optional[str] = None,
        data_item_labels: Optional[Dict] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "_Dataset":
        """Creates a new dataset and optionally imports data into dataset when
        source and import_schema_uri are passed.

        Args:
            display_name (str):
                Required. The user-defined name of the Dataset.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            metadata_schema_uri (str):
                Required. Points to a YAML file stored on Google Cloud Storage
                describing additional information about the Dataset. The schema
                is defined as an OpenAPI 3.0.2 Schema Object. The schema files
                that can be used here are found in gs://google-cloud-
                aiplatform/schema/dataset/metadata/.
            gcs_source (Union[str, Sequence[str]]):
                Google Cloud Storage URI(-s) to the
                input file(s). May contain wildcards. For more
                information on wildcards, see
                https://cloud.google.com/storage/docs/gsutil/addlhelp/WildcardNames.
                examples:
                    str: "gs://bucket/file.csv"
                    Sequence[str]: ["gs://bucket/file1.csv", "gs://bucket/file2.csv"]
            bq_source (str):
                BigQuery URI to the input table.
                example:
                    "bq://project.dataset.table_name"
            import_schema_uri (str):
                Points to a YAML file stored on Google Cloud
                Storage describing the import format. Validation will be
                done against the schema. The schema is defined as an
                `OpenAPI 3.0.2 Schema
                Object <https://tinyurl.com/y538mdwt>`__.
            data_item_labels (Dict):
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
                ``import_schema_uri``,
                e.g. jsonl file.
                This arg is not for specifying the annotation name or the
                training target of your data, but for some global labels of
                the dataset. E.g.,
                'data_item_labels={"aiplatform.googleapis.com/ml_use":"training"}'
                specifies that all the uploaded data are used for training.
            project (str):
                Project to upload this dataset to. Overrides project set in
                aiplatform.init.
            location (str):
                Location to upload this dataset to. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to upload this dataset. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Strings which should be sent along with the request as metadata.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your datasets.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                No more than 64 user labels can be associated with one Dataset
                (System labels are excluded).
                See https://goo.gl/xmQnxf for more information and examples of labels.
                System reserved label keys are prefixed with "aiplatform.googleapis.com/"
                and are immutable.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the dataset. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Dataset and all sub-resources of this Dataset will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.

        Returns:
            dataset (Dataset):
                Instantiated representation of the managed dataset resource.
        """
        if not display_name:
            display_name = cls._generate_display_name()
        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)

        api_client = cls._instantiate_client(location=location, credentials=credentials)

        datasource = _datasources.create_datasource(
            metadata_schema_uri=metadata_schema_uri,
            import_schema_uri=import_schema_uri,
            gcs_source=gcs_source,
            bq_source=bq_source,
            data_item_labels=data_item_labels,
        )

        return cls._create_and_import(
            api_client=api_client,
            parent=initializer.global_config.common_location_path(
                project=project, location=location
            ),
            display_name=display_name,
            metadata_schema_uri=metadata_schema_uri,
            datasource=datasource,
            project=project or initializer.global_config.project,
            location=location or initializer.global_config.location,
            credentials=credentials or initializer.global_config.credentials,
            request_metadata=request_metadata,
            labels=labels,
            encryption_spec=initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            ),
            sync=sync,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    @base.optional_sync()
    def _create_and_import(
        cls,
        api_client: dataset_service_client.DatasetServiceClient,
        parent: str,
        display_name: str,
        metadata_schema_uri: str,
        datasource: _datasources.Datasource,
        project: str,
        location: str,
        credentials: Optional[auth_credentials.Credentials],
        request_metadata: Optional[Sequence[Tuple[str, str]]] = (),
        labels: Optional[Dict[str, str]] = None,
        encryption_spec: Optional[gca_encryption_spec.EncryptionSpec] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
        import_request_timeout: Optional[float] = None,
    ) -> "_Dataset":
        """Creates a new dataset and optionally imports data into dataset when
        source and import_schema_uri are passed.

        Args:
            api_client (dataset_service_client.DatasetServiceClient):
                An instance of DatasetServiceClient with the correct api_endpoint
                already set based on user's preferences.
            parent (str):
                Required. Also known as common location path, that usually contains the
                project and location that the user provided to the upstream method.
                Example: "projects/my-prj/locations/us-central1"
            display_name (str):
                Required. The user-defined name of the Dataset.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            metadata_schema_uri (str):
                Required. Points to a YAML file stored on Google Cloud Storage
                describing additional information about the Dataset. The schema
                is defined as an OpenAPI 3.0.2 Schema Object. The schema files
                that can be used here are found in gs://google-cloud-
                aiplatform/schema/dataset/metadata/.
            datasource (_datasources.Datasource):
                Required. Datasource for creating a dataset for Vertex AI.
            project (str):
                Required. Project to upload this model to. Overrides project set in
                aiplatform.init.
            location (str):
                Required. Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials (Optional[auth_credentials.Credentials]):
                Custom credentials to use to upload this model. Overrides
                credentials set in aiplatform.init.
            request_metadata (Sequence[Tuple[str, str]]):
                Strings which should be sent along with the request as metadata.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your Tensorboards.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                No more than 64 user labels can be associated with one Tensorboard
                (System labels are excluded).
                See https://goo.gl/xmQnxf for more information and examples of labels.
                System reserved label keys are prefixed with "aiplatform.googleapis.com/"
                and are immutable.
            encryption_spec (Optional[gca_encryption_spec.EncryptionSpec]):
                Optional. The Cloud KMS customer managed encryption key used to protect the dataset.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Dataset and all sub-resources of this Dataset will be secured by this key.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            import_request_timeout (float):
                Optional. The timeout for the import request in seconds.

        Returns:
            dataset (Dataset):
                Instantiated representation of the managed dataset resource.
        """

        create_dataset_lro = cls._create(
            api_client=api_client,
            parent=parent,
            display_name=display_name,
            metadata_schema_uri=metadata_schema_uri,
            datasource=datasource,
            request_metadata=request_metadata,
            labels=labels,
            encryption_spec=encryption_spec,
            create_request_timeout=create_request_timeout,
        )

        _LOGGER.log_create_with_lro(cls, create_dataset_lro)

        created_dataset = create_dataset_lro.result(timeout=None)

        _LOGGER.log_create_complete(cls, created_dataset, "ds")

        dataset_obj = cls(
            dataset_name=created_dataset.name,
            project=project,
            location=location,
            credentials=credentials,
        )

        # Import if import datasource is DatasourceImportable
        if isinstance(datasource, _datasources.DatasourceImportable):
            dataset_obj._import_and_wait(
                datasource, import_request_timeout=import_request_timeout
            )

        return dataset_obj

    def _import_and_wait(
        self,
        datasource,
        import_request_timeout: Optional[float] = None,
    ):
        _LOGGER.log_action_start_against_resource(
            "Importing",
            "data",
            self,
        )

        import_lro = self._import(
            datasource=datasource, import_request_timeout=import_request_timeout
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Import", "data", self.__class__, import_lro
        )

        import_lro.result(timeout=None)

        _LOGGER.log_action_completed_against_resource("data", "imported", self)

    @classmethod
    def _create(
        cls,
        api_client: dataset_service_client.DatasetServiceClient,
        parent: str,
        display_name: str,
        metadata_schema_uri: str,
        datasource: _datasources.Datasource,
        request_metadata: Sequence[Tuple[str, str]] = (),
        labels: Optional[Dict[str, str]] = None,
        encryption_spec: Optional[gca_encryption_spec.EncryptionSpec] = None,
        create_request_timeout: Optional[float] = None,
    ) -> operation.Operation:
        """Creates a new managed dataset by directly calling API client.

        Args:
            api_client (dataset_service_client.DatasetServiceClient):
                An instance of DatasetServiceClient with the correct api_endpoint
                already set based on user's preferences.
            parent (str):
                Required. Also known as common location path, that usually contains the
                project and location that the user provided to the upstream method.
                Example: "projects/my-prj/locations/us-central1"
            display_name (str):
                Required. The user-defined name of the Dataset.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            metadata_schema_uri (str):
                Required. Points to a YAML file stored on Google Cloud Storage
                describing additional information about the Dataset. The schema
                is defined as an OpenAPI 3.0.2 Schema Object. The schema files
                that can be used here are found in gs://google-cloud-
                aiplatform/schema/dataset/metadata/.
            datasource (_datasources.Datasource):
                Required. Datasource for creating a dataset for Vertex AI.
            request_metadata (Sequence[Tuple[str, str]]):
                Strings which should be sent along with the create_dataset
                request as metadata. Usually to specify special dataset config.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your Tensorboards.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                No more than 64 user labels can be associated with one Tensorboard
                (System labels are excluded).
                See https://goo.gl/xmQnxf for more information and examples of labels.
                System reserved label keys are prefixed with "aiplatform.googleapis.com/"
                and are immutable.
            encryption_spec (Optional[gca_encryption_spec.EncryptionSpec]):
                Optional. The Cloud KMS customer managed encryption key used to protect the dataset.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Dataset and all sub-resources of this Dataset will be secured by this key.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
        Returns:
            operation (Operation):
                An object representing a long-running operation.
        """

        gapic_dataset = gca_dataset.Dataset(
            display_name=display_name,
            metadata_schema_uri=metadata_schema_uri,
            metadata=datasource.dataset_metadata,
            labels=labels,
            encryption_spec=encryption_spec,
        )

        return api_client.create_dataset(
            parent=parent,
            dataset=gapic_dataset,
            metadata=request_metadata,
            timeout=create_request_timeout,
        )

    def _import(
        self,
        datasource: _datasources.DatasourceImportable,
        import_request_timeout: Optional[float] = None,
    ) -> operation.Operation:
        """Imports data into managed dataset by directly calling API client.

        Args:
            datasource (_datasources.DatasourceImportable):
                Required. Datasource for importing data to an existing dataset for Vertex AI.
            import_request_timeout (float):
                Optional. The timeout for the import request in seconds.

        Returns:
            operation (Operation):
                An object representing a long-running operation.
        """
        return self.api_client.import_data(
            name=self.resource_name,
            import_configs=[datasource.import_data_config],
            timeout=import_request_timeout,
        )

    @base.optional_sync(return_input_arg="self")
    def import_data(
        self,
        gcs_source: Union[str, Sequence[str]],
        import_schema_uri: str,
        data_item_labels: Optional[Dict] = None,
        sync: bool = True,
        import_request_timeout: Optional[float] = None,
    ) -> "_Dataset":
        """Upload data to existing managed dataset.

        Args:
            gcs_source (Union[str, Sequence[str]]):
                Required. Google Cloud Storage URI(-s) to the
                input file(s). May contain wildcards. For more
                information on wildcards, see
                https://cloud.google.com/storage/docs/gsutil/addlhelp/WildcardNames.
                examples:
                    str: "gs://bucket/file.csv"
                    Sequence[str]: ["gs://bucket/file1.csv", "gs://bucket/file2.csv"]
            import_schema_uri (str):
                Required. Points to a YAML file stored on Google Cloud
                Storage describing the import format. Validation will be
                done against the schema. The schema is defined as an
                `OpenAPI 3.0.2 Schema
                Object <https://tinyurl.com/y538mdwt>`__.
            data_item_labels (Dict):
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
                ``import_schema_uri``,
                e.g. jsonl file.
                This arg is not for specifying the annotation name or the
                training target of your data, but for some global labels of
                the dataset. E.g.,
                'data_item_labels={"aiplatform.googleapis.com/ml_use":"training"}'
                specifies that all the uploaded data are used for training.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            import_request_timeout (float):
                Optional. The timeout for the import request in seconds.

        Returns:
            dataset (Dataset):
                Instantiated representation of the managed dataset resource.
        """
        datasource = _datasources.create_datasource(
            metadata_schema_uri=self.metadata_schema_uri,
            import_schema_uri=import_schema_uri,
            gcs_source=gcs_source,
            data_item_labels=data_item_labels,
        )

        self._import_and_wait(
            datasource=datasource, import_request_timeout=import_request_timeout
        )
        return self

    def _validate_and_convert_export_split(
        self,
        split: Union[Dict[str, str], Dict[str, float]],
    ) -> Union[gca_dataset.ExportFilterSplit, gca_dataset.ExportFractionSplit]:
        """
        Validates the split for data export. Valid splits are dicts
        encoding the contents of proto messages ExportFilterSplit or
        ExportFractionSplit. If the split is valid, this function returns
        the corresponding convertered proto message.

        split (Union[Dict[str, str], Dict[str, float]]):
            The instructions how the export data should be split between the
            training, validation and test sets.
        """
        if len(split) != 3:
            raise ValueError(
                "The provided split for data export does not provide enough"
                "information. It must have three fields, mapping to training,"
                "validation and test splits respectively."
            )

        if not ("training_filter" in split or "training_fraction" in split):
            raise ValueError(
                "The provided filter for data export does not provide enough"
                "information. It must have three fields, mapping to training,"
                "validation and test respectively."
            )

        if "training_filter" in split:
            if (
                "validation_filter" in split
                and "test_filter" in split
                and isinstance(split["training_filter"], str)
                and isinstance(split["validation_filter"], str)
                and isinstance(split["test_filter"], str)
            ):
                return gca_dataset.ExportFilterSplit(
                    training_filter=split["training_filter"],
                    validation_filter=split["validation_filter"],
                    test_filter=split["test_filter"],
                )
            else:
                raise ValueError(
                    "The provided ExportFilterSplit does not contain all"
                    "three required fields: training_filter, "
                    "validation_filter and test_filter."
                )
        else:
            if (
                "validation_fraction" in split
                and "test_fraction" in split
                and isinstance(split["training_fraction"], float)
                and isinstance(split["validation_fraction"], float)
                and isinstance(split["test_fraction"], float)
            ):
                return gca_dataset.ExportFractionSplit(
                    training_fraction=split["training_fraction"],
                    validation_fraction=split["validation_fraction"],
                    test_fraction=split["test_fraction"],
                )
            else:
                raise ValueError(
                    "The provided ExportFractionSplit does not contain all"
                    "three required fields: training_fraction, "
                    "validation_fraction and test_fraction."
                )

    def _get_completed_export_data_operation(
        self,
        output_dir: str,
        export_use: Optional[gca_dataset.ExportDataConfig.ExportUse] = None,
        annotation_filter: Optional[str] = None,
        saved_query_id: Optional[str] = None,
        annotation_schema_uri: Optional[str] = None,
        split: Optional[
            Union[gca_dataset.ExportFilterSplit, gca_dataset.ExportFractionSplit]
        ] = None,
    ) -> gca_dataset_service.ExportDataResponse:
        self.wait()

        # TODO(b/171311614): Add support for BigQuery export path
        export_data_config = gca_dataset.ExportDataConfig(
            gcs_destination=gca_io.GcsDestination(output_uri_prefix=output_dir)
        )
        if export_use is not None:
            export_data_config.export_use = export_use
        if annotation_filter is not None:
            export_data_config.annotation_filter = annotation_filter
        if saved_query_id is not None:
            export_data_config.saved_query_id = saved_query_id
        if annotation_schema_uri is not None:
            export_data_config.annotation_schema_uri = annotation_schema_uri
        if split is not None:
            if isinstance(split, gca_dataset.ExportFilterSplit):
                export_data_config.filter_split = split
            elif isinstance(split, gca_dataset.ExportFractionSplit):
                export_data_config.fraction_split = split

        _LOGGER.log_action_start_against_resource("Exporting", "data", self)

        export_lro = self.api_client.export_data(
            name=self.resource_name, export_config=export_data_config
        )

        _LOGGER.log_action_started_against_resource_with_lro(
            "Export", "data", self.__class__, export_lro
        )

        export_data_response = export_lro.result()

        _LOGGER.log_action_completed_against_resource("data", "export", self)

        return export_data_response

    # TODO(b/174751568) add optional sync support
    def export_data(self, output_dir: str) -> Sequence[str]:
        """Exports data to output dir to GCS.

        Args:
            output_dir (str):
                Required. The Google Cloud Storage location where the output is to
                be written to. In the given directory a new directory will be
                created with name:
                ``export-data-<dataset-display-name>-<timestamp-of-export-call>``
                where timestamp is in YYYYMMDDHHMMSS format. All export
                output will be written into that directory. Inside that
                directory, annotations with the same schema will be grouped
                into sub directories which are named with the corresponding
                annotations' schema title. Inside these sub directories, a
                schema.yaml will be created to describe the output format.

                If the uri doesn't end with '/', a '/' will be automatically
                appended. The directory is created if it doesn't exist.

        Returns:
            exported_files (Sequence[str]):
                All of the files that are exported in this export operation.
        """
        return self._get_completed_export_data_operation(output_dir).exported_files

    def export_data_for_custom_training(
        self,
        output_dir: str,
        annotation_filter: Optional[str] = None,
        saved_query_id: Optional[str] = None,
        annotation_schema_uri: Optional[str] = None,
        split: Optional[Union[Dict[str, str], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Exports data to output dir to GCS for custom training use case.

           Example annotation_schema_uri (image classification):
           gs://google-cloud-aiplatform/schema/dataset/annotation/image_classification_1.0.0.yaml

           Example split (filter split):
           {
                "training_filter": "labels.aiplatform.googleapis.com/ml_use=training",
                "validation_filter": "labels.aiplatform.googleapis.com/ml_use=validation",
                "test_filter": "labels.aiplatform.googleapis.com/ml_use=test",
           }
           Example split (fraction split):
           {
                "training_fraction": 0.7,
                "validation_fraction": 0.2,
                "test_fraction": 0.1,
           }

        Args:
            output_dir (str):
                Required. The Google Cloud Storage location where the output is to
                be written to. In the given directory a new directory will be
                created with name:
                ``export-data-<dataset-display-name>-<timestamp-of-export-call>``
                where timestamp is in YYYYMMDDHHMMSS format. All export
                output will be written into that directory. Inside that
                directory, annotations with the same schema will be grouped
                into sub directories which are named with the corresponding
                annotations' schema title. Inside these sub directories, a
                schema.yaml will be created to describe the output format.

                If the uri doesn't end with '/', a '/' will be automatically
                appended. The directory is created if it doesn't exist.
            annotation_filter (str):
                Optional. An expression for filtering what part of the Dataset
                is to be exported.
                Only Annotations that match this filter will be exported.
                The filter syntax is the same as in
                [ListAnnotations][DatasetService.ListAnnotations].
            saved_query_id (str):
                Optional. The ID of a SavedQuery (annotation set) under this
                Dataset used for filtering Annotations for training.

                Only used for custom training data export use cases.
                Only applicable to Datasets that have SavedQueries.

                Only Annotations that are associated with this SavedQuery are
                used in respectively training. When used in conjunction with
                annotations_filter, the Annotations used for training are
                filtered by both saved_query_id and annotations_filter.

                Only one of saved_query_id and annotation_schema_uri should be
                specified as both of them represent the same thing: problem
                type.
            annotation_schema_uri (str):
                Optional. The Cloud Storage URI that points to a YAML file
                describing the annotation schema. The schema is defined as an
                OpenAPI 3.0.2 Schema Object. The schema files that can be used
                here are found in
                gs://google-cloud-aiplatform/schema/dataset/annotation/, note
                that the chosen schema must be consistent with
                metadata_schema_uri of this Dataset.

                Only used for custom training data export use cases.
                Only applicable if this Dataset that have DataItems and
                Annotations.

                Only Annotations that both match this schema and belong to
                DataItems not ignored by the split method are used in
                respectively training, validation or test role, depending on the
                role of the DataItem they are on.

                When used in conjunction with annotations_filter, the
                Annotations used for training are filtered by both
                annotations_filter and annotation_schema_uri.
            split (Union[Dict[str, str], Dict[str, float]]):
                The instructions how the export data should be split between the
                training, validation and test sets.

        Returns:
            export_data_response (Dict):
                Response message for DatasetService.ExportData in Dictionary
                format.
        """
        split = self._validate_and_convert_export_split(split)

        return json_format.MessageToDict(
            self._get_completed_export_data_operation(
                output_dir,
                gca_dataset.ExportDataConfig.ExportUse.CUSTOM_CODE_TRAINING,
                annotation_filter,
                saved_query_id,
                annotation_schema_uri,
                split,
            )._pb
        )

    def update(
        self,
        *,
        display_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        update_request_timeout: Optional[float] = None,
    ) -> "_Dataset":
        """Update the dataset.
            Updatable fields:
                -  ``display_name``
                -  ``description``
                -  ``labels``

        Args:
            display_name (str):
                Optional. The user-defined name of the Dataset.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your Tensorboards.
                Label keys and values can be no longer than 64 characters
                (Unicode codepoints), can only contain lowercase letters, numeric
                characters, underscores and dashes. International characters are allowed.
                No more than 64 user labels can be associated with one Tensorboard
                (System labels are excluded).
                See https://goo.gl/xmQnxf for more information and examples of labels.
                System reserved label keys are prefixed with "aiplatform.googleapis.com/"
                and are immutable.
            description (str):
                Optional. The description of the Dataset.
            update_request_timeout (float):
                Optional. The timeout for the update request in seconds.

        Returns:
            dataset (Dataset):
                Updated dataset.
        """

        update_mask = field_mask_pb2.FieldMask()
        if display_name:
            update_mask.paths.append("display_name")

        if labels:
            update_mask.paths.append("labels")

        if description:
            update_mask.paths.append("description")

        update_dataset = gca_dataset.Dataset(
            name=self.resource_name,
            display_name=display_name,
            description=description,
            labels=labels,
        )

        self._gca_resource = self.api_client.update_dataset(
            dataset=update_dataset,
            update_mask=update_mask,
            timeout=update_request_timeout,
        )

        return self

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List[base.VertexAiResourceNoun]:
        """List all instances of this Dataset resource.

        Example Usage:

        aiplatform.TabularDataset.list(
            filter='labels.my_key="my_value"',
            order_by='display_name'
        )

        Args:
            filter (str):
                Optional. An expression for filtering the results of the request.
                For field names both snake_case and camelCase are supported.
            order_by (str):
                Optional. A comma-separated list of fields to order by, sorted in
                ascending order. Use "desc" after a field name for descending.
                Supported fields: `display_name`, `create_time`, `update_time`
            project (str):
                Optional. Project to retrieve list from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve list from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve list. Overrides
                credentials set in aiplatform.init.

        Returns:
            List[base.VertexAiResourceNoun] - A list of Dataset resource objects
        """

        dataset_subclass_filter = (
            lambda gapic_obj: gapic_obj.metadata_schema_uri
            in cls._supported_metadata_schema_uris
        )

        return cls._list_with_local_order(
            cls_filter=dataset_subclass_filter,
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
        )
