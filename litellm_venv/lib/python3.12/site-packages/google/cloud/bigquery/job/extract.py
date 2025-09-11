# Copyright 2015 Google LLC
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

"""Classes for extract (export) jobs."""

import typing

from google.cloud.bigquery import _helpers
from google.cloud.bigquery.model import ModelReference
from google.cloud.bigquery.table import Table
from google.cloud.bigquery.table import TableListItem
from google.cloud.bigquery.table import TableReference
from google.cloud.bigquery.job.base import _AsyncJob
from google.cloud.bigquery.job.base import _JobConfig
from google.cloud.bigquery.job.base import _JobReference


class ExtractJobConfig(_JobConfig):
    """Configuration options for extract jobs.

    All properties in this class are optional. Values which are :data:`None` ->
    server defaults. Set properties on the constructed configuration by using
    the property name as the name of a keyword argument.
    """

    def __init__(self, **kwargs):
        super(ExtractJobConfig, self).__init__("extract", **kwargs)

    @property
    def compression(self):
        """google.cloud.bigquery.job.Compression: Compression type to use for
        exported files.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationExtract.FIELDS.compression
        """
        return self._get_sub_prop("compression")

    @compression.setter
    def compression(self, value):
        self._set_sub_prop("compression", value)

    @property
    def destination_format(self):
        """google.cloud.bigquery.job.DestinationFormat: Exported file format.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationExtract.FIELDS.destination_format
        """
        return self._get_sub_prop("destinationFormat")

    @destination_format.setter
    def destination_format(self, value):
        self._set_sub_prop("destinationFormat", value)

    @property
    def field_delimiter(self):
        """str: Delimiter to use between fields in the exported data.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationExtract.FIELDS.field_delimiter
        """
        return self._get_sub_prop("fieldDelimiter")

    @field_delimiter.setter
    def field_delimiter(self, value):
        self._set_sub_prop("fieldDelimiter", value)

    @property
    def print_header(self):
        """bool: Print a header row in the exported data.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationExtract.FIELDS.print_header
        """
        return self._get_sub_prop("printHeader")

    @print_header.setter
    def print_header(self, value):
        self._set_sub_prop("printHeader", value)

    @property
    def use_avro_logical_types(self):
        """bool: For loads of Avro data, governs whether Avro logical types are
        converted to their corresponding BigQuery types (e.g. TIMESTAMP) rather than
        raw types (e.g. INTEGER).
        """
        return self._get_sub_prop("useAvroLogicalTypes")

    @use_avro_logical_types.setter
    def use_avro_logical_types(self, value):
        self._set_sub_prop("useAvroLogicalTypes", bool(value))


class ExtractJob(_AsyncJob):
    """Asynchronous job: extract data from a table into Cloud Storage.

    Args:
        job_id (str): the job's ID.

        source (Union[ \
            google.cloud.bigquery.table.TableReference, \
            google.cloud.bigquery.model.ModelReference \
        ]):
            Table or Model from which data is to be loaded or extracted.

        destination_uris (List[str]):
            URIs describing where the extracted data will be written in Cloud
            Storage, using the format ``gs://<bucket_name>/<object_name_or_glob>``.

        client (google.cloud.bigquery.client.Client):
            A client which holds credentials and project configuration.

        job_config (Optional[google.cloud.bigquery.job.ExtractJobConfig]):
            Extra configuration options for the extract job.
    """

    _JOB_TYPE = "extract"
    _CONFIG_CLASS = ExtractJobConfig

    def __init__(self, job_id, source, destination_uris, client, job_config=None):
        super(ExtractJob, self).__init__(job_id, client)

        if job_config is not None:
            self._properties["configuration"] = job_config._properties

        if source:
            source_ref = {"projectId": source.project, "datasetId": source.dataset_id}

            if isinstance(source, (Table, TableListItem, TableReference)):
                source_ref["tableId"] = source.table_id
                source_key = "sourceTable"
            else:
                source_ref["modelId"] = source.model_id
                source_key = "sourceModel"

            _helpers._set_sub_prop(
                self._properties, ["configuration", "extract", source_key], source_ref
            )

        if destination_uris:
            _helpers._set_sub_prop(
                self._properties,
                ["configuration", "extract", "destinationUris"],
                destination_uris,
            )

    @property
    def configuration(self) -> ExtractJobConfig:
        """The configuration for this extract job."""
        return typing.cast(ExtractJobConfig, super().configuration)

    @property
    def source(self):
        """Union[ \
            google.cloud.bigquery.table.TableReference, \
            google.cloud.bigquery.model.ModelReference \
        ]: Table or Model from which data is to be loaded or extracted.
        """
        source_config = _helpers._get_sub_prop(
            self._properties, ["configuration", "extract", "sourceTable"]
        )
        if source_config:
            return TableReference.from_api_repr(source_config)
        else:
            source_config = _helpers._get_sub_prop(
                self._properties, ["configuration", "extract", "sourceModel"]
            )
            return ModelReference.from_api_repr(source_config)

    @property
    def destination_uris(self):
        """List[str]: URIs describing where the extracted data will be
        written in Cloud Storage, using the format
        ``gs://<bucket_name>/<object_name_or_glob>``.
        """
        return _helpers._get_sub_prop(
            self._properties, ["configuration", "extract", "destinationUris"]
        )

    @property
    def compression(self):
        """See
        :attr:`google.cloud.bigquery.job.ExtractJobConfig.compression`.
        """
        return self.configuration.compression

    @property
    def destination_format(self):
        """See
        :attr:`google.cloud.bigquery.job.ExtractJobConfig.destination_format`.
        """
        return self.configuration.destination_format

    @property
    def field_delimiter(self):
        """See
        :attr:`google.cloud.bigquery.job.ExtractJobConfig.field_delimiter`.
        """
        return self.configuration.field_delimiter

    @property
    def print_header(self):
        """See
        :attr:`google.cloud.bigquery.job.ExtractJobConfig.print_header`.
        """
        return self.configuration.print_header

    @property
    def destination_uri_file_counts(self):
        """Return file counts from job statistics, if present.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics4.FIELDS.destination_uri_file_counts

        Returns:
            List[int]:
                A list of integer counts, each representing the number of files
                per destination URI or URI pattern specified in the extract
                configuration. These values will be in the same order as the URIs
                specified in the 'destinationUris' field.  Returns None if job is
                not yet complete.
        """
        counts = self._job_statistics().get("destinationUriFileCounts")
        if counts is not None:
            return [int(count) for count in counts]
        return None

    def to_api_repr(self):
        """Generate a resource for :meth:`_begin`."""
        # Exclude statistics, if set.
        return {
            "jobReference": self._properties["jobReference"],
            "configuration": self._properties["configuration"],
        }

    @classmethod
    def from_api_repr(cls, resource: dict, client) -> "ExtractJob":
        """Factory:  construct a job given its API representation

        .. note::

           This method assumes that the project found in the resource matches
           the client's project.

        Args:
            resource (Dict): dataset job representation returned from the API

            client (google.cloud.bigquery.client.Client):
                Client which holds credentials and project
                configuration for the dataset.

        Returns:
            google.cloud.bigquery.job.ExtractJob: Job parsed from ``resource``.
        """
        cls._check_resource_config(resource)
        job_ref = _JobReference._from_api_repr(resource["jobReference"])
        job = cls(job_ref, None, None, client=client)
        job._set_properties(resource)
        return job
