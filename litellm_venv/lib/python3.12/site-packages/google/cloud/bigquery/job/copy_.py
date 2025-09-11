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

"""Classes for copy jobs."""

import typing
from typing import Optional

from google.cloud.bigquery.encryption_configuration import EncryptionConfiguration
from google.cloud.bigquery import _helpers
from google.cloud.bigquery.table import TableReference

from google.cloud.bigquery.job.base import _AsyncJob
from google.cloud.bigquery.job.base import _JobConfig
from google.cloud.bigquery.job.base import _JobReference


class OperationType:
    """Different operation types supported in table copy job.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#operationtype
    """

    OPERATION_TYPE_UNSPECIFIED = "OPERATION_TYPE_UNSPECIFIED"
    """Unspecified operation type."""

    COPY = "COPY"
    """The source and destination table have the same table type."""

    SNAPSHOT = "SNAPSHOT"
    """The source table type is TABLE and the destination table type is SNAPSHOT."""

    CLONE = "CLONE"
    """The source table type is TABLE and the destination table type is CLONE."""

    RESTORE = "RESTORE"
    """The source table type is SNAPSHOT and the destination table type is TABLE."""


class CopyJobConfig(_JobConfig):
    """Configuration options for copy jobs.

    All properties in this class are optional. Values which are :data:`None` ->
    server defaults. Set properties on the constructed configuration by using
    the property name as the name of a keyword argument.
    """

    def __init__(self, **kwargs) -> None:
        super(CopyJobConfig, self).__init__("copy", **kwargs)

    @property
    def create_disposition(self):
        """google.cloud.bigquery.job.CreateDisposition: Specifies behavior
        for creating tables.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationTableCopy.FIELDS.create_disposition
        """
        return self._get_sub_prop("createDisposition")

    @create_disposition.setter
    def create_disposition(self, value):
        self._set_sub_prop("createDisposition", value)

    @property
    def write_disposition(self):
        """google.cloud.bigquery.job.WriteDisposition: Action that occurs if
        the destination table already exists.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationTableCopy.FIELDS.write_disposition
        """
        return self._get_sub_prop("writeDisposition")

    @write_disposition.setter
    def write_disposition(self, value):
        self._set_sub_prop("writeDisposition", value)

    @property
    def destination_encryption_configuration(self):
        """google.cloud.bigquery.encryption_configuration.EncryptionConfiguration: Custom
        encryption configuration for the destination table.

        Custom encryption configuration (e.g., Cloud KMS keys) or :data:`None`
        if using default encryption.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationTableCopy.FIELDS.destination_encryption_configuration
        """
        prop = self._get_sub_prop("destinationEncryptionConfiguration")
        if prop is not None:
            prop = EncryptionConfiguration.from_api_repr(prop)
        return prop

    @destination_encryption_configuration.setter
    def destination_encryption_configuration(self, value):
        api_repr = value
        if value is not None:
            api_repr = value.to_api_repr()
        self._set_sub_prop("destinationEncryptionConfiguration", api_repr)

    @property
    def operation_type(self) -> str:
        """The operation to perform with this copy job.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationTableCopy.FIELDS.operation_type
        """
        return self._get_sub_prop(
            "operationType", OperationType.OPERATION_TYPE_UNSPECIFIED
        )

    @operation_type.setter
    def operation_type(self, value: Optional[str]):
        if value is None:
            value = OperationType.OPERATION_TYPE_UNSPECIFIED
        self._set_sub_prop("operationType", value)

    @property
    def destination_expiration_time(self) -> str:
        """google.cloud.bigquery.job.DestinationExpirationTime: The time when the
        destination table expires. Expired tables will be deleted and their storage reclaimed.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationTableCopy.FIELDS.destination_expiration_time
        """
        return self._get_sub_prop("destinationExpirationTime")

    @destination_expiration_time.setter
    def destination_expiration_time(self, value: str):
        self._set_sub_prop("destinationExpirationTime", value)


class CopyJob(_AsyncJob):
    """Asynchronous job: copy data into a table from other tables.

    Args:
        job_id (str): the job's ID, within the project belonging to ``client``.

        sources (List[google.cloud.bigquery.table.TableReference]): Table from which data is to be loaded.

        destination (google.cloud.bigquery.table.TableReference): Table into which data is to be loaded.

        client (google.cloud.bigquery.client.Client):
            A client which holds credentials and project configuration
            for the dataset (which requires a project).

        job_config (Optional[google.cloud.bigquery.job.CopyJobConfig]):
            Extra configuration options for the copy job.
    """

    _JOB_TYPE = "copy"
    _CONFIG_CLASS = CopyJobConfig

    def __init__(self, job_id, sources, destination, client, job_config=None):
        super(CopyJob, self).__init__(job_id, client)

        if job_config is not None:
            self._properties["configuration"] = job_config._properties

        if destination:
            _helpers._set_sub_prop(
                self._properties,
                ["configuration", "copy", "destinationTable"],
                destination.to_api_repr(),
            )

        if sources:
            source_resources = [source.to_api_repr() for source in sources]
            _helpers._set_sub_prop(
                self._properties,
                ["configuration", "copy", "sourceTables"],
                source_resources,
            )

    @property
    def configuration(self) -> CopyJobConfig:
        """The configuration for this copy job."""
        return typing.cast(CopyJobConfig, super().configuration)

    @property
    def destination(self):
        """google.cloud.bigquery.table.TableReference: Table into which data
        is to be loaded.
        """
        return TableReference.from_api_repr(
            _helpers._get_sub_prop(
                self._properties, ["configuration", "copy", "destinationTable"]
            )
        )

    @property
    def sources(self):
        """List[google.cloud.bigquery.table.TableReference]): Table(s) from
        which data is to be loaded.
        """
        source_configs = _helpers._get_sub_prop(
            self._properties, ["configuration", "copy", "sourceTables"]
        )
        if source_configs is None:
            single = _helpers._get_sub_prop(
                self._properties, ["configuration", "copy", "sourceTable"]
            )
            if single is None:
                raise KeyError("Resource missing 'sourceTables' / 'sourceTable'")
            source_configs = [single]

        sources = []
        for source_config in source_configs:
            table_ref = TableReference.from_api_repr(source_config)
            sources.append(table_ref)
        return sources

    @property
    def create_disposition(self):
        """See
        :attr:`google.cloud.bigquery.job.CopyJobConfig.create_disposition`.
        """
        return self.configuration.create_disposition

    @property
    def write_disposition(self):
        """See
        :attr:`google.cloud.bigquery.job.CopyJobConfig.write_disposition`.
        """
        return self.configuration.write_disposition

    @property
    def destination_encryption_configuration(self):
        """google.cloud.bigquery.encryption_configuration.EncryptionConfiguration: Custom
        encryption configuration for the destination table.

        Custom encryption configuration (e.g., Cloud KMS keys) or :data:`None`
        if using default encryption.

        See
        :attr:`google.cloud.bigquery.job.CopyJobConfig.destination_encryption_configuration`.
        """
        return self.configuration.destination_encryption_configuration

    def to_api_repr(self):
        """Generate a resource for :meth:`_begin`."""
        # Exclude statistics, if set.
        return {
            "jobReference": self._properties["jobReference"],
            "configuration": self._properties["configuration"],
        }

    @classmethod
    def from_api_repr(cls, resource, client):
        """Factory: construct a job given its API representation

        .. note::

           This method assumes that the project found in the resource matches
           the client's project.

        Args:
            resource (Dict): dataset job representation returned from the API
            client (google.cloud.bigquery.client.Client):
                Client which holds credentials and project
                configuration for the dataset.

        Returns:
            google.cloud.bigquery.job.CopyJob: Job parsed from ``resource``.
        """
        cls._check_resource_config(resource)
        job_ref = _JobReference._from_api_repr(resource["jobReference"])
        job = cls(job_ref, None, None, client=client)
        job._set_properties(resource)
        return job
