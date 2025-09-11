# -*- coding: utf-8 -*-
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define resources for the BigQuery Routines API."""

from typing import Any, Dict, Optional, Union

import google.cloud._helpers  # type: ignore
from google.cloud.bigquery import _helpers
from google.cloud.bigquery.standard_sql import StandardSqlDataType
from google.cloud.bigquery.standard_sql import StandardSqlTableType


class RoutineType:
    """The fine-grained type of the routine.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/routines#routinetype

    .. versionadded:: 2.22.0
    """

    ROUTINE_TYPE_UNSPECIFIED = "ROUTINE_TYPE_UNSPECIFIED"
    SCALAR_FUNCTION = "SCALAR_FUNCTION"
    PROCEDURE = "PROCEDURE"
    TABLE_VALUED_FUNCTION = "TABLE_VALUED_FUNCTION"


class Routine(object):
    """Resource representing a user-defined routine.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/routines

    Args:
        routine_ref (Union[str, google.cloud.bigquery.routine.RoutineReference]):
            A pointer to a routine. If ``routine_ref`` is a string, it must
            included a project ID, dataset ID, and routine ID, each separated
            by ``.``.
        ``**kwargs`` (Dict):
            Initial property values.
    """

    _PROPERTY_TO_API_FIELD = {
        "arguments": "arguments",
        "body": "definitionBody",
        "created": "creationTime",
        "etag": "etag",
        "imported_libraries": "importedLibraries",
        "language": "language",
        "modified": "lastModifiedTime",
        "reference": "routineReference",
        "return_type": "returnType",
        "return_table_type": "returnTableType",
        "type_": "routineType",
        "description": "description",
        "determinism_level": "determinismLevel",
        "remote_function_options": "remoteFunctionOptions",
        "data_governance_type": "dataGovernanceType",
    }

    def __init__(self, routine_ref, **kwargs) -> None:
        if isinstance(routine_ref, str):
            routine_ref = RoutineReference.from_string(routine_ref)

        self._properties = {"routineReference": routine_ref.to_api_repr()}
        for property_name in kwargs:
            setattr(self, property_name, kwargs[property_name])

    @property
    def reference(self):
        """google.cloud.bigquery.routine.RoutineReference: Reference
        describing the ID of this routine.
        """
        return RoutineReference.from_api_repr(
            self._properties[self._PROPERTY_TO_API_FIELD["reference"]]
        )

    @property
    def path(self):
        """str: URL path for the routine's APIs."""
        return self.reference.path

    @property
    def project(self):
        """str: ID of the project containing the routine."""
        return self.reference.project

    @property
    def dataset_id(self):
        """str: ID of dataset containing the routine."""
        return self.reference.dataset_id

    @property
    def routine_id(self):
        """str: The routine ID."""
        return self.reference.routine_id

    @property
    def etag(self):
        """str: ETag for the resource (:data:`None` until set from the
        server).

        Read-only.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["etag"])

    @property
    def type_(self):
        """str: The fine-grained type of the routine.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/routines#RoutineType
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["type_"])

    @type_.setter
    def type_(self, value):
        self._properties[self._PROPERTY_TO_API_FIELD["type_"]] = value

    @property
    def created(self):
        """Optional[datetime.datetime]: Datetime at which the routine was
        created (:data:`None` until set from the server).

        Read-only.
        """
        value = self._properties.get(self._PROPERTY_TO_API_FIELD["created"])
        if value is not None and value != 0:
            # value will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(value)
            )

    @property
    def modified(self):
        """Optional[datetime.datetime]: Datetime at which the routine was
        last modified (:data:`None` until set from the server).

        Read-only.
        """
        value = self._properties.get(self._PROPERTY_TO_API_FIELD["modified"])
        if value is not None and value != 0:
            # value will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(value)
            )

    @property
    def language(self):
        """Optional[str]: The language of the routine.

        Defaults to ``SQL``.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["language"])

    @language.setter
    def language(self, value):
        self._properties[self._PROPERTY_TO_API_FIELD["language"]] = value

    @property
    def arguments(self):
        """List[google.cloud.bigquery.routine.RoutineArgument]: Input/output
        argument of a function or a stored procedure.

        In-place modification is not supported. To set, replace the entire
        property value with the modified list of
        :class:`~google.cloud.bigquery.routine.RoutineArgument` objects.
        """
        resources = self._properties.get(self._PROPERTY_TO_API_FIELD["arguments"], [])
        return [RoutineArgument.from_api_repr(resource) for resource in resources]

    @arguments.setter
    def arguments(self, value):
        if not value:
            resource = []
        else:
            resource = [argument.to_api_repr() for argument in value]
        self._properties[self._PROPERTY_TO_API_FIELD["arguments"]] = resource

    @property
    def return_type(self):
        """google.cloud.bigquery.StandardSqlDataType: Return type of
        the routine.

        If absent, the return type is inferred from
        :attr:`~google.cloud.bigquery.routine.Routine.body` at query time in
        each query that references this routine. If present, then the
        evaluated result will be cast to the specified returned type at query
        time.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/routines#Routine.FIELDS.return_type
        """
        resource = self._properties.get(self._PROPERTY_TO_API_FIELD["return_type"])
        if not resource:
            return resource

        return StandardSqlDataType.from_api_repr(resource)

    @return_type.setter
    def return_type(self, value: StandardSqlDataType):
        resource = None if not value else value.to_api_repr()
        self._properties[self._PROPERTY_TO_API_FIELD["return_type"]] = resource

    @property
    def return_table_type(self) -> Union[StandardSqlTableType, Any, None]:
        """The return type of a Table Valued Function (TVF) routine.

        .. versionadded:: 2.22.0
        """
        resource = self._properties.get(
            self._PROPERTY_TO_API_FIELD["return_table_type"]
        )
        if not resource:
            return resource

        return StandardSqlTableType.from_api_repr(resource)

    @return_table_type.setter
    def return_table_type(self, value: Optional[StandardSqlTableType]):
        if not value:
            resource = None
        else:
            resource = value.to_api_repr()

        self._properties[self._PROPERTY_TO_API_FIELD["return_table_type"]] = resource

    @property
    def imported_libraries(self):
        """List[str]: The path of the imported JavaScript libraries.

        The :attr:`~google.cloud.bigquery.routine.Routine.language` must
        equal ``JAVACRIPT``.

        Examples:
            Set the ``imported_libraries`` to a list of Google Cloud Storage
            URIs.

            .. code-block:: python

               routine = bigquery.Routine("proj.dataset.routine_id")
               routine.imported_libraries = [
                   "gs://cloud-samples-data/bigquery/udfs/max-value.js",
               ]
        """
        return self._properties.get(
            self._PROPERTY_TO_API_FIELD["imported_libraries"], []
        )

    @imported_libraries.setter
    def imported_libraries(self, value):
        if not value:
            resource = []
        else:
            resource = value
        self._properties[self._PROPERTY_TO_API_FIELD["imported_libraries"]] = resource

    @property
    def body(self):
        """str: The body of the routine."""
        return self._properties.get(self._PROPERTY_TO_API_FIELD["body"])

    @body.setter
    def body(self, value):
        self._properties[self._PROPERTY_TO_API_FIELD["body"]] = value

    @property
    def description(self):
        """Optional[str]: Description of the routine (defaults to
        :data:`None`).
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["description"])

    @description.setter
    def description(self, value):
        self._properties[self._PROPERTY_TO_API_FIELD["description"]] = value

    @property
    def determinism_level(self):
        """Optional[str]: (experimental) The determinism level of the JavaScript UDF
        if defined.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["determinism_level"])

    @determinism_level.setter
    def determinism_level(self, value):
        self._properties[self._PROPERTY_TO_API_FIELD["determinism_level"]] = value

    @property
    def remote_function_options(self):
        """Optional[google.cloud.bigquery.routine.RemoteFunctionOptions]:
        Configures remote function options for a routine.

        Raises:
            ValueError:
                If the value is not
                :class:`~google.cloud.bigquery.routine.RemoteFunctionOptions` or
                :data:`None`.
        """
        prop = self._properties.get(
            self._PROPERTY_TO_API_FIELD["remote_function_options"]
        )
        if prop is not None:
            return RemoteFunctionOptions.from_api_repr(prop)

    @remote_function_options.setter
    def remote_function_options(self, value):
        api_repr = value
        if isinstance(value, RemoteFunctionOptions):
            api_repr = value.to_api_repr()
        elif value is not None:
            raise ValueError(
                "value must be google.cloud.bigquery.routine.RemoteFunctionOptions "
                "or None"
            )
        self._properties[
            self._PROPERTY_TO_API_FIELD["remote_function_options"]
        ] = api_repr

    @property
    def data_governance_type(self):
        """Optional[str]: If set to ``DATA_MASKING``, the function is validated
        and made available as a masking function.

        Raises:
            ValueError:
                If the value is not :data:`string` or :data:`None`.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["data_governance_type"])

    @data_governance_type.setter
    def data_governance_type(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError(
                "invalid data_governance_type, must be a string or `None`."
            )
        self._properties[self._PROPERTY_TO_API_FIELD["data_governance_type"]] = value

    @classmethod
    def from_api_repr(cls, resource: dict) -> "Routine":
        """Factory: construct a routine given its API representation.

        Args:
            resource (Dict[str, object]):
                Resource, as returned from the API.

        Returns:
            google.cloud.bigquery.routine.Routine:
                Python object, as parsed from ``resource``.
        """
        ref = cls(RoutineReference.from_api_repr(resource["routineReference"]))
        ref._properties = resource
        return ref

    def to_api_repr(self) -> dict:
        """Construct the API resource representation of this routine.

        Returns:
            Dict[str, object]: Routine represented as an API resource.
        """
        return self._properties

    def _build_resource(self, filter_fields):
        """Generate a resource for ``update``."""
        return _helpers._build_resource_from_properties(self, filter_fields)

    def __repr__(self):
        return "Routine('{}.{}.{}')".format(
            self.project, self.dataset_id, self.routine_id
        )


class RoutineArgument(object):
    """Input/output argument of a function or a stored procedure.

    See:
    https://cloud.google.com/bigquery/docs/reference/rest/v2/routines#argument

    Args:
        ``**kwargs`` (Dict):
            Initial property values.
    """

    _PROPERTY_TO_API_FIELD = {
        "data_type": "dataType",
        "kind": "argumentKind",
        # Even though it's not necessary for field mapping to map when the
        # property name equals the resource name, we add these here so that we
        # have an exhaustive list of all properties.
        "name": "name",
        "mode": "mode",
    }

    def __init__(self, **kwargs) -> None:
        self._properties: Dict[str, Any] = {}
        for property_name in kwargs:
            setattr(self, property_name, kwargs[property_name])

    @property
    def name(self):
        """Optional[str]: Name of this argument.

        Can be absent for function return argument.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["name"])

    @name.setter
    def name(self, value):
        self._properties[self._PROPERTY_TO_API_FIELD["name"]] = value

    @property
    def kind(self):
        """Optional[str]: The kind of argument, for example ``FIXED_TYPE`` or
        ``ANY_TYPE``.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/routines#Argument.FIELDS.argument_kind
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["kind"])

    @kind.setter
    def kind(self, value):
        self._properties[self._PROPERTY_TO_API_FIELD["kind"]] = value

    @property
    def mode(self):
        """Optional[str]: The input/output mode of the argument."""
        return self._properties.get(self._PROPERTY_TO_API_FIELD["mode"])

    @mode.setter
    def mode(self, value):
        self._properties[self._PROPERTY_TO_API_FIELD["mode"]] = value

    @property
    def data_type(self):
        """Optional[google.cloud.bigquery.StandardSqlDataType]: Type
        of a variable, e.g., a function argument.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/routines#Argument.FIELDS.data_type
        """
        resource = self._properties.get(self._PROPERTY_TO_API_FIELD["data_type"])
        if not resource:
            return resource

        return StandardSqlDataType.from_api_repr(resource)

    @data_type.setter
    def data_type(self, value):
        if value:
            resource = value.to_api_repr()
        else:
            resource = None
        self._properties[self._PROPERTY_TO_API_FIELD["data_type"]] = resource

    @classmethod
    def from_api_repr(cls, resource: dict) -> "RoutineArgument":
        """Factory: construct a routine argument given its API representation.

        Args:
            resource (Dict[str, object]): Resource, as returned from the API.

        Returns:
            google.cloud.bigquery.routine.RoutineArgument:
                Python object, as parsed from ``resource``.
        """
        ref = cls()
        ref._properties = resource
        return ref

    def to_api_repr(self) -> dict:
        """Construct the API resource representation of this routine argument.

        Returns:
            Dict[str, object]: Routine argument represented as an API resource.
        """
        return self._properties

    def __eq__(self, other):
        if not isinstance(other, RoutineArgument):
            return NotImplemented
        return self._properties == other._properties

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        all_properties = [
            "{}={}".format(property_name, repr(getattr(self, property_name)))
            for property_name in sorted(self._PROPERTY_TO_API_FIELD)
        ]
        return "RoutineArgument({})".format(", ".join(all_properties))


class RoutineReference(object):
    """A pointer to a routine.

    See:
    https://cloud.google.com/bigquery/docs/reference/rest/v2/routines#routinereference
    """

    def __init__(self):
        self._properties = {}

    @property
    def project(self):
        """str: ID of the project containing the routine."""
        return self._properties.get("projectId", "")

    @property
    def dataset_id(self):
        """str: ID of dataset containing the routine."""
        return self._properties.get("datasetId", "")

    @property
    def routine_id(self):
        """str: The routine ID."""
        return self._properties.get("routineId", "")

    @property
    def path(self):
        """str: URL path for the routine's APIs."""
        return "/projects/%s/datasets/%s/routines/%s" % (
            self.project,
            self.dataset_id,
            self.routine_id,
        )

    @classmethod
    def from_api_repr(cls, resource: dict) -> "RoutineReference":
        """Factory: construct a routine reference given its API representation.

        Args:
            resource (Dict[str, object]):
                Routine reference representation returned from the API.

        Returns:
            google.cloud.bigquery.routine.RoutineReference:
                Routine reference parsed from ``resource``.
        """
        ref = cls()
        ref._properties = resource
        return ref

    @classmethod
    def from_string(
        cls, routine_id: str, default_project: Optional[str] = None
    ) -> "RoutineReference":
        """Factory: construct a routine reference from routine ID string.

        Args:
            routine_id (str):
                A routine ID in standard SQL format. If ``default_project``
                is not specified, this must included a project ID, dataset
                ID, and routine ID, each separated by ``.``.
            default_project (Optional[str]):
                The project ID to use when ``routine_id`` does not
                include a project ID.

        Returns:
            google.cloud.bigquery.routine.RoutineReference:
                Routine reference parsed from ``routine_id``.

        Raises:
            ValueError:
                If ``routine_id`` is not a fully-qualified routine ID in
                standard SQL format.
        """
        proj, dset, routine = _helpers._parse_3_part_id(
            routine_id, default_project=default_project, property_name="routine_id"
        )
        return cls.from_api_repr(
            {"projectId": proj, "datasetId": dset, "routineId": routine}
        )

    def to_api_repr(self) -> dict:
        """Construct the API resource representation of this routine reference.

        Returns:
            Dict[str, object]: Routine reference represented as an API resource.
        """
        return self._properties

    def __eq__(self, other):
        """Two RoutineReferences are equal if they point to the same routine."""
        if not isinstance(other, RoutineReference):
            return NotImplemented
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "RoutineReference.from_string('{}')".format(str(self))

    def __str__(self):
        """String representation of the reference.

        This is a fully-qualified ID, including the project ID and dataset ID.
        """
        return "{}.{}.{}".format(self.project, self.dataset_id, self.routine_id)


class RemoteFunctionOptions(object):
    """Configuration options for controlling remote BigQuery functions."""

    _PROPERTY_TO_API_FIELD = {
        "endpoint": "endpoint",
        "connection": "connection",
        "max_batching_rows": "maxBatchingRows",
        "user_defined_context": "userDefinedContext",
    }

    def __init__(
        self,
        endpoint=None,
        connection=None,
        max_batching_rows=None,
        user_defined_context=None,
        _properties=None,
    ) -> None:
        if _properties is None:
            _properties = {}
        self._properties = _properties

        if endpoint is not None:
            self.endpoint = endpoint
        if connection is not None:
            self.connection = connection
        if max_batching_rows is not None:
            self.max_batching_rows = max_batching_rows
        if user_defined_context is not None:
            self.user_defined_context = user_defined_context

    @property
    def connection(self):
        """string: Fully qualified name of the user-provided connection object which holds the authentication information to send requests to the remote service.

        Format is  "projects/{projectId}/locations/{locationId}/connections/{connectionId}"
        """
        return _helpers._str_or_none(self._properties.get("connection"))

    @connection.setter
    def connection(self, value):
        self._properties["connection"] = _helpers._str_or_none(value)

    @property
    def endpoint(self):
        """string: Endpoint of the user-provided remote service

        Example: "https://us-east1-my_gcf_project.cloudfunctions.net/remote_add"
        """
        return _helpers._str_or_none(self._properties.get("endpoint"))

    @endpoint.setter
    def endpoint(self, value):
        self._properties["endpoint"] = _helpers._str_or_none(value)

    @property
    def max_batching_rows(self):
        """int64: Max number of rows in each batch sent to the remote service.

        If absent or if 0, BigQuery dynamically decides the number of rows in a batch.
        """
        return _helpers._int_or_none(self._properties.get("maxBatchingRows"))

    @max_batching_rows.setter
    def max_batching_rows(self, value):
        self._properties["maxBatchingRows"] = _helpers._str_or_none(value)

    @property
    def user_defined_context(self):
        """Dict[str, str]: User-defined context as a set of key/value pairs,
            which will be sent as function invocation context together with
        batched arguments in the requests to the remote service. The total
            number of bytes of keys and values must be less than 8KB.
        """
        return self._properties.get("userDefinedContext")

    @user_defined_context.setter
    def user_defined_context(self, value):
        if not isinstance(value, dict):
            raise ValueError("value must be dictionary")
        self._properties["userDefinedContext"] = value

    @classmethod
    def from_api_repr(cls, resource: dict) -> "RemoteFunctionOptions":
        """Factory: construct remote function options given its API representation.

        Args:
            resource (Dict[str, object]): Resource, as returned from the API.

        Returns:
            google.cloud.bigquery.routine.RemoteFunctionOptions:
                Python object, as parsed from ``resource``.
        """
        ref = cls()
        ref._properties = resource
        return ref

    def to_api_repr(self) -> dict:
        """Construct the API resource representation of this RemoteFunctionOptions.

        Returns:
            Dict[str, object]: Remote function options represented as an API resource.
        """
        return self._properties

    def __eq__(self, other):
        if not isinstance(other, RemoteFunctionOptions):
            return NotImplemented
        return self._properties == other._properties

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        all_properties = [
            "{}={}".format(property_name, repr(getattr(self, property_name)))
            for property_name in sorted(self._PROPERTY_TO_API_FIELD)
        ]
        return "RemoteFunctionOptions({})".format(", ".join(all_properties))
