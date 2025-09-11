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

"""Define API Datasets."""

from __future__ import absolute_import

import copy
import json

import typing
from typing import Optional, List, Dict, Any, Union

import google.cloud._helpers  # type: ignore

from google.cloud.bigquery import _helpers
from google.cloud.bigquery.model import ModelReference
from google.cloud.bigquery.routine import Routine, RoutineReference
from google.cloud.bigquery.table import Table, TableReference
from google.cloud.bigquery.encryption_configuration import EncryptionConfiguration
from google.cloud.bigquery import external_config


def _get_table_reference(self, table_id: str) -> TableReference:
    """Constructs a TableReference.

    Args:
        table_id (str): The ID of the table.

    Returns:
        google.cloud.bigquery.table.TableReference:
            A table reference for a table in this dataset.
    """
    return TableReference(self, table_id)


def _get_model_reference(self, model_id):
    """Constructs a ModelReference.

    Args:
        model_id (str): the ID of the model.

    Returns:
        google.cloud.bigquery.model.ModelReference:
            A ModelReference for a model in this dataset.
    """
    return ModelReference.from_api_repr(
        {"projectId": self.project, "datasetId": self.dataset_id, "modelId": model_id}
    )


def _get_routine_reference(self, routine_id):
    """Constructs a RoutineReference.

    Args:
        routine_id (str): the ID of the routine.

    Returns:
        google.cloud.bigquery.routine.RoutineReference:
            A RoutineReference for a routine in this dataset.
    """
    return RoutineReference.from_api_repr(
        {
            "projectId": self.project,
            "datasetId": self.dataset_id,
            "routineId": routine_id,
        }
    )


class DatasetReference(object):
    """DatasetReferences are pointers to datasets.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets#datasetreference

    Args:
        project (str): The ID of the project
        dataset_id (str): The ID of the dataset

    Raises:
        ValueError: If either argument is not of type ``str``.
    """

    def __init__(self, project: str, dataset_id: str):
        if not isinstance(project, str):
            raise ValueError("Pass a string for project")
        if not isinstance(dataset_id, str):
            raise ValueError("Pass a string for dataset_id")
        self._project = project
        self._dataset_id = dataset_id

    @property
    def project(self):
        """str: Project ID of the dataset."""
        return self._project

    @property
    def dataset_id(self):
        """str: Dataset ID."""
        return self._dataset_id

    @property
    def path(self):
        """str: URL path for the dataset based on project and dataset ID."""
        return "/projects/%s/datasets/%s" % (self.project, self.dataset_id)

    table = _get_table_reference

    model = _get_model_reference

    routine = _get_routine_reference

    @classmethod
    def from_api_repr(cls, resource: dict) -> "DatasetReference":
        """Factory: construct a dataset reference given its API representation

        Args:
            resource (Dict[str, str]):
                Dataset reference resource representation returned from the API

        Returns:
            google.cloud.bigquery.dataset.DatasetReference:
                Dataset reference parsed from ``resource``.
        """
        project = resource["projectId"]
        dataset_id = resource["datasetId"]
        return cls(project, dataset_id)

    @classmethod
    def from_string(
        cls, dataset_id: str, default_project: Optional[str] = None
    ) -> "DatasetReference":
        """Construct a dataset reference from dataset ID string.

        Args:
            dataset_id (str):
                A dataset ID in standard SQL format. If ``default_project``
                is not specified, this must include both the project ID and
                the dataset ID, separated by ``.``.
            default_project (Optional[str]):
                The project ID to use when ``dataset_id`` does not include a
                project ID.

        Returns:
            DatasetReference:
                Dataset reference parsed from ``dataset_id``.

        Examples:
            >>> DatasetReference.from_string('my-project-id.some_dataset')
            DatasetReference('my-project-id', 'some_dataset')

        Raises:
            ValueError:
                If ``dataset_id`` is not a fully-qualified dataset ID in
                standard SQL format.
        """
        output_dataset_id = dataset_id
        parts = _helpers._split_id(dataset_id)

        if len(parts) == 1:
            if default_project is not None:
                output_project_id = default_project
            else:
                raise ValueError(
                    "When default_project is not set, dataset_id must be a "
                    "fully-qualified dataset ID in standard SQL format, "
                    'e.g., "project.dataset_id" got {}'.format(dataset_id)
                )
        elif len(parts) == 2:
            output_project_id, output_dataset_id = parts
        else:
            raise ValueError(
                "Too many parts in dataset_id. Expected a fully-qualified "
                "dataset ID in standard SQL format, "
                'e.g. "project.dataset_id", got {}'.format(dataset_id)
            )

        return cls(output_project_id, output_dataset_id)

    def to_api_repr(self) -> dict:
        """Construct the API resource representation of this dataset reference

        Returns:
            Dict[str, str]: dataset reference represented as an API resource
        """
        return {"projectId": self._project, "datasetId": self._dataset_id}

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple[str]: The contents of this :class:`.DatasetReference`.
        """
        return (self._project, self._dataset_id)

    def __eq__(self, other):
        if not isinstance(other, DatasetReference):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        return f"{self.project}.{self._dataset_id}"

    def __repr__(self):
        return "DatasetReference{}".format(self._key())


class AccessEntry(object):
    """Represents grant of an access role to an entity.

    An entry must have exactly one of the allowed
    :class:`google.cloud.bigquery.enums.EntityTypes`. If anything but ``view``, ``routine``,
    or ``dataset`` are set, a ``role`` is also required. ``role`` is omitted for ``view``,
    ``routine``, ``dataset``, because they are always read-only.

    See https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets.

    Args:
        role:
            Role granted to the entity. The following string values are
            supported: `'READER'`, `'WRITER'`, `'OWNER'`. It may also be
            :data:`None` if the ``entity_type`` is ``view``, ``routine``, or ``dataset``.

        entity_type:
            Type of entity being granted the role. See
            :class:`google.cloud.bigquery.enums.EntityTypes` for supported types.

        entity_id:
            If the ``entity_type`` is not 'view', 'routine', or 'dataset', the
            ``entity_id`` is the ``str`` ID of the entity being granted the role. If
            the ``entity_type`` is 'view' or 'routine', the ``entity_id`` is a ``dict``
            representing the view or routine from a different dataset to grant access
            to in the following format for views::

                {
                    'projectId': string,
                    'datasetId': string,
                    'tableId': string
                }

            For routines::

                {
                    'projectId': string,
                    'datasetId': string,
                    'routineId': string
                }

            If the ``entity_type`` is 'dataset', the ``entity_id`` is a ``dict`` that includes
            a 'dataset' field with a ``dict`` representing the dataset and a 'target_types'
            field with a ``str`` value of the dataset's resource type::

                {
                    'dataset': {
                        'projectId': string,
                        'datasetId': string,
                    },
                    'target_types: 'VIEWS'
                }

    Raises:
        ValueError:
            If a ``view``, ``routine``, or ``dataset`` has ``role`` set, or a non ``view``,
            non ``routine``, and non ``dataset`` **does not** have a ``role`` set.

    Examples:
        >>> entry = AccessEntry('OWNER', 'userByEmail', 'user@example.com')

        >>> view = {
        ...     'projectId': 'my-project',
        ...     'datasetId': 'my_dataset',
        ...     'tableId': 'my_table'
        ... }
        >>> entry = AccessEntry(None, 'view', view)
    """

    def __init__(
        self,
        role: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs,
    ):
        self._properties: Dict[str, Any] = {}
        if entity_type is not None:
            self._properties[entity_type] = entity_id
        self._properties["role"] = role
        self._entity_type: Optional[str] = entity_type
        for prop, val in kwargs.items():
            setattr(self, prop, val)

    @property
    def role(self) -> Optional[str]:
        """The role of the entry."""
        return typing.cast(Optional[str], self._properties.get("role"))

    @role.setter
    def role(self, value):
        self._properties["role"] = value

    @property
    def dataset(self) -> Optional[DatasetReference]:
        """API resource representation of a dataset reference."""
        value = _helpers._get_sub_prop(self._properties, ["dataset", "dataset"])
        return DatasetReference.from_api_repr(value) if value else None

    @dataset.setter
    def dataset(self, value):
        if self.role is not None:
            raise ValueError(
                "Role must be None for a dataset. Current " "role: %r" % (self.role)
            )

        if isinstance(value, str):
            value = DatasetReference.from_string(value).to_api_repr()

        if isinstance(value, DatasetReference):
            value = value.to_api_repr()

        if isinstance(value, (Dataset, DatasetListItem)):
            value = value.reference.to_api_repr()

        _helpers._set_sub_prop(self._properties, ["dataset", "dataset"], value)
        _helpers._set_sub_prop(
            self._properties,
            ["dataset", "targetTypes"],
            self._properties.get("targetTypes"),
        )

    @property
    def dataset_target_types(self) -> Optional[List[str]]:
        """Which resources that the dataset in this entry applies to."""
        return typing.cast(
            Optional[List[str]],
            _helpers._get_sub_prop(self._properties, ["dataset", "targetTypes"]),
        )

    @dataset_target_types.setter
    def dataset_target_types(self, value):
        self._properties.setdefault("dataset", {})
        _helpers._set_sub_prop(self._properties, ["dataset", "targetTypes"], value)

    @property
    def routine(self) -> Optional[RoutineReference]:
        """API resource representation of a routine reference."""
        value = typing.cast(Optional[Dict], self._properties.get("routine"))
        return RoutineReference.from_api_repr(value) if value else None

    @routine.setter
    def routine(self, value):
        if self.role is not None:
            raise ValueError(
                "Role must be None for a routine. Current " "role: %r" % (self.role)
            )

        if isinstance(value, str):
            value = RoutineReference.from_string(value).to_api_repr()

        if isinstance(value, RoutineReference):
            value = value.to_api_repr()

        if isinstance(value, Routine):
            value = value.reference.to_api_repr()

        self._properties["routine"] = value

    @property
    def view(self) -> Optional[TableReference]:
        """API resource representation of a view reference."""
        value = typing.cast(Optional[Dict], self._properties.get("view"))
        return TableReference.from_api_repr(value) if value else None

    @view.setter
    def view(self, value):
        if self.role is not None:
            raise ValueError(
                "Role must be None for a view. Current " "role: %r" % (self.role)
            )

        if isinstance(value, str):
            value = TableReference.from_string(value).to_api_repr()

        if isinstance(value, TableReference):
            value = value.to_api_repr()

        if isinstance(value, Table):
            value = value.reference.to_api_repr()

        self._properties["view"] = value

    @property
    def group_by_email(self) -> Optional[str]:
        """An email address of a Google Group to grant access to."""
        return typing.cast(Optional[str], self._properties.get("groupByEmail"))

    @group_by_email.setter
    def group_by_email(self, value):
        self._properties["groupByEmail"] = value

    @property
    def user_by_email(self) -> Optional[str]:
        """An email address of a user to grant access to."""
        return typing.cast(Optional[str], self._properties.get("userByEmail"))

    @user_by_email.setter
    def user_by_email(self, value):
        self._properties["userByEmail"] = value

    @property
    def domain(self) -> Optional[str]:
        """A domain to grant access to."""
        return typing.cast(Optional[str], self._properties.get("domain"))

    @domain.setter
    def domain(self, value):
        self._properties["domain"] = value

    @property
    def special_group(self) -> Optional[str]:
        """A special group to grant access to."""
        return typing.cast(Optional[str], self._properties.get("specialGroup"))

    @special_group.setter
    def special_group(self, value):
        self._properties["specialGroup"] = value

    @property
    def condition(self) -> Optional["Condition"]:
        """Optional[Condition]: The IAM condition associated with this entry."""
        value = typing.cast(Dict[str, Any], self._properties.get("condition"))
        return Condition.from_api_repr(value) if value else None

    @condition.setter
    def condition(self, value: Union["Condition", dict, None]):
        """Set the IAM condition for this entry."""
        if value is None:
            self._properties["condition"] = None
        elif isinstance(value, Condition):
            self._properties["condition"] = value.to_api_repr()
        elif isinstance(value, dict):
            self._properties["condition"] = value
        else:
            raise TypeError("condition must be a Condition object, dict, or None")

    @property
    def entity_type(self) -> Optional[str]:
        """The entity_type of the entry."""

        # The api_repr for an AccessEntry object is expected to be a dict with
        # only a few keys. Two keys that may be present are role and condition.
        # Any additional key is going to have one of ~eight different names:
        #   userByEmail, groupByEmail, domain, dataset, specialGroup, view,
        #   routine, iamMember

        # if self._entity_type is None, see if it needs setting
        # i.e. is there a key: value pair that should be associated with
        # entity_type and entity_id?
        if self._entity_type is None:
            resource = self._properties.copy()
            # we are empyting the dict to get to the last `key: value`` pair
            # so we don't keep these first entries
            _ = resource.pop("role", None)
            _ = resource.pop("condition", None)

            try:
                # we only need entity_type, because entity_id gets set elsewhere.
                entity_type, _ = resource.popitem()
            except KeyError:
                entity_type = None

            self._entity_type = entity_type

        return self._entity_type

    @property
    def entity_id(self) -> Optional[Union[Dict[str, Any], str]]:
        """The entity_id of the entry."""
        if self.entity_type:
            entity_type = self.entity_type
        else:
            return None
        return typing.cast(
            Optional[Union[Dict[str, Any], str]],
            self._properties.get(entity_type, None),
        )

    def __eq__(self, other):
        if not isinstance(other, AccessEntry):
            return NotImplemented
        return (
            self.role == other.role
            and self.entity_type == other.entity_type
            and self._normalize_entity_id(self.entity_id)
            == self._normalize_entity_id(other.entity_id)
            and self.condition == other.condition
        )

    @staticmethod
    def _normalize_entity_id(value):
        """Ensure consistent equality for dicts like 'view'."""
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        return value

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"<AccessEntry: role={self.role}, {self.entity_type}={self.entity_id}>"

    def _key(self):
        """A tuple key that uniquely describes this field.
        Used to compute this instance's hashcode and evaluate equality.
        Returns:
            Tuple: The contents of this :class:`~google.cloud.bigquery.dataset.AccessEntry`.
        """

        properties = self._properties.copy()

        # Dicts are not hashable.
        # Convert condition to a hashable datatype(s)
        condition = properties.get("condition")
        if isinstance(condition, dict):
            condition_key = tuple(sorted(condition.items()))
            properties["condition"] = condition_key

        prop_tup = tuple(sorted(properties.items()))
        return (self.role, self.entity_type, self.entity_id, prop_tup)

    def __hash__(self):
        return hash(self._key())

    def to_api_repr(self):
        """Construct the API resource representation of this access entry

        Returns:
            Dict[str, object]: Access entry represented as an API resource
        """
        resource = copy.deepcopy(self._properties)
        return resource

    @classmethod
    def from_api_repr(cls, resource: dict) -> "AccessEntry":
        """Factory: construct an access entry given its API representation

        Args:
            resource (Dict[str, object]):
                Access entry resource representation returned from the API

        Returns:
            google.cloud.bigquery.dataset.AccessEntry:
                Access entry parsed from ``resource``.
        """
        access_entry = cls()
        access_entry._properties = resource.copy()
        return access_entry


class Dataset(object):
    """Datasets are containers for tables.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets#resource-dataset

    Args:
        dataset_ref (Union[google.cloud.bigquery.dataset.DatasetReference, str]):
            A pointer to a dataset. If ``dataset_ref`` is a string, it must
            include both the project ID and the dataset ID, separated by
            ``.``.

    Note:
        Fields marked as "Output Only" are populated by the server and will only be
        available after calling :meth:`google.cloud.bigquery.client.Client.get_dataset`.
    """

    _PROPERTY_TO_API_FIELD = {
        "access_entries": "access",
        "created": "creationTime",
        "default_partition_expiration_ms": "defaultPartitionExpirationMs",
        "default_table_expiration_ms": "defaultTableExpirationMs",
        "friendly_name": "friendlyName",
        "default_encryption_configuration": "defaultEncryptionConfiguration",
        "is_case_insensitive": "isCaseInsensitive",
        "storage_billing_model": "storageBillingModel",
        "max_time_travel_hours": "maxTimeTravelHours",
        "default_rounding_mode": "defaultRoundingMode",
        "resource_tags": "resourceTags",
        "external_catalog_dataset_options": "externalCatalogDatasetOptions",
        "access_policy_version": "accessPolicyVersion",
    }

    def __init__(self, dataset_ref) -> None:
        if isinstance(dataset_ref, str):
            dataset_ref = DatasetReference.from_string(dataset_ref)
        self._properties = {"datasetReference": dataset_ref.to_api_repr(), "labels": {}}

    @property
    def max_time_travel_hours(self):
        """
        Optional[int]: Defines the time travel window in hours. The value can
        be from 48 to 168 hours (2 to 7 days), and in multiple of 24 hours
        (48, 72, 96, 120, 144, 168).
        The default value is 168 hours if this is not set.
        """
        return self._properties.get("maxTimeTravelHours")

    @max_time_travel_hours.setter
    def max_time_travel_hours(self, hours):
        if not isinstance(hours, int):
            raise ValueError(f"max_time_travel_hours must be an integer. Got {hours}")
        if hours < 2 * 24 or hours > 7 * 24:
            raise ValueError(
                "Time Travel Window should be from 48 to 168 hours (2 to 7 days)"
            )
        if hours % 24 != 0:
            raise ValueError("Time Travel Window should be multiple of 24")
        self._properties["maxTimeTravelHours"] = hours

    @property
    def default_rounding_mode(self):
        """Union[str, None]: defaultRoundingMode of the dataset as set by the user
        (defaults to :data:`None`).

        Set the value to one of ``'ROUND_HALF_AWAY_FROM_ZERO'``, ``'ROUND_HALF_EVEN'``, or
        ``'ROUNDING_MODE_UNSPECIFIED'``.

        See `default rounding mode
        <https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets#Dataset.FIELDS.default_rounding_mode>`_
        in REST API docs and `updating the default rounding model
        <https://cloud.google.com/bigquery/docs/updating-datasets#update_rounding_mode>`_
        guide.

        Raises:
            ValueError: for invalid value types.
        """
        return self._properties.get("defaultRoundingMode")

    @default_rounding_mode.setter
    def default_rounding_mode(self, value):
        possible_values = [
            "ROUNDING_MODE_UNSPECIFIED",
            "ROUND_HALF_AWAY_FROM_ZERO",
            "ROUND_HALF_EVEN",
        ]
        if not isinstance(value, str) and value is not None:
            raise ValueError("Pass a string, or None")
        if value is None:
            self._properties["defaultRoundingMode"] = "ROUNDING_MODE_UNSPECIFIED"
        if value not in possible_values and value is not None:
            raise ValueError(
                f'rounding mode needs to be one of {",".join(possible_values)}'
            )
        if value:
            self._properties["defaultRoundingMode"] = value

    @property
    def project(self):
        """str: Project ID of the project bound to the dataset."""
        return self._properties["datasetReference"]["projectId"]

    @property
    def path(self):
        """str: URL path for the dataset based on project and dataset ID."""
        return "/projects/%s/datasets/%s" % (self.project, self.dataset_id)

    @property
    def access_entries(self):
        """List[google.cloud.bigquery.dataset.AccessEntry]: Dataset's access
        entries.

        ``role`` augments the entity type and must be present **unless** the
        entity type is ``view`` or ``routine``.

        Raises:
            TypeError: If 'value' is not a sequence
            ValueError:
                If any item in the sequence is not an
                :class:`~google.cloud.bigquery.dataset.AccessEntry`.
        """
        entries = self._properties.get("access", [])
        return [AccessEntry.from_api_repr(entry) for entry in entries]

    @access_entries.setter
    def access_entries(self, value):
        if not all(isinstance(field, AccessEntry) for field in value):
            raise ValueError("Values must be AccessEntry instances")
        entries = [entry.to_api_repr() for entry in value]
        self._properties["access"] = entries

    @property
    def created(self):
        """Union[datetime.datetime, None]: Output only. Datetime at which the dataset was
        created (:data:`None` until set from the server).
        """
        creation_time = self._properties.get("creationTime")
        if creation_time is not None:
            # creation_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(creation_time)
            )

    @property
    def dataset_id(self):
        """str: Dataset ID."""
        return self._properties["datasetReference"]["datasetId"]

    @property
    def full_dataset_id(self):
        """Union[str, None]: Output only. ID for the dataset resource
        (:data:`None` until set from the server).

        In the format ``project_id:dataset_id``.
        """
        return self._properties.get("id")

    @property
    def reference(self):
        """google.cloud.bigquery.dataset.DatasetReference: A reference to this
        dataset.
        """
        return DatasetReference(self.project, self.dataset_id)

    @property
    def etag(self):
        """Union[str, None]: Output only. ETag for the dataset resource
        (:data:`None` until set from the server).
        """
        return self._properties.get("etag")

    @property
    def modified(self):
        """Union[datetime.datetime, None]: Output only. Datetime at which the dataset was
        last modified (:data:`None` until set from the server).
        """
        modified_time = self._properties.get("lastModifiedTime")
        if modified_time is not None:
            # modified_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(modified_time)
            )

    @property
    def self_link(self):
        """Union[str, None]: Output only. URL for the dataset resource
        (:data:`None` until set from the server).
        """
        return self._properties.get("selfLink")

    @property
    def default_partition_expiration_ms(self):
        """Optional[int]: The default partition expiration for all
        partitioned tables in the dataset, in milliseconds.

        Once this property is set, all newly-created partitioned tables in
        the dataset will have an ``time_paritioning.expiration_ms`` property
        set to this value, and changing the value will only affect new
        tables, not existing ones. The storage in a partition will have an
        expiration time of its partition time plus this value.

        Setting this property overrides the use of
        ``default_table_expiration_ms`` for partitioned tables: only one of
        ``default_table_expiration_ms`` and
        ``default_partition_expiration_ms`` will be used for any new
        partitioned table. If you provide an explicit
        ``time_partitioning.expiration_ms`` when creating or updating a
        partitioned table, that value takes precedence over the default
        partition expiration time indicated by this property.
        """
        return _helpers._int_or_none(
            self._properties.get("defaultPartitionExpirationMs")
        )

    @default_partition_expiration_ms.setter
    def default_partition_expiration_ms(self, value):
        self._properties["defaultPartitionExpirationMs"] = _helpers._str_or_none(value)

    @property
    def default_table_expiration_ms(self):
        """Union[int, None]: Default expiration time for tables in the dataset
        (defaults to :data:`None`).

        Raises:
            ValueError: For invalid value types.
        """
        return _helpers._int_or_none(self._properties.get("defaultTableExpirationMs"))

    @default_table_expiration_ms.setter
    def default_table_expiration_ms(self, value):
        if not isinstance(value, int) and value is not None:
            raise ValueError("Pass an integer, or None")
        self._properties["defaultTableExpirationMs"] = _helpers._str_or_none(value)

    @property
    def description(self):
        """Optional[str]: Description of the dataset as set by the user
        (defaults to :data:`None`).

        Raises:
            ValueError: for invalid value types.
        """
        return self._properties.get("description")

    @description.setter
    def description(self, value):
        if not isinstance(value, str) and value is not None:
            raise ValueError("Pass a string, or None")
        self._properties["description"] = value

    @property
    def friendly_name(self):
        """Union[str, None]: Title of the dataset as set by the user
        (defaults to :data:`None`).

        Raises:
            ValueError: for invalid value types.
        """
        return self._properties.get("friendlyName")

    @friendly_name.setter
    def friendly_name(self, value):
        if not isinstance(value, str) and value is not None:
            raise ValueError("Pass a string, or None")
        self._properties["friendlyName"] = value

    @property
    def location(self):
        """Union[str, None]: Location in which the dataset is hosted as set by
        the user (defaults to :data:`None`).

        Raises:
            ValueError: for invalid value types.
        """
        return self._properties.get("location")

    @location.setter
    def location(self, value):
        if not isinstance(value, str) and value is not None:
            raise ValueError("Pass a string, or None")
        self._properties["location"] = value

    @property
    def labels(self):
        """Dict[str, str]: Labels for the dataset.

        This method always returns a dict. To change a dataset's labels,
        modify the dict, then call
        :meth:`google.cloud.bigquery.client.Client.update_dataset`. To delete
        a label, set its value to :data:`None` before updating.

        Raises:
            ValueError: for invalid value types.
        """
        return self._properties.setdefault("labels", {})

    @labels.setter
    def labels(self, value):
        if not isinstance(value, dict):
            raise ValueError("Pass a dict")
        self._properties["labels"] = value

    @property
    def resource_tags(self):
        """Dict[str, str]: Resource tags of the dataset.

        Optional. The tags attached to this dataset. Tag keys are globally
        unique. Tag key is expected to be in the namespaced format, for
        example "123456789012/environment" where 123456789012 is
        the ID of the parent organization or project resource for this tag
        key. Tag value is expected to be the short name, for example
        "Production".

        Raises:
            ValueError: for invalid value types.
        """
        return self._properties.setdefault("resourceTags", {})

    @resource_tags.setter
    def resource_tags(self, value):
        if not isinstance(value, dict) and value is not None:
            raise ValueError("Pass a dict")
        self._properties["resourceTags"] = value

    @property
    def default_encryption_configuration(self):
        """google.cloud.bigquery.encryption_configuration.EncryptionConfiguration: Custom
        encryption configuration for all tables in the dataset.

        Custom encryption configuration (e.g., Cloud KMS keys) or :data:`None`
        if using default encryption.

        See `protecting data with Cloud KMS keys
        <https://cloud.google.com/bigquery/docs/customer-managed-encryption>`_
        in the BigQuery documentation.
        """
        prop = self._properties.get("defaultEncryptionConfiguration")
        if prop:
            prop = EncryptionConfiguration.from_api_repr(prop)
        return prop

    @default_encryption_configuration.setter
    def default_encryption_configuration(self, value):
        api_repr = value
        if value:
            api_repr = value.to_api_repr()
        self._properties["defaultEncryptionConfiguration"] = api_repr

    @property
    def is_case_insensitive(self):
        """Optional[bool]: True if the dataset and its table names are case-insensitive, otherwise False.
        By default, this is False, which means the dataset and its table names are case-sensitive.
        This field does not affect routine references.

        Raises:
            ValueError: for invalid value types.
        """
        return self._properties.get("isCaseInsensitive") or False

    @is_case_insensitive.setter
    def is_case_insensitive(self, value):
        if not isinstance(value, bool) and value is not None:
            raise ValueError("Pass a boolean value, or None")
        if value is None:
            value = False
        self._properties["isCaseInsensitive"] = value

    @property
    def storage_billing_model(self):
        """Union[str, None]: StorageBillingModel of the dataset as set by the user
        (defaults to :data:`None`).

        Set the value to one of ``'LOGICAL'``, ``'PHYSICAL'``, or
        ``'STORAGE_BILLING_MODEL_UNSPECIFIED'``. This change takes 24 hours to
        take effect and you must wait 14 days before you can change the storage
        billing model again.

        See `storage billing model
        <https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets#Dataset.FIELDS.storage_billing_model>`_
        in REST API docs and `updating the storage billing model
        <https://cloud.google.com/bigquery/docs/updating-datasets#update_storage_billing_models>`_
        guide.

        Raises:
            ValueError: for invalid value types.
        """
        return self._properties.get("storageBillingModel")

    @storage_billing_model.setter
    def storage_billing_model(self, value):
        if not isinstance(value, str) and value is not None:
            raise ValueError(
                "storage_billing_model must be a string (e.g. 'LOGICAL',"
                " 'PHYSICAL', 'STORAGE_BILLING_MODEL_UNSPECIFIED'), or None."
                f" Got {repr(value)}."
            )
        self._properties["storageBillingModel"] = value

    @property
    def external_catalog_dataset_options(self):
        """Options defining open source compatible datasets living in the
        BigQuery catalog. Contains metadata of open source database, schema
        or namespace represented by the current dataset."""

        prop = _helpers._get_sub_prop(
            self._properties, ["externalCatalogDatasetOptions"]
        )

        if prop is not None:
            prop = external_config.ExternalCatalogDatasetOptions.from_api_repr(prop)
        return prop

    @external_catalog_dataset_options.setter
    def external_catalog_dataset_options(self, value):
        value = _helpers._isinstance_or_raise(
            value, external_config.ExternalCatalogDatasetOptions, none_allowed=True
        )
        self._properties[
            self._PROPERTY_TO_API_FIELD["external_catalog_dataset_options"]
        ] = (value.to_api_repr() if value is not None else None)

    @property
    def access_policy_version(self):
        return self._properties.get("accessPolicyVersion")

    @access_policy_version.setter
    def access_policy_version(self, value):
        if not isinstance(value, int) and value is not None:
            raise ValueError("Pass an integer, or None")
        self._properties["accessPolicyVersion"] = value

    @classmethod
    def from_string(cls, full_dataset_id: str) -> "Dataset":
        """Construct a dataset from fully-qualified dataset ID.

        Args:
            full_dataset_id (str):
                A fully-qualified dataset ID in standard SQL format. Must
                include both the project ID and the dataset ID, separated by
                ``.``.

        Returns:
            Dataset: Dataset parsed from ``full_dataset_id``.

        Examples:
            >>> Dataset.from_string('my-project-id.some_dataset')
            Dataset(DatasetReference('my-project-id', 'some_dataset'))

        Raises:
            ValueError:
                If ``full_dataset_id`` is not a fully-qualified dataset ID in
                standard SQL format.
        """
        return cls(DatasetReference.from_string(full_dataset_id))

    @classmethod
    def from_api_repr(cls, resource: dict) -> "Dataset":
        """Factory: construct a dataset given its API representation

        Args:
            resource (Dict[str: object]):
                Dataset resource representation returned from the API

        Returns:
            google.cloud.bigquery.dataset.Dataset:
                Dataset parsed from ``resource``.
        """
        if (
            "datasetReference" not in resource
            or "datasetId" not in resource["datasetReference"]
        ):
            raise KeyError(
                "Resource lacks required identity information:"
                '["datasetReference"]["datasetId"]'
            )
        project_id = resource["datasetReference"]["projectId"]
        dataset_id = resource["datasetReference"]["datasetId"]
        dataset = cls(DatasetReference(project_id, dataset_id))
        dataset._properties = copy.deepcopy(resource)
        return dataset

    def to_api_repr(self) -> dict:
        """Construct the API resource representation of this dataset

        Returns:
            Dict[str, object]: The dataset represented as an API resource
        """
        return copy.deepcopy(self._properties)

    def _build_resource(self, filter_fields):
        """Generate a resource for ``update``."""
        return _helpers._build_resource_from_properties(self, filter_fields)

    table = _get_table_reference

    model = _get_model_reference

    routine = _get_routine_reference

    def __repr__(self):
        return "Dataset({})".format(repr(self.reference))


class DatasetListItem(object):
    """A read-only dataset resource from a list operation.

    For performance reasons, the BigQuery API only includes some of the
    dataset properties when listing datasets. Notably,
    :attr:`~google.cloud.bigquery.dataset.Dataset.access_entries` is missing.

    For a full list of the properties that the BigQuery API returns, see the
    `REST documentation for datasets.list
    <https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets/list>`_.


    Args:
        resource (Dict[str, str]):
            A dataset-like resource object from a dataset list response. A
            ``datasetReference`` property is required.

    Raises:
        ValueError:
            If ``datasetReference`` or one of its required members is missing
            from ``resource``.
    """

    def __init__(self, resource):
        if "datasetReference" not in resource:
            raise ValueError("resource must contain a datasetReference value")
        if "projectId" not in resource["datasetReference"]:
            raise ValueError(
                "resource['datasetReference'] must contain a projectId value"
            )
        if "datasetId" not in resource["datasetReference"]:
            raise ValueError(
                "resource['datasetReference'] must contain a datasetId value"
            )
        self._properties = resource

    @property
    def project(self):
        """str: Project bound to the dataset."""
        return self._properties["datasetReference"]["projectId"]

    @property
    def dataset_id(self):
        """str: Dataset ID."""
        return self._properties["datasetReference"]["datasetId"]

    @property
    def full_dataset_id(self):
        """Union[str, None]: ID for the dataset resource (:data:`None` until
        set from the server)

        In the format ``project_id:dataset_id``.
        """
        return self._properties.get("id")

    @property
    def friendly_name(self):
        """Union[str, None]: Title of the dataset as set by the user
        (defaults to :data:`None`).
        """
        return self._properties.get("friendlyName")

    @property
    def labels(self):
        """Dict[str, str]: Labels for the dataset."""
        return self._properties.setdefault("labels", {})

    @property
    def reference(self):
        """google.cloud.bigquery.dataset.DatasetReference: A reference to this
        dataset.
        """
        return DatasetReference(self.project, self.dataset_id)

    table = _get_table_reference

    model = _get_model_reference

    routine = _get_routine_reference


class Condition(object):
    """Represents a textual expression in the Common Expression Language (CEL) syntax.

    Typically used for filtering or policy rules, such as in IAM Conditions
    or BigQuery row/column access policies.

    See:
        https://cloud.google.com/iam/docs/reference/rest/Shared.Types/Expr
        https://github.com/google/cel-spec

    Args:
        expression (str):
            The condition expression string using CEL syntax. This is required.
            Example: ``resource.type == "compute.googleapis.com/Instance"``
        title (Optional[str]):
            An optional title for the condition, providing a short summary.
            Example: ``"Request is for a GCE instance"``
        description (Optional[str]):
            An optional description of the condition, providing a detailed explanation.
            Example: ``"This condition checks whether the resource is a GCE instance."``
    """

    def __init__(
        self,
        expression: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._properties: Dict[str, Any] = {}
        # Use setters to initialize properties, which also handle validation
        self.expression = expression
        self.title = title
        self.description = description

    @property
    def title(self) -> Optional[str]:
        """Optional[str]: The title for the condition."""
        return self._properties.get("title")

    @title.setter
    def title(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise ValueError("Pass a string for title, or None")
        self._properties["title"] = value

    @property
    def description(self) -> Optional[str]:
        """Optional[str]: The description for the condition."""
        return self._properties.get("description")

    @description.setter
    def description(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise ValueError("Pass a string for description, or None")
        self._properties["description"] = value

    @property
    def expression(self) -> str:
        """str: The expression string for the condition."""

        # Cast assumes expression is always set due to __init__ validation
        return typing.cast(str, self._properties.get("expression"))

    @expression.setter
    def expression(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Pass a non-empty string for expression")
        if not value:
            raise ValueError("expression cannot be an empty string")
        self._properties["expression"] = value

    def to_api_repr(self) -> Dict[str, Any]:
        """Construct the API resource representation of this Condition."""
        return self._properties

    @classmethod
    def from_api_repr(cls, resource: Dict[str, Any]) -> "Condition":
        """Factory: construct a Condition instance given its API representation."""

        # Ensure required fields are present in the resource if necessary
        if "expression" not in resource:
            raise ValueError("API representation missing required 'expression' field.")

        return cls(
            expression=resource["expression"],
            title=resource.get("title"),
            description=resource.get("description"),
        )

    def __eq__(self, other: object) -> bool:
        """Check for equality based on expression, title, and description."""
        if not isinstance(other, Condition):
            return NotImplemented
        return self._key() == other._key()

    def _key(self):
        """A tuple key that uniquely describes this field.
        Used to compute this instance's hashcode and evaluate equality.
        Returns:
            Tuple: The contents of this :class:`~google.cloud.bigquery.dataset.AccessEntry`.
        """

        properties = self._properties.copy()

        # Dicts are not hashable.
        # Convert object to a hashable datatype(s)
        prop_tup = tuple(sorted(properties.items()))
        return prop_tup

    def __ne__(self, other: object) -> bool:
        """Check for inequality."""
        return not self == other

    def __hash__(self) -> int:
        """Generate a hash based on expression, title, and description."""
        return hash(self._key())

    def __repr__(self) -> str:
        """Return a string representation of the Condition object."""
        parts = [f"expression={self.expression!r}"]
        if self.title is not None:
            parts.append(f"title={self.title!r}")
        if self.description is not None:
            parts.append(f"description={self.description!r}")
        return f"Condition({', '.join(parts)})"
