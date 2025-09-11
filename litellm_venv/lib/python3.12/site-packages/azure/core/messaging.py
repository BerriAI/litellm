# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations
import uuid
from base64 import b64decode
from datetime import datetime
from typing import cast, Union, Any, Optional, Dict, TypeVar, Generic
from .utils._utils import _convert_to_isoformat, TZ_UTC
from .utils._messaging_shared import _get_json_content
from .serialization import NULL


__all__ = ["CloudEvent"]


_Unset: Any = object()

DataType = TypeVar("DataType")


class CloudEvent(Generic[DataType]):
    """Properties of the CloudEvent 1.0 Schema.
    All required parameters must be populated in order to send to Azure.

    :param source: Required. Identifies the context in which an event happened. The combination of id and source must
     be unique for each distinct event. If publishing to a domain topic, source must be the domain topic name.
    :type source: str
    :param type: Required. Type of event related to the originating occurrence.
    :type type: str
    :keyword specversion: Optional. The version of the CloudEvent spec. Defaults to "1.0"
    :paramtype specversion: str
    :keyword data: Optional. Event data specific to the event type.
    :paramtype data: object
    :keyword time: Optional. The time (in UTC) the event was generated.
    :paramtype time: ~datetime.datetime
    :keyword dataschema: Optional. Identifies the schema that data adheres to.
    :paramtype dataschema: str
    :keyword datacontenttype: Optional. Content type of data value.
    :paramtype datacontenttype: str
    :keyword subject: Optional. This describes the subject of the event in the context of the event producer
     (identified by source).
    :paramtype subject: str
    :keyword id: Optional. An identifier for the event. The combination of id and source must be
     unique for each distinct event. If not provided, a random UUID will be generated and used.
    :paramtype id: Optional[str]
    :keyword extensions: Optional. A CloudEvent MAY include any number of additional context attributes
     with distinct names represented as key - value pairs. Each extension must be alphanumeric, lower cased
     and must not exceed the length of 20 characters.
    :paramtype extensions: Optional[dict]
    """

    source: str
    """Identifies the context in which an event happened. The combination of id and source must
       be unique for each distinct event. If publishing to a domain topic, source must be the domain topic name."""

    type: str
    """Type of event related to the originating occurrence."""

    specversion: str = "1.0"
    """The version of the CloudEvent spec. Defaults to "1.0" """

    id: str
    """An identifier for the event. The combination of id and source must be
       unique for each distinct event. If not provided, a random UUID will be generated and used."""

    data: Optional[DataType]
    """Event data specific to the event type."""

    time: Optional[datetime]
    """The time (in UTC) the event was generated."""

    dataschema: Optional[str]
    """Identifies the schema that data adheres to."""

    datacontenttype: Optional[str]
    """Content type of data value."""

    subject: Optional[str]
    """This describes the subject of the event in the context of the event producer
       (identified by source)"""

    extensions: Optional[Dict[str, Any]]
    """A CloudEvent MAY include any number of additional context attributes
       with distinct names represented as key - value pairs. Each extension must be alphanumeric, lower cased
       and must not exceed the length of 20 characters."""

    def __init__(
        self,
        source: str,
        type: str,  # pylint: disable=redefined-builtin
        *,
        specversion: Optional[str] = None,
        id: Optional[str] = None,  # pylint: disable=redefined-builtin
        time: Optional[datetime] = _Unset,
        datacontenttype: Optional[str] = None,
        dataschema: Optional[str] = None,
        subject: Optional[str] = None,
        data: Optional[DataType] = None,
        extensions: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.source: str = source
        self.type: str = type

        if specversion:
            self.specversion: str = specversion
        self.id: str = id if id else str(uuid.uuid4())

        self.time: Optional[datetime]
        if time is _Unset:
            self.time = datetime.now(TZ_UTC)
        else:
            self.time = time

        self.datacontenttype: Optional[str] = datacontenttype
        self.dataschema: Optional[str] = dataschema
        self.subject: Optional[str] = subject
        self.data: Optional[DataType] = data

        self.extensions: Optional[Dict[str, Any]] = extensions
        if self.extensions:
            for key in self.extensions.keys():
                if not key.islower() or not key.isalnum():
                    raise ValueError(
                        "Extension attributes should be lower cased and alphanumeric."
                    )

        if kwargs:
            remaining = ", ".join(kwargs.keys())
            raise ValueError(
                f"Unexpected keyword arguments {remaining}. "
                + "Any extension attributes must be passed explicitly using extensions."
            )

    def __repr__(self) -> str:
        return "CloudEvent(source={}, type={}, specversion={}, id={}, time={})".format(
            self.source, self.type, self.specversion, self.id, self.time
        )[:1024]

    @classmethod
    def from_dict(cls, event: Dict[str, Any]) -> CloudEvent[DataType]:
        """Returns the deserialized CloudEvent object when a dict is provided.

        :param event: The dict representation of the event which needs to be deserialized.
        :type event: dict
        :rtype: CloudEvent
        :return: The deserialized CloudEvent object.
        """
        kwargs: Dict[str, Any] = {}
        reserved_attr = [
            "data",
            "data_base64",
            "id",
            "source",
            "type",
            "specversion",
            "time",
            "dataschema",
            "datacontenttype",
            "subject",
        ]

        if "data" in event and "data_base64" in event:
            raise ValueError(
                "Invalid input. Only one of data and data_base64 must be present."
            )

        if "data" in event:
            data = event.get("data")
            kwargs["data"] = data if data is not None else NULL
        elif "data_base64" in event:
            kwargs["data"] = b64decode(
                cast(Union[str, bytes], event.get("data_base64"))
            )

        for item in ["datacontenttype", "dataschema", "subject"]:
            if item in event:
                val = event.get(item)
                kwargs[item] = val if val is not None else NULL

        extensions = {k: v for k, v in event.items() if k not in reserved_attr}
        if extensions:
            kwargs["extensions"] = extensions

        try:
            event_obj = cls(
                id=event.get("id"),
                source=event["source"],
                type=event["type"],
                specversion=event.get("specversion"),
                time=_convert_to_isoformat(event.get("time")),
                **kwargs,
            )
        except KeyError as err:
            # https://github.com/cloudevents/spec Cloud event spec requires source, type,
            # specversion. We autopopulate everything other than source, type.
            # So we will assume the KeyError is coming from source/type access.
            if all(
                key in event
                for key in (
                    "subject",
                    "eventType",
                    "data",
                    "dataVersion",
                    "id",
                    "eventTime",
                )
            ):
                raise ValueError(
                    "The event you are trying to parse follows the Eventgrid Schema. You can parse"
                    + " EventGrid events using EventGridEvent.from_dict method in the azure-eventgrid library."
                ) from err
            raise ValueError(
                "The event does not conform to the cloud event spec https://github.com/cloudevents/spec."
                + " The `source` and `type` params are required."
            ) from err
        return event_obj

    @classmethod
    def from_json(cls, event: Any) -> CloudEvent[DataType]:
        """Returns the deserialized CloudEvent object when a json payload is provided.

        :param event: The json string that should be converted into a CloudEvent. This can also be
         a storage QueueMessage, eventhub's EventData or ServiceBusMessage
        :type event: object
        :rtype: CloudEvent
        :return: The deserialized CloudEvent object.
        :raises ValueError: If the provided JSON is invalid.
        """
        dict_event = _get_json_content(event)
        return CloudEvent.from_dict(dict_event)
