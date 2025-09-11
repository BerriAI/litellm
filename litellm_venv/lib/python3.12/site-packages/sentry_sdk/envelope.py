import io
import json
import mimetypes

from sentry_sdk.session import Session
from sentry_sdk.utils import json_dumps, capture_internal_exceptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Optional
    from typing import Union
    from typing import Dict
    from typing import List
    from typing import Iterator

    from sentry_sdk._types import Event, EventDataCategory


def parse_json(data):
    # type: (Union[bytes, str]) -> Any
    # on some python 3 versions this needs to be bytes
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    return json.loads(data)


class Envelope:
    """
    Represents a Sentry Envelope. The calling code is responsible for adhering to the constraints
    documented in the Sentry docs: https://develop.sentry.dev/sdk/envelopes/#data-model. In particular,
    each envelope may have at most one Item with type "event" or "transaction" (but not both).
    """

    def __init__(
        self,
        headers=None,  # type: Optional[Dict[str, Any]]
        items=None,  # type: Optional[List[Item]]
    ):
        # type: (...) -> None
        if headers is not None:
            headers = dict(headers)
        self.headers = headers or {}
        if items is None:
            items = []
        else:
            items = list(items)
        self.items = items

    @property
    def description(self):
        # type: (...) -> str
        return "envelope with %s items (%s)" % (
            len(self.items),
            ", ".join(x.data_category for x in self.items),
        )

    def add_event(
        self, event  # type: Event
    ):
        # type: (...) -> None
        self.add_item(Item(payload=PayloadRef(json=event), type="event"))

    def add_transaction(
        self, transaction  # type: Event
    ):
        # type: (...) -> None
        self.add_item(Item(payload=PayloadRef(json=transaction), type="transaction"))

    def add_profile(
        self, profile  # type: Any
    ):
        # type: (...) -> None
        self.add_item(Item(payload=PayloadRef(json=profile), type="profile"))

    def add_profile_chunk(
        self, profile_chunk  # type: Any
    ):
        # type: (...) -> None
        self.add_item(
            Item(payload=PayloadRef(json=profile_chunk), type="profile_chunk")
        )

    def add_checkin(
        self, checkin  # type: Any
    ):
        # type: (...) -> None
        self.add_item(Item(payload=PayloadRef(json=checkin), type="check_in"))

    def add_session(
        self, session  # type: Union[Session, Any]
    ):
        # type: (...) -> None
        if isinstance(session, Session):
            session = session.to_json()
        self.add_item(Item(payload=PayloadRef(json=session), type="session"))

    def add_sessions(
        self, sessions  # type: Any
    ):
        # type: (...) -> None
        self.add_item(Item(payload=PayloadRef(json=sessions), type="sessions"))

    def add_item(
        self, item  # type: Item
    ):
        # type: (...) -> None
        self.items.append(item)

    def get_event(self):
        # type: (...) -> Optional[Event]
        for items in self.items:
            event = items.get_event()
            if event is not None:
                return event
        return None

    def get_transaction_event(self):
        # type: (...) -> Optional[Event]
        for item in self.items:
            event = item.get_transaction_event()
            if event is not None:
                return event
        return None

    def __iter__(self):
        # type: (...) -> Iterator[Item]
        return iter(self.items)

    def serialize_into(
        self, f  # type: Any
    ):
        # type: (...) -> None
        f.write(json_dumps(self.headers))
        f.write(b"\n")
        for item in self.items:
            item.serialize_into(f)

    def serialize(self):
        # type: (...) -> bytes
        out = io.BytesIO()
        self.serialize_into(out)
        return out.getvalue()

    @classmethod
    def deserialize_from(
        cls, f  # type: Any
    ):
        # type: (...) -> Envelope
        headers = parse_json(f.readline())
        items = []
        while 1:
            item = Item.deserialize_from(f)
            if item is None:
                break
            items.append(item)
        return cls(headers=headers, items=items)

    @classmethod
    def deserialize(
        cls, bytes  # type: bytes
    ):
        # type: (...) -> Envelope
        return cls.deserialize_from(io.BytesIO(bytes))

    def __repr__(self):
        # type: (...) -> str
        return "<Envelope headers=%r items=%r>" % (self.headers, self.items)


class PayloadRef:
    def __init__(
        self,
        bytes=None,  # type: Optional[bytes]
        path=None,  # type: Optional[Union[bytes, str]]
        json=None,  # type: Optional[Any]
    ):
        # type: (...) -> None
        self.json = json
        self.bytes = bytes
        self.path = path

    def get_bytes(self):
        # type: (...) -> bytes
        if self.bytes is None:
            if self.path is not None:
                with capture_internal_exceptions():
                    with open(self.path, "rb") as f:
                        self.bytes = f.read()
            elif self.json is not None:
                self.bytes = json_dumps(self.json)
        return self.bytes or b""

    @property
    def inferred_content_type(self):
        # type: (...) -> str
        if self.json is not None:
            return "application/json"
        elif self.path is not None:
            path = self.path
            if isinstance(path, bytes):
                path = path.decode("utf-8", "replace")
            ty = mimetypes.guess_type(path)[0]
            if ty:
                return ty
        return "application/octet-stream"

    def __repr__(self):
        # type: (...) -> str
        return "<Payload %r>" % (self.inferred_content_type,)


class Item:
    def __init__(
        self,
        payload,  # type: Union[bytes, str, PayloadRef]
        headers=None,  # type: Optional[Dict[str, Any]]
        type=None,  # type: Optional[str]
        content_type=None,  # type: Optional[str]
        filename=None,  # type: Optional[str]
    ):
        if headers is not None:
            headers = dict(headers)
        elif headers is None:
            headers = {}
        self.headers = headers
        if isinstance(payload, bytes):
            payload = PayloadRef(bytes=payload)
        elif isinstance(payload, str):
            payload = PayloadRef(bytes=payload.encode("utf-8"))
        else:
            payload = payload

        if filename is not None:
            headers["filename"] = filename
        if type is not None:
            headers["type"] = type
        if content_type is not None:
            headers["content_type"] = content_type
        elif "content_type" not in headers:
            headers["content_type"] = payload.inferred_content_type

        self.payload = payload

    def __repr__(self):
        # type: (...) -> str
        return "<Item headers=%r payload=%r data_category=%r>" % (
            self.headers,
            self.payload,
            self.data_category,
        )

    @property
    def type(self):
        # type: (...) -> Optional[str]
        return self.headers.get("type")

    @property
    def data_category(self):
        # type: (...) -> EventDataCategory
        ty = self.headers.get("type")
        if ty == "session" or ty == "sessions":
            return "session"
        elif ty == "attachment":
            return "attachment"
        elif ty == "transaction":
            return "transaction"
        elif ty == "event":
            return "error"
        elif ty == "client_report":
            return "internal"
        elif ty == "profile":
            return "profile"
        elif ty == "profile_chunk":
            return "profile_chunk"
        elif ty == "statsd":
            return "metric_bucket"
        elif ty == "check_in":
            return "monitor"
        else:
            return "default"

    def get_bytes(self):
        # type: (...) -> bytes
        return self.payload.get_bytes()

    def get_event(self):
        # type: (...) -> Optional[Event]
        """
        Returns an error event if there is one.
        """
        if self.type == "event" and self.payload.json is not None:
            return self.payload.json
        return None

    def get_transaction_event(self):
        # type: (...) -> Optional[Event]
        if self.type == "transaction" and self.payload.json is not None:
            return self.payload.json
        return None

    def serialize_into(
        self, f  # type: Any
    ):
        # type: (...) -> None
        headers = dict(self.headers)
        bytes = self.get_bytes()
        headers["length"] = len(bytes)
        f.write(json_dumps(headers))
        f.write(b"\n")
        f.write(bytes)
        f.write(b"\n")

    def serialize(self):
        # type: (...) -> bytes
        out = io.BytesIO()
        self.serialize_into(out)
        return out.getvalue()

    @classmethod
    def deserialize_from(
        cls, f  # type: Any
    ):
        # type: (...) -> Optional[Item]
        line = f.readline().rstrip()
        if not line:
            return None
        headers = parse_json(line)
        length = headers.get("length")
        if length is not None:
            payload = f.read(length)
            f.readline()
        else:
            # if no length was specified we need to read up to the end of line
            # and remove it (if it is present, i.e. not the very last char in an eof terminated envelope)
            payload = f.readline().rstrip(b"\n")
        if headers.get("type") in ("event", "transaction", "metric_buckets"):
            rv = cls(headers=headers, payload=PayloadRef(json=parse_json(payload)))
        else:
            rv = cls(headers=headers, payload=payload)
        return rv

    @classmethod
    def deserialize(
        cls, bytes  # type: bytes
    ):
        # type: (...) -> Optional[Item]
        return cls.deserialize_from(io.BytesIO(bytes))
