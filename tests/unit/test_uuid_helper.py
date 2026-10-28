from typing import Final
from uuid import NAMESPACE_DNS, UUID, uuid5 as stdlib_uuid5

from litellm._uuid import uuid, uuid4


def test_uuid4_returns_canonical_random_uuid() -> None:
    for value in (uuid4(), uuid.uuid4()):
        encoded: Final = str(value)
        parsed: Final = UUID(encoded)
        assert parsed.version == 4
        assert encoded == str(parsed)
        assert value.bytes == parsed.bytes


def test_uuid5_accepts_byte_names_for_stable_ids() -> None:
    namespace: Final = uuid.UUID(str(NAMESPACE_DNS))
    name: Final = "guardrail-\u00e9"
    generated: Final = uuid.uuid5(namespace, name.encode("utf-8"))
    assert str(generated) == str(stdlib_uuid5(NAMESPACE_DNS, name))
