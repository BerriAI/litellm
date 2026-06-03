import uuid as stdlib_uuid

from litellm import _uuid as mod


def test_uses_stdlib_uuid_and_uuid4_works():
    assert mod.uuid is stdlib_uuid

    val = mod.uuid4()
    assert isinstance(val, stdlib_uuid.UUID)
    assert val.version == 4
