import importlib
import uuid as stdlib_uuid


def test_uses_fastuuid_or_stdlib_and_uuid4_works():
    mod = importlib.import_module("litellm._uuid")
    assert hasattr(mod, "uuid4")
    assert hasattr(mod, "uuid")

    try:
        fastuuid_mod = importlib.import_module("fastuuid")
        assert mod.uuid is fastuuid_mod
    except ImportError:
        assert mod.uuid is stdlib_uuid

    # Ensure uuid4 returns something that looks like a UUID string
    val = str(mod.uuid4())
    assert isinstance(val, str)
    assert len(val) >= 8

