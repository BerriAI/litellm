import importlib


def test_uses_fastuuid_and_uuid4_works():
    mod = importlib.import_module("litellm._uuid")
    fastuuid_mod = importlib.import_module("fastuuid")
    assert hasattr(mod, "uuid4")
    assert hasattr(mod, "uuid")
    assert mod.uuid is fastuuid_mod

    # Ensure uuid4 returns something that looks like a UUID string
    val = str(mod.uuid4())
    assert isinstance(val, str)
    assert len(val) >= 8
