import importlib


def test_fastuuid_flag_exposed():
    mod = importlib.import_module("litellm._uuid")
    assert hasattr(mod, "FASTUUID_AVAILABLE")
    assert hasattr(mod, "uuid4")
    # Ensure uuid4 returns something that looks like a UUID string
    val = str(mod.uuid4())
    assert isinstance(val, str)
    assert len(val) >= 8
