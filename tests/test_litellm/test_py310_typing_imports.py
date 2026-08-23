import litellm


def test_import_litellm_succeeds():
    # regression for py3.10 typing imports (requires-python >=3.10)
    assert hasattr(litellm, "completion")
