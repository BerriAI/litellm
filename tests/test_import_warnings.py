import warnings

def test_import_litellm():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import litellm
        assert len(w) == 1, f"Warnings were raised: {[str(warning.message) for warning in w]}"
