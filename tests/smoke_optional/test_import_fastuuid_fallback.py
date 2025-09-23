import importlib
import sys


def test_import_without_fastuuid(monkeypatch):
    # Ensure fastuuid is not importable
    monkeypatch.setitem(sys.modules, 'fastuuid', None)
    # Purge litellm modules to force a clean import
    for k in list(sys.modules.keys()):
        if k == 'litellm' or k.startswith('litellm.'):
            sys.modules.pop(k, None)

    # Import should not raise even if fastuuid is absent
    mod = importlib.import_module('litellm')
    assert mod is not None

