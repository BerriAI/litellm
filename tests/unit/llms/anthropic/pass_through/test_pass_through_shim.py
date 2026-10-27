import importlib
import sys

import pytest

OLD_PACKAGE = "litellm.llms.anthropic.pass_through"
NEW_PACKAGE = "litellm.llms.anthropic.messages"


def _forget_old_package(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name == OLD_PACKAGE or name.startswith(OLD_PACKAGE + "."):
            monkeypatch.delitem(sys.modules, name)


def test_old_deep_import_is_the_new_module_object_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    _forget_old_package(monkeypatch)
    with pytest.warns(DeprecationWarning, match=NEW_PACKAGE):
        old_handler = importlib.import_module(f"{OLD_PACKAGE}.messages.handler")
    from litellm.llms.anthropic.messages.native import handler as new_handler

    assert old_handler is new_handler
    assert (
        sys.modules[f"{OLD_PACKAGE}.adapters.transformation"] is sys.modules[f"{NEW_PACKAGE}.adapters.transformation"]
    )
    assert sys.modules[f"{OLD_PACKAGE}.safeguards"] is sys.modules[f"{NEW_PACKAGE}.safeguards"]


def test_old_package_attribute_access_reaches_renamed_subpackage(monkeypatch: pytest.MonkeyPatch) -> None:
    _forget_old_package(monkeypatch)
    with pytest.warns(DeprecationWarning):
        old_package = importlib.import_module(OLD_PACKAGE)

    assert old_package.messages.handler is sys.modules[f"{NEW_PACKAGE}.native.handler"]  # pyright: ignore[reportAny]  # module attributes are untyped
