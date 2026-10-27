import importlib
import sys
from typing import Final

import pytest

OLD_PACKAGE: Final = "litellm.llms.anthropic.pass_through"
NEW_PACKAGE: Final = "litellm.llms.anthropic.messages"


def _forget_old_package(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name == OLD_PACKAGE or name.startswith(OLD_PACKAGE + "."):
            monkeypatch.delitem(sys.modules, name)


def test_old_deep_import_is_the_new_module_object_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    _forget_old_package(monkeypatch)
    with pytest.warns(DeprecationWarning, match=NEW_PACKAGE):
        old_handler: Final = importlib.import_module(f"{OLD_PACKAGE}.messages.handler")
    from litellm.llms.anthropic.messages.native import handler as new_handler

    assert old_handler is new_handler
    assert sys.modules[f"{OLD_PACKAGE}.messages.handler"] is new_handler
    assert importlib.import_module(f"{OLD_PACKAGE}.adapters.transformation") is importlib.import_module(
        f"{NEW_PACKAGE}.adapters.transformation"
    )
    assert importlib.import_module(f"{OLD_PACKAGE}.safeguards") is importlib.import_module(f"{NEW_PACKAGE}.safeguards")


def test_old_package_attribute_access_reaches_renamed_subpackage(monkeypatch: pytest.MonkeyPatch) -> None:
    _forget_old_package(monkeypatch)
    with pytest.warns(DeprecationWarning, match=NEW_PACKAGE):
        old_package: Final = importlib.import_module(OLD_PACKAGE)

    assert old_package.messages is importlib.import_module(f"{NEW_PACKAGE}.native")  # pyright: ignore[reportAny]  # module attributes are untyped
    assert old_package.messages.handler is importlib.import_module(f"{NEW_PACKAGE}.native.handler")  # pyright: ignore[reportAny]  # module attributes are untyped
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = old_package.nope  # pyright: ignore[reportAny]  # module attributes are untyped


def test_old_transformation_import_loads_only_its_new_counterpart(monkeypatch: pytest.MonkeyPatch) -> None:
    _forget_old_package(monkeypatch)
    monkeypatch.delitem(sys.modules, f"{NEW_PACKAGE}.native.streaming_iterator", raising=False)
    with pytest.warns(DeprecationWarning, match=NEW_PACKAGE):
        old_transformation: Final = importlib.import_module(f"{OLD_PACKAGE}.messages.transformation")

    assert old_transformation is sys.modules[f"{NEW_PACKAGE}.native.transformation"]
    assert f"{NEW_PACKAGE}.native.streaming_iterator" not in sys.modules, "shim imported the streaming iterator"


def test_old_subpackage_import_keeps_the_new_package_path_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    _forget_old_package(monkeypatch)
    monkeypatch.delitem(sys.modules, f"{NEW_PACKAGE}.native.mcp_handler", raising=False)
    with pytest.warns(DeprecationWarning, match=NEW_PACKAGE):
        native: Final = importlib.import_module(f"{OLD_PACKAGE}.messages")

    assert [path.rsplit("/", 2)[-2:] for path in native.__path__] == [["messages", "native"]]  # pyright: ignore[reportAny]  # module attributes are untyped
    assert native.__spec__ is not None and native.__spec__.name == f"{NEW_PACKAGE}.native"
    assert importlib.import_module(f"{NEW_PACKAGE}.native.mcp_handler").__name__ == f"{NEW_PACKAGE}.native.mcp_handler"
