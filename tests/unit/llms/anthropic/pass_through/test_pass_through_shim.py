import importlib
import sys

import pytest

OLD_PACKAGE = "litellm.llms.anthropic.pass_through"


def _forget_old_package(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name == OLD_PACKAGE or name.startswith(OLD_PACKAGE + "."):
            monkeypatch.delitem(sys.modules, name)


def test_old_deep_import_is_the_new_module_object_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    _forget_old_package(monkeypatch)
    with pytest.warns(DeprecationWarning, match="litellm.llms.anthropic.messages"):
        old_handler = importlib.import_module(f"{OLD_PACKAGE}.messages.handler")
    from litellm.llms.anthropic.messages.native import handler as new_handler
    from litellm.llms.anthropic.messages.native.handler import anthropic_messages_handler

    assert old_handler is new_handler
    assert old_handler.anthropic_messages_handler is anthropic_messages_handler


def test_old_package_attribute_access_reaches_renamed_subpackage(monkeypatch: pytest.MonkeyPatch) -> None:
    _forget_old_package(monkeypatch)
    with pytest.warns(DeprecationWarning):
        old_package = importlib.import_module(OLD_PACKAGE)
    from litellm.llms.anthropic.messages import safeguards
    from litellm.llms.anthropic.messages.adapters.transformation import LiteLLMAnthropicMessagesAdapter

    assert old_package.safeguards is safeguards
    assert old_package.adapters.transformation.LiteLLMAnthropicMessagesAdapter is LiteLLMAnthropicMessagesAdapter
    assert old_package.messages.handler is sys.modules["litellm.llms.anthropic.messages.native.handler"]
