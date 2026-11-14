import importlib
import sys
from typing import Final

import pytest

OLD_PACKAGE: Final = "litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through"
NEW_PACKAGE: Final = "litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.messages"


def test_old_deep_import_is_the_new_module_object_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name == OLD_PACKAGE or name.startswith(OLD_PACKAGE + "."):
            monkeypatch.delitem(sys.modules, name)
    with pytest.warns(DeprecationWarning, match=NEW_PACKAGE):
        old_transformation = importlib.import_module(f"{OLD_PACKAGE}.transformation")
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.messages import transformation

    assert old_transformation is transformation
    assert sys.modules[f"{OLD_PACKAGE}.transformation"] is sys.modules[f"{NEW_PACKAGE}.transformation"]
