import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_langfuse_get_prompt_caching(monkeypatch):
    """Ensure get_prompt uses cache to avoid repeated API calls."""

    call_count = {"count": 0}

    class DummyClientProjects:
        def get(self):
            return types.SimpleNamespace(data=[types.SimpleNamespace(id="proj")])

    class DummyLangfuse:
        def __init__(self, *args, **kwargs):
            self.client = types.SimpleNamespace(projects=DummyClientProjects())

        def get_prompt(self, name, variables=None):
            call_count["count"] += 1
            return {"name": name, "prompt": "hello"}

    dummy_module = types.SimpleNamespace(
        Langfuse=DummyLangfuse, version=types.SimpleNamespace(__version__="2.7.3")
    )
    monkeypatch.setitem(sys.modules, "langfuse", dummy_module)

    from litellm.integrations.langfuse import LangFuseLogger

    logger = LangFuseLogger()

    first = logger.get_prompt("test-prompt")
    second = logger.get_prompt("test-prompt")

    assert first == second
    assert call_count["count"] == 1

