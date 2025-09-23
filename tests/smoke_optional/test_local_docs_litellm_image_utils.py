import importlib.util
import sys
import types
import pytest


@pytest.mark.smoke
def test_local_docs_litellm_image_utils_reexport(monkeypatch):
    # Provide a fake extractor image_helpers so the shim can import it
    calls = {"compress": 0, "fetch": 0}

    def _compress(path_str, max_kb=1000, cache_dir=None):
        calls["compress"] += 1
        return "data:image/png;base64,FAKE"

    def _fetch(url, cache_dir=None):
        calls["fetch"] += 1
        return "data:image/jpeg;base64,FAKE"

    import types as _types
    # Ensure parent modules exist
    monkeypatch.setitem(sys.modules, "extractor", _types.ModuleType("extractor"))
    monkeypatch.setitem(sys.modules, "extractor.pipeline", _types.ModuleType("extractor.pipeline"))
    monkeypatch.setitem(sys.modules, "extractor.pipeline.utils", _types.ModuleType("extractor.pipeline.utils"))

    fake_image_helpers = types.SimpleNamespace(
        compress_image=_compress,
        fetch_remote_image=_fetch,
        extract_images=lambda text: (["http://x/a.jpg", "/tmp/p.png"], "clean"),
    )
    monkeypatch.setitem(
        sys.modules, "extractor.pipeline.utils.image_helpers", fake_image_helpers
    )

    path = "local/docs/05_ideas/litellm_image_utils.py"
    spec = importlib.util.spec_from_file_location("litellm_image_utils_local", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    assert isinstance(mod.compress_image("/tmp/f.png"), str)
    assert isinstance(mod.fetch_remote_image("http://x/a.jpg"), str)
    assert calls["compress"] == 1 and calls["fetch"] == 1
