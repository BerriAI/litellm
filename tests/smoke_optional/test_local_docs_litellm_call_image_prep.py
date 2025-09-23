import importlib.util
import sys
import types
import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_litellm_call_image_prep_uses_utils(monkeypatch):
    # Stub extractor utils used inside litellm_call before we import it by path
    fake_img_utils = types.SimpleNamespace(
        compress_image=lambda path, cache_dir=None: "data:image/png;base64,LOCAL",
        fetch_remote_image=lambda url, cache_dir=None: "data:image/jpeg;base64,REMOTE",
        IMAGE_EXT={".png", ".jpg"},
    )
    fake_resp_utils = types.SimpleNamespace(
        assemble_stream_text=lambda s: "",
        format_answer_with_logging=lambda *a, **k: "OK",
        to_messages_and_model=lambda item, default_model, **k: (default_model, item["messages"], {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "extractor.pipeline.utils.litellm_image_utils",
        fake_img_utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "extractor.pipeline.utils.litellm_response_utils",
        fake_resp_utils,
    )
    # Ensure parent extractor modules and response_utils stub exist
    import types as _types
    monkeypatch.setitem(sys.modules, "extractor", _types.ModuleType("extractor"))
    monkeypatch.setitem(sys.modules, "extractor.pipeline", _types.ModuleType("extractor.pipeline"))
    monkeypatch.setitem(sys.modules, "extractor.pipeline.utils", _types.ModuleType("extractor.pipeline.utils"))
    monkeypatch.setitem(
        sys.modules,
        "extractor.pipeline.utils.response_utils",
        types.SimpleNamespace(to_messages_and_model=lambda item, default_model, **k: (default_model, item["messages"], {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "extractor.pipeline.utils.log_utils",
        types.SimpleNamespace(sanitize_messages_for_return=lambda msgs, *_args, **_kw: msgs),
    )
    # Stub optional dependency pulled in by litellm
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))

    # Load local/docs litellm_call by file path
    path = "local/docs/05_ideas/litellm_call.py"
    spec = importlib.util.spec_from_file_location("litellm_call_local", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = mod  # register to satisfy dataclasses module lookups
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    # Access the internal helper
    prep = getattr(mod, "_prepare_messages_image_urls")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see both"},
                {"type": "image_url", "image_url": {"url": "http://x/a.jpg"}},
                {"type": "image_url", "image_url": {"url": "/tmp/a.png"}},
            ],
        }
    ]
    out = await prep(msgs, image_cache_dir=None)
    parts = out[0]["content"]
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    assert "data:image/jpeg;base64,REMOTE" in urls
    assert "data:image/png;base64,LOCAL" in urls
