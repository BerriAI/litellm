from types import SimpleNamespace

from litellm.rust_bridge import bindings


def test_binding_distinguishes_disable_from_reset(monkeypatch) -> None:
    native = SimpleNamespace(route=lambda: "native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    binding: bindings.NativeBinding[object] = bindings.NativeBinding("route")

    assert binding.load() is native.route

    binding.override(None)
    assert binding.load() is None

    replacement = object()
    binding.override(replacement)
    assert binding.load() is replacement

    binding.reset()
    assert binding.load() is native.route
