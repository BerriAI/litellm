from types import SimpleNamespace

import pytest

from litellm.rust_bridge import fork_guard


def _reserve_with(monkeypatch: pytest.MonkeyPatch, native: object) -> None:
    monkeypatch.setattr(fork_guard, "get_native_bridge", lambda: native)
    fork_guard.reserve_process_for_forking("the gunicorn master")


def test_missing_extension_has_nothing_to_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    _reserve_with(monkeypatch, None)


def test_extension_built_before_reservation_existed_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _reserve_with(monkeypatch, SimpleNamespace())


def test_unused_extension_is_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    _reserve_with(monkeypatch, SimpleNamespace(reserve_process_for_forking=lambda: calls.append(None)))

    assert calls == [None]


def test_used_extension_refuses_and_names_the_place(monkeypatch: pytest.MonkeyPatch) -> None:
    def reserve() -> None:
        raise RuntimeError("the native runtime already started in this process")

    with pytest.raises(fork_guard.NativeStateStartedBeforeFork, match="the gunicorn master") as raised:
        _reserve_with(monkeypatch, SimpleNamespace(reserve_process_for_forking=reserve))

    assert isinstance(raised.value.__cause__, RuntimeError)
