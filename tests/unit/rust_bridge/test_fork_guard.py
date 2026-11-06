import pytest

from litellm.rust_bridge import fork_guard


def test_unused_extension_is_reserved() -> None:
    calls: list[None] = []

    fork_guard.reserve_process_for_forking("the gunicorn master", reserve=lambda: calls.append(None))

    assert calls == [None]


def test_used_extension_refuses_and_names_the_place() -> None:
    def reserve() -> None:
        raise RuntimeError("the native runtime already started in this process")

    with pytest.raises(fork_guard.NativeStateStartedBeforeFork, match="the gunicorn master") as raised:
        fork_guard.reserve_process_for_forking("the gunicorn master", reserve=reserve)

    assert isinstance(raised.value.__cause__, RuntimeError)
