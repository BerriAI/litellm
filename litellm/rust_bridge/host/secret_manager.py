from __future__ import annotations


def read_secrets(names: tuple[str, ...]) -> tuple[str | None, ...]:
    from litellm.secret_managers.main import get_secret_str

    return tuple(get_secret_str(name) for name in names)
