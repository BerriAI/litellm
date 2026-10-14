"""Failure values raised by `litellm/proxy/auth/auth_checks.py`. Kept free of `litellm` imports so any proxy module can import them without joining the `litellm.proxy` import cycle."""


class UserNotFoundError(ValueError):
    """The user row is provably absent, as opposed to merely unreadable, so a caller that reads a missing row as no user-level limits can key on it without also swallowing a database that would not answer."""

    def __init__(self, user_id: str) -> None:
        super().__init__(f"User doesn't exist in db. 'user_id'={user_id}. Create user via `/user/new` call.")
