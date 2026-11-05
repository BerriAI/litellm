import pytest

from litellm._auth.decision import (
    Allow,
    Defer,
    Deny,
    DenyReason,
    MasterKey,
    NoMasterKey,
    Outcome,
    RequestFacts,
    Settings,
    UserAuthRoute,
)
from litellm._auth.python_core import PythonCore

MASTER: str = "sk-master"


@pytest.mark.parametrize(
    ("route", "token", "settings", "expected"),
    [
        ("/chat/completions", None, Settings(master_key=None, allow_user_auth=False), Allow(NoMasterKey(token=None))),
        (
            "/chat/completions",
            "sk-x",
            Settings(master_key=None, allow_user_auth=False),
            Allow(NoMasterKey(token="sk-x")),
        ),
        ("/chat/completions", None, Settings(master_key=MASTER, allow_user_auth=False), Deny(DenyReason.MISSING_KEY)),
        ("/chat/completions", "", Settings(master_key=MASTER, allow_user_auth=False), Deny(DenyReason.MALFORMED_KEY)),
        ("/user/auth", "", Settings(master_key=MASTER, allow_user_auth=True), Deny(DenyReason.MALFORMED_KEY)),
        ("/user/auth", "sk-x", Settings(master_key=MASTER, allow_user_auth=True), Allow(UserAuthRoute())),
        ("/user/auth", "sk-x", Settings(master_key=MASTER, allow_user_auth=False), Deny(DenyReason.USER_AUTH_DISABLED)),
        ("/user/auth", MASTER, Settings(master_key=MASTER, allow_user_auth=False), Deny(DenyReason.USER_AUTH_DISABLED)),
        ("/chat/completions", MASTER, Settings(master_key=MASTER, allow_user_auth=False), Allow(MasterKey())),
        ("/key/generate", MASTER, Settings(master_key=MASTER, allow_user_auth=False), Allow(MasterKey())),
        ("/chat/completions", "sk-master2", Settings(master_key=MASTER, allow_user_auth=False), Defer()),
        ("/chat/completions", "sk-x", Settings(master_key=MASTER, allow_user_auth=False), Defer()),
    ],
)
def test_decide(route: str, token: str | None, settings: Settings, expected: Outcome):
    assert PythonCore().decide(RequestFacts(route=route, token=token), settings) == expected
