from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from litellm.proxy.auth.v2.principal import build_principal


@dataclass
class _Identity:
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    token: Optional[str] = None
    user_role: Any = None


class _Role(str, Enum):
    # Mirrors LitellmUserRoles (a str Enum); build_principal reads .value off enums.
    PROXY_ADMIN = "proxy_admin"


def test_user_id_becomes_subject_and_team_becomes_domain():
    p = build_principal(_Identity(user_id="u1", team_id="t1"))
    assert p.subject == "user:u1"
    assert p.domain == "team:t1"


def test_role_is_bridged_into_a_grouping():
    p = build_principal(_Identity(user_id="u1", user_role=_Role("proxy_admin")))
    assert p.groupings == [["user:u1", "role:proxy_admin"]]


def test_plain_string_role_is_supported():
    p = build_principal(_Identity(user_id="u1", user_role="internal_user"))
    assert p.groupings == [["user:u1", "role:internal_user"]]


def test_no_role_yields_no_grouping():
    p = build_principal(_Identity(user_id="u1"))
    assert p.groupings == []


def test_falls_back_to_key_subject_and_global_domain():
    p = build_principal(_Identity(token="hashed", team_id=None))
    assert p.subject == "key:hashed"
    assert p.domain == "*"


def test_anonymous_when_no_identifiers():
    p = build_principal(_Identity())
    assert p.subject == "anonymous"
