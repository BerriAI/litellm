import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import partial
from types import MappingProxyType, TracebackType
from typing import Final, Literal

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy import proxy_server
from litellm.proxy._lazy_openapi_snapshot import load_snapshot
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient
from litellm.repositories import rust_database
from litellm.rust_bridge._native import RustJwtKeyMappingRepository, RustObjectPermissionRepository, RustTeamRepository
from tests.unit.proxy.rust_auth_stores import FakeJwtKeyMappingStore

Entity = Literal["key", "team", "agent", "user", "organization", "customer", "project"]


@dataclass(frozen=True)
class RouteContract:
    entity: Entity
    discoverable: bool = True


ROUTE_MATRIX: Final[Mapping[tuple[str, str], RouteContract]] = MappingProxyType(
    {
        ("POST", "/key/generate"): RouteContract("key"),
        ("POST", "/key/service-account/generate"): RouteContract("key"),
        ("POST", "/key/update"): RouteContract("key"),
        ("POST", "/key/bulk_update"): RouteContract("key"),
        ("POST", "/key/regenerate"): RouteContract("key"),
        ("POST", "/key/{key}/regenerate"): RouteContract("key"),
        ("POST", "/team/new"): RouteContract("team"),
        ("POST", "/team/update"): RouteContract("team"),
        ("PATCH", "/team/{team_id}"): RouteContract("team"),
        ("POST", "/v1/agents"): RouteContract("agent"),
        ("PUT", "/v1/agents/{agent_id}"): RouteContract("agent"),
        ("PATCH", "/v1/agents/{agent_id}"): RouteContract("agent"),
        ("POST", "/user/new"): RouteContract("user"),
        ("POST", "/user/update"): RouteContract("user"),
        ("POST", "/user/bulk_update"): RouteContract("user"),
        ("POST", "/management/v1/users/bulk"): RouteContract("user"),
        ("POST", "/organization/new"): RouteContract("organization"),
        ("PATCH", "/organization/update"): RouteContract("organization", discoverable=False),
        ("PATCH", "/v2/organization/{organization_id}"): RouteContract("organization"),
        ("POST", "/customer/new"): RouteContract("customer"),
        ("POST", "/customer/update"): RouteContract("customer"),
        ("POST", "/end_user/new"): RouteContract("customer"),
        ("POST", "/end_user/update"): RouteContract("customer"),
        ("POST", "/project/new"): RouteContract("project"),
        ("POST", "/project/update"): RouteContract("project"),
    }
)


def _schema_names_object_permission(schema: object, definitions: Mapping[str, object], seen: set[str]) -> bool:
    if isinstance(schema, list):
        return any(_schema_names_object_permission(item, definitions, seen) for item in schema)
    if not isinstance(schema, dict):
        return False
    reference: Final = schema.get("$ref")
    if isinstance(reference, str):
        name: Final = reference.rsplit("/", 1)[-1]
        if name in seen:
            return False
        seen.add(name)
        return _schema_names_object_permission(definitions.get(name, {}), definitions, seen)
    properties: Final = schema.get("properties")
    if isinstance(properties, dict) and "object_permission" in properties:
        return True
    return any(_schema_names_object_permission(value, definitions, seen) for value in schema.values())


def _body_takes_object_permission(route: APIRoute) -> bool:
    for param in route.dependant.body_params:
        schema: Final = TypeAdapter(param.field_info.annotation).json_schema(ref_template="#/$defs/{model}")
        if _schema_names_object_permission(schema, schema.get("$defs", {}), set()):
            return True
    return False


def _normalized(path: str) -> str:
    return path.replace(":path}", "}")


def _discovered_routes() -> frozenset[tuple[str, str]]:
    loaded: Final = {
        (method, _normalized(route.path))
        for route in proxy_server.app.routes
        if isinstance(route, APIRoute) and _body_takes_object_permission(route)
        for method in route.methods
    }
    lazy: Final = {
        (method.upper(), _normalized(path))
        for feature in (load_snapshot() or {}).values()
        for path, operations in feature.get("paths", {}).items()
        for method, operation in operations.items()
        if isinstance(operation, dict)
        and _schema_names_object_permission(
            operation.get("requestBody"), feature.get("components", {}).get("schemas", {}), set()
        )
    }
    return frozenset(loaded | lazy)


def test_every_route_taking_object_permission_is_in_the_matrix() -> None:
    discoverable: Final = frozenset(route for route, contract in ROUTE_MATRIX.items() if contract.discoverable)

    assert _discovered_routes() == discoverable


_WRITES: Final = frozenset({"create", "create_many", "update", "update_many", "upsert", "delete", "delete_many"})
_READS: Final = frozenset(
    {"find_unique", "find_unique_or_raise", "find_first", "find_first_or_raise", "find_many", "count", "group_by"}
)
_RELATION_WRITES: Final = frozenset({"create", "connect", "update", "upsert", "disconnect", "connectOrCreate", "set"})
_NOT_COLUMNS: Final = frozenset({"keys", "items", "values", "get"})
_STORED_AT: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
_PERMISSION_LISTS: Final = (
    "mcp_servers",
    "mcp_access_groups",
    "vector_stores",
    "agents",
    "agent_access_groups",
    "models",
    "mcp_toolsets",
    "blocked_tools",
    "search_tools",
    "skills",
    "mcp_auth_profiles",
)
_OWNED_LISTS: Final = MappingProxyType(
    dict.fromkeys(("models", "members", "teams", "admins", "members_with_roles"), ())
)
_TABLE_DEFAULTS: Final[Mapping[str, Mapping[str, object]]] = MappingProxyType(
    {
        "litellm_objectpermissiontable": dict.fromkeys(_PERMISSION_LISTS, ()),
        "litellm_organizationtable": {**_OWNED_LISTS, "spend": 0.0, "metadata": {}},
        "litellm_teamtable": {**_OWNED_LISTS, "spend": 0.0, "blocked": False, "metadata": {}},
        "litellm_endusertable": {"blocked": False, "spend": 0.0},
        "litellm_projecttable": {"spend": 0.0, "models": (), "metadata": {}},
        "litellm_agentstable": {"litellm_params": {}, "agent_card_params": {}},
        "litellm_verificationtoken": {"spend": 0.0, "models": (), "metadata": {}},
        "litellm_usertable": {"spend": 0.0, "models": (), "teams": (), "metadata": {}},
    }
)
_PRIMARY_KEYS: Final = MappingProxyType(
    {
        "litellm_budgettable": "budget_id",
        "litellm_objectpermissiontable": "object_permission_id",
        "litellm_organizationtable": "organization_id",
        "litellm_teamtable": "team_id",
        "litellm_projecttable": "project_id",
        "litellm_agentstable": "agent_id",
        "litellm_proxymodeltable": "model_id",
    }
)


class Row(BaseModel, extra="allow"):  # frozen-ok: routes assign to the rows they read, as Prisma models allow
    def __getattr__(self, name: str) -> object:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("_") or name in _NOT_COLUMNS:
                raise
            return None


@dataclass(frozen=True)
class Write:
    table: str
    method: str
    data: object


def _matches(row: Mapping[str, object], where: Mapping[str, object] | None) -> bool:
    for column, condition in (where or {}).items():
        if column in ("AND", "OR", "NOT"):
            continue
        value: Final = row.get(column)
        if isinstance(condition, Mapping):
            if "in" in condition and value not in condition["in"]:
                return False
            if "equals" in condition and value != condition["equals"]:
                return False
        elif value != condition:
            return False
    return True


def _as_stored(value: object) -> object:
    if isinstance(value, str) and (value[:1] in ("{", "[") or value == "null"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _is_relation_write(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value) and set(value) <= _RELATION_WRITES


class _Table:
    def __init__(self, store: "RecordingDb", name: str) -> None:
        self._store = store
        self._name = name

    def __getattr__(self, method: str) -> Callable[..., object]:
        if method in _WRITES:
            return partial(self._write, method)
        if method in _READS:
            return partial(self._read, method)
        raise AttributeError(f"the fake database has no {self._name}.{method}")

    async def _write(
        self,
        method: str,
        data: object = None,
        where: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        **_: object,
    ) -> object:
        self._store.writes.append(Write(self._name, method, data))
        if method in ("create_many", "update_many", "delete_many"):
            return 1
        found: Final = self._store.find(self._name, where) if where is not None else None
        if method == "delete":
            return None if found is None else Row(**found)
        given: Final = data if isinstance(data, Mapping) else {}
        sent: Final = given["update" if found is not None else "create"] if method == "upsert" else given
        if found is None and method == "update":
            return None
        row: Final = found if found is not None else self._store.insert(self._name, {})
        self._store.assign(row, sent)
        return self._store.loaded(row, include)

    async def _read(
        self,
        method: str,
        where: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        **_: object,
    ) -> object:
        rows: Final = [
            self._store.loaded(row, include) for row in self._store.rows.get(self._name, ()) if _matches(row, where)
        ]
        if method == "count":
            return len(rows)
        if method in ("find_many", "group_by"):
            return rows
        if not rows and method.endswith("_or_raise"):
            raise LookupError(f"{self._name} has no row matching {where}")
        return rows[0] if rows else None


class _Transaction:
    def __init__(self, store: "RecordingDb") -> None:
        self._store = store

    def __getattr__(self, name: str) -> object:
        return getattr(self._store, name)

    async def __aenter__(self) -> "RecordingDb":
        return self._store

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> bool:
        return False


@dataclass
class RecordingDb:
    rows: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    writes: list[Write] = field(default_factory=list)

    def __getattr__(self, name: str) -> object:
        if name.startswith("litellm_"):
            return _Table(self, name)
        raise AttributeError(f"the fake database has no {name}")

    def find(self, table: str, where: Mapping[str, object]) -> dict[str, object] | None:
        return next((row for row in self.rows.get(table, ()) if _matches(row, where)), None)

    def insert(self, table: str, data: Mapping[str, object]) -> dict[str, object]:
        rows: Final = self.rows.setdefault(table, [])
        primary_key: Final = _PRIMARY_KEYS.get(table)
        row: Final = {
            "created_at": _STORED_AT,
            "updated_at": _STORED_AT,
            "created_by": "admin-1",
            "updated_by": "admin-1",
            **_TABLE_DEFAULTS.get(table, {}),
            **({primary_key: f"{table}-{len(rows) + 1}"} if primary_key is not None else {}),
        }
        self.assign(row, data)
        rows.append(row)
        return row

    def assign(self, row: dict[str, object], data: Mapping[str, object]) -> None:
        for column, value in data.items():
            if column == "object_permission" and _is_relation_write(value):
                row["object_permission_id"] = self._permission_written(row.get("object_permission_id"), value)
            elif not _is_relation_write(value):
                row[str(column)] = _as_stored(value)

    def _permission_written(self, current: object, relation: Mapping[str, object]) -> object:
        if "connect" in relation:
            return relation["connect"]["object_permission_id"]
        stored: Final = self.find("litellm_objectpermissiontable", {"object_permission_id": current})
        upsert: Final = relation.get("upsert", {})
        update: Final = relation.get("update", upsert.get("update"))
        if stored is not None and update is not None:
            self.assign(stored, update)
            return current
        created: Final = relation.get("create", upsert.get("create"))
        if created is None:
            return current
        return self.insert("litellm_objectpermissiontable", created)["object_permission_id"]

    def loaded(self, row: Mapping[str, object], include: Mapping[str, object] | None) -> Row:
        permission: Final = (
            self.find("litellm_objectpermissiontable", {"object_permission_id": row.get("object_permission_id")})
            if (include or {}).get("object_permission") and row.get("object_permission_id") is not None
            else None
        )
        return Row(**row, **({"object_permission": Row(**permission)} if permission is not None else {}))

    async def execute_raw(self, query: str, *args: object) -> int:
        self.writes.append(Write("raw", "execute_raw", query))
        return 0

    async def query_raw(self, query: str, *args: object) -> list[object]:
        if not query.lstrip().upper().startswith(("SELECT", "WITH")):
            self.writes.append(Write("raw", "query_raw", query))
        if query.startswith('SELECT members_with_roles FROM "LiteLLM_TeamTable"'):
            team: Final = self.find("litellm_teamtable", {"team_id": args[0]})
            return [] if team is None else [{"members_with_roles": team["members_with_roles"]}]
        return []

    async def query_first(self, query: str, *args: object) -> None:
        if not query.lstrip().upper().startswith(("SELECT", "WITH")):
            self.writes.append(Write("raw", "query_first", query))

    def tx(self, **_: object) -> _Transaction:
        return _Transaction(self)

    def batch_(self) -> _Transaction:
        return _Transaction(self)


def _hashed(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1", api_key="sk-admin")
MEMBER: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="alice", api_key="sk-alice")


def _seeded_rows() -> dict[str, list[dict[str, object]]]:
    return {
        "litellm_mcpauthprofiletable": [
            {"profile_id": "ana-bots", "profile_id_normalized": "ana-bots", "server_id": "analytics"},
            {"profile_id": "ana-readonly", "profile_id_normalized": "ana-readonly", "server_id": "analytics"},
        ],
        "litellm_objectpermissiontable": [
            {"object_permission_id": f"perm-{entity}", "mcp_servers": [], "mcp_auth_profiles": []}
            for entity in ("key", "team", "user", "organization", "customer", "project", "agent")
        ],
        "litellm_verificationtoken": [
            {
                "token": _hashed("sk-old"),
                "key_alias": "old",
                "user_id": "alice",
                "team_id": "team-1",
                "object_permission_id": "perm-key",
            }
        ],
        "litellm_teamtable": [
            {
                "team_id": "team-1",
                "team_alias": "spruce",
                "members_with_roles": [{"user_id": "alice", "role": "admin"}],
                "object_permission_id": "perm-team",
                "metadata": {},
                "models": [],
            }
        ],
        "litellm_usertable": [{"user_id": "alice", "user_role": "internal_user", "object_permission_id": "perm-user"}],
        "litellm_organizationtable": [
            {
                "organization_id": "org-1",
                "organization_alias": "acme",
                "budget_id": "budget-org",
                "object_permission_id": "perm-organization",
            }
        ],
        "litellm_endusertable": [{"user_id": "c1", "blocked": False, "object_permission_id": "perm-customer"}],
        "litellm_projecttable": [{"project_id": "proj-1", "team_id": "team-1", "object_permission_id": "perm-project"}],
        "litellm_agentstable": [
            {
                "agent_id": "agent-1",
                "agent_name": "helper",
                "litellm_params": {},
                "agent_card_params": {},
                "object_permission_id": "perm-agent",
            }
        ],
    }


class _RustReads:
    def __init__(self, db: RecordingDb) -> None:
        self._db = db

    async def find_with_model_aliases(self, team_id: str, *, use_writer: bool = False) -> str | None:
        row: Final = self._db.find("litellm_teamtable", {"team_id": team_id})
        return None if row is None else LiteLLM_TeamTable.model_validate(row).model_dump_json()

    async def find_by_object_permission_id(self, object_permission_id: str, *, use_writer: bool = False) -> str | None:
        row: Final = self._db.find("litellm_objectpermissiontable", {"object_permission_id": object_permission_id})
        return None if row is None else LiteLLM_ObjectPermissionTable.model_validate(row).model_dump_json()


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> Iterator[RecordingDb]:
    db: Final = RecordingDb()
    for table, rows in _seeded_rows().items():
        for row in rows:
            db.insert(table, row)
    rust_reads: Final = _RustReads(db)
    rust_stores: Final[Mapping[object, object]] = {
        RustTeamRepository: rust_reads,
        RustObjectPermissionRepository: rust_reads,
        RustJwtKeyMappingRepository: FakeJwtKeyMappingStore(),
    }
    monkeypatch.setattr(rust_database, "repository", rust_stores.get)
    client: Final = PrismaClient.__new__(PrismaClient)
    client.db = db  # pyright: ignore[reportAttributeAccessIssue]  # the database is the dependency we fake
    client.proxy_logging_obj = proxy_server.proxy_logging_obj
    monkeypatch.setattr(proxy_server, "prisma_client", client)  # test-quality-ok: substitute the database dependency
    monkeypatch.setattr(proxy_server, "premium_user", True)  # test-quality-ok: enterprise routes need a license
    monkeypatch.setattr(  # test-quality-ok: organizations need a model list to grant from
        proxy_server,
        "llm_router",
        litellm.Router(model_list=[{"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}]),
    )
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache(in_memory_cache=InMemoryCache()))
    monkeypatch.setattr(global_agent_registry, "agent_list", [])
    monkeypatch.setattr(litellm, "default_key_generate_params", None)
    monkeypatch.setattr(litellm, "default_internal_user_params", None)
    yield db
    proxy_server.app.dependency_overrides.pop(user_api_key_auth, None)


def _send(method: str, path: str, body: Mapping[str, object], caller: UserAPIKeyAuth) -> tuple[int, str]:
    proxy_server.app.dependency_overrides[user_api_key_auth] = lambda: caller
    response: Final = TestClient(proxy_server.app).request(
        method, path, json=body, headers={"Authorization": "Bearer sk-test"}
    )
    return response.status_code, response.text


Grant = Sequence[str]
Body = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class Drive:
    method: str
    template: str
    path: str
    body: Body
    variant: str = ""
    member_refusal: str | None = None
    rows_fail_independently: bool = False


_TEAM_ADMIN_FIELD_GATE: Final = "Team admins on this proxy cannot edit team settings"
_AGENTS_ADMIN_ONLY: Final = "Only proxy admins can create, update, or delete agents"


DRIVES: Final[tuple[Drive, ...]] = (
    Drive(
        "POST",
        "/key/generate",
        "/key/generate",
        lambda op: {"user_id": "alice", "soft_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/key/service-account/generate",
        "/key/service-account/generate",
        lambda op: {"team_id": "team-1", "key_alias": "svc", "object_permission": op},
    ),
    Drive(
        "POST",
        "/key/update",
        "/key/update",
        lambda op: {"key": "sk-old", "key_alias": "renamed", "soft_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/key/bulk_update",
        "/key/bulk_update",
        lambda op: {"keys": [{"key": "sk-old", "max_budget": 10.0, "object_permission": op}]},
        member_refusal="Only proxy admins can perform bulk key updates",
        rows_fail_independently=True,
    ),
    Drive(
        "POST",
        "/key/regenerate",
        "/key/regenerate",
        lambda op: {"key": "sk-old", "soft_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/key/{key}/regenerate",
        "/key/sk-old/regenerate",
        lambda op: {"soft_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/team/new",
        "/team/new",
        lambda op: {"team_alias": "birch", "model_aliases": {"fast": "gpt-4o-mini"}, "object_permission": op},
    ),
    Drive(
        "POST",
        "/team/update",
        "/team/update",
        lambda op: {
            "team_id": "team-1",
            "model_aliases": {"fast": "gpt-4o-mini"},
            "team_member_budget": 5.0,
            "object_permission": op,
        },
        member_refusal=_TEAM_ADMIN_FIELD_GATE,
    ),
    Drive(
        "PATCH",
        "/team/{team_id}",
        "/team/team-1",
        lambda op: {"model_aliases": {"fast": "gpt-4o-mini"}, "team_member_budget": 5.0, "object_permission": op},
        member_refusal=_TEAM_ADMIN_FIELD_GATE,
    ),
    Drive(
        "POST",
        "/v1/agents",
        "/v1/agents",
        lambda op: {"agent_name": "fresh", "object_permission": op},
        member_refusal=_AGENTS_ADMIN_ONLY,
    ),
    Drive(
        "PUT",
        "/v1/agents/{agent_id}",
        "/v1/agents/agent-1",
        lambda op: {"agent_name": "helper", "object_permission": op},
        member_refusal=_AGENTS_ADMIN_ONLY,
    ),
    Drive(
        "PATCH",
        "/v1/agents/{agent_id}",
        "/v1/agents/agent-1",
        lambda op: {"object_permission": op},
        member_refusal=_AGENTS_ADMIN_ONLY,
    ),
    Drive("POST", "/user/new", "/user/new", lambda op: {"user_id": "bob", "object_permission": op}),
    Drive(
        "POST",
        "/user/update",
        "/user/update",
        lambda op: {"user_id": "alice", "max_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/user/bulk_update",
        "/user/bulk_update",
        lambda op: {"all_users": True, "user_updates": {"object_permission": op}},
        variant="all-users",
    ),
    Drive(
        "POST",
        "/user/bulk_update",
        "/user/bulk_update",
        lambda op: {"users": [{"user_id": "alice", "object_permission": op}]},
        variant="listed-users",
        rows_fail_independently=True,
    ),
    Drive(
        "POST",
        "/management/v1/users/bulk",
        "/management/v1/users/bulk",
        lambda op: {"users": [{"user_id": "bob", "object_permission": op}]},
        rows_fail_independently=True,
    ),
    Drive(
        "POST",
        "/organization/new",
        "/organization/new",
        lambda op: {
            "organization_alias": "globex",
            "models": ["gpt-4o-mini"],
            "max_budget": 10.0,
            "object_permission": op,
        },
    ),
    Drive(
        "PATCH",
        "/organization/update",
        "/organization/update",
        lambda op: {"organization_id": "org-1", "max_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "PATCH",
        "/v2/organization/{organization_id}",
        "/v2/organization/org-1",
        lambda op: {"max_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/customer/new",
        "/customer/new",
        lambda op: {"user_id": "c2", "max_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/customer/update",
        "/customer/update",
        lambda op: {"user_id": "c1", "max_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/end_user/new",
        "/end_user/new",
        lambda op: {"user_id": "c2", "max_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/end_user/update",
        "/end_user/update",
        lambda op: {"user_id": "c1", "max_budget": 5.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/project/new",
        "/project/new",
        lambda op: {"project_alias": "aspen", "team_id": "team-1", "max_budget": 10.0, "object_permission": op},
    ),
    Drive(
        "POST",
        "/project/update",
        "/project/update",
        lambda op: {"project_id": "proj-1", "max_budget": 5.0, "object_permission": op},
    ),
)


def test_every_matrix_route_is_driven() -> None:
    assert {(drive.method, drive.template) for drive in DRIVES} == set(ROUTE_MATRIX)


def _id(drive: Drive) -> str:
    return f"{drive.method} {drive.template}{f' {drive.variant}' if drive.variant else ''}"


@dataclass(frozen=True)
class BadGrant:
    name: str
    grant: Grant
    caller: UserAPIKeyAuth
    status: int
    says: str


_GRANT_CARRIER_REFUSALS: Final = (
    BadGrant("unknown-profile", ("no-such-profile",), ADMIN, 400, "Unknown MCP auth profile ids"),
    BadGrant("two-profiles-one-server", ("ana-bots", "ana-readonly"), ADMIN, 400, "one profile per MCP server"),
    BadGrant("non-admin-changes-grant", ("ana-bots",), MEMBER, 403, "Only a proxy admin can change mcp_auth_profiles"),
    BadGrant("non-admin-unknown-profile", ("no-such-profile",), MEMBER, 403, "Only a proxy admin can change"),
    BadGrant("non-admin-two-profiles-one-server", ("ana-bots", "ana-readonly"), MEMBER, 403, "Only a proxy admin"),
)


def _documented_refusal(drive: Drive, bad: BadGrant) -> BadGrant:
    if bad.caller is MEMBER and drive.member_refusal is not None:
        return replace(bad, says=drive.member_refusal)
    if bad.status == 400 and drive.rows_fail_independently:
        return replace(bad, status=200)
    return bad


def _refusals(drive: Drive) -> tuple[BadGrant, ...]:
    entity: Final = ROUTE_MATRIX[(drive.method, drive.template)].entity
    documented: Final = (
        _GRANT_CARRIER_REFUSALS
        if entity in ("key", "team", "agent")
        else (BadGrant("any-grant", ("ana-bots",), ADMIN, 400, f"not on a {entity}"),)
    )
    return tuple(_documented_refusal(drive, bad) for bad in documented)


def _refusal_cases() -> Iterator[object]:
    for drive in DRIVES:
        for bad in _refusals(drive):
            yield pytest.param(drive, bad, id=f"{_id(drive)} {bad.name}")


@pytest.mark.parametrize(("drive", "bad"), tuple(_refusal_cases()))
def test_a_bad_grant_is_refused_before_any_write(database: RecordingDb, drive: Drive, bad: BadGrant) -> None:
    status, text = _send(drive.method, drive.path, drive.body({"mcp_auth_profiles": list(bad.grant)}), bad.caller)

    assert (status, bad.says in text, database.writes) == (bad.status, True, []), text


def test_a_member_resending_the_stored_key_grant_never_writes_the_grant(database: RecordingDb) -> None:
    stored: Final = database.find("litellm_objectpermissiontable", {"object_permission_id": "perm-key"})
    assert stored is not None
    stored["mcp_auth_profiles"] = ["ana-bots"]
    key_update: Final = next(drive for drive in DRIVES if drive.template == "/key/update")

    status, text = _send(
        key_update.method, key_update.path, key_update.body({"mcp_auth_profiles": ["ana-bots"]}), MEMBER
    )

    permission_writes: Final = [write for write in database.writes if write.table == "litellm_objectpermissiontable"]
    assert status == 200, text
    assert permission_writes, "the resend must still write the rest of the permission"
    assert all("mcp_auth_profiles" not in str(write.data) for write in permission_writes), permission_writes


def test_a_member_creating_a_key_with_an_explicit_empty_grant_gets_no_default_grant(
    database: RecordingDb, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        litellm, "default_key_generate_params", {"object_permission": {"mcp_auth_profiles": ["ana-bots"]}}
    )
    key_generate: Final = next(drive for drive in DRIVES if drive.template == "/key/generate")

    status, text = _send(key_generate.method, key_generate.path, key_generate.body({"mcp_auth_profiles": []}), MEMBER)

    created: Final = [
        TypeAdapter(Mapping[str, object]).validate_python(write.data)
        for write in database.writes
        if (write.table, write.method) == ("litellm_objectpermissiontable", "create")
    ]
    assert status == 200, text
    assert [permission.get("mcp_auth_profiles") for permission in created] == [[]]


@pytest.mark.parametrize(
    ("caller", "sent", "default", "stored"),
    [
        (ADMIN, {"mcp_auth_profiles": []}, ["old-bot"], []),
        (MEMBER, {"mcp_auth_profiles": []}, ["old-bot"], []),
        (ADMIN, {"mcp_servers": []}, ["ANA-Bots"], ["ana-bots"]),
        (MEMBER, {"mcp_servers": []}, ["ANA-Bots"], ["ana-bots"]),
    ],
    ids=[
        "admin-overrides-unknown-default",
        "member-overrides-unknown-default",
        "admin-gets-default",
        "member-gets-default",
    ],
)
def test_a_key_is_written_with_the_default_grant_only_where_the_caller_left_it_unset(
    database: RecordingDb,
    monkeypatch: pytest.MonkeyPatch,
    caller: UserAPIKeyAuth,
    sent: Mapping[str, object],
    default: list[str],
    stored: list[str],
) -> None:
    monkeypatch.setattr(litellm, "default_key_generate_params", {"object_permission": {"mcp_auth_profiles": default}})

    status, text = _send("POST", "/key/generate", {"object_permission": sent}, caller)

    created: Final = [
        TypeAdapter(Mapping[str, object]).validate_python(write.data)
        for write in database.writes
        if (write.table, write.method) == ("litellm_objectpermissiontable", "create")
    ]
    assert status == 200, text
    assert [permission.get("mcp_auth_profiles") for permission in created] == [stored]


@dataclass(frozen=True)
class DefaultGrant:
    setting: Literal["default_key_generate_params", "default_internal_user_params"]
    grant: Grant
    status: int
    says: str


_DEFAULT_GRANT_CASES: Final = (
    *(
        pytest.param(
            drive,
            DefaultGrant("default_key_generate_params", bad.grant, 400, bad.says),
            id=f"{_id(drive)} default_key_generate_params {bad.name}",
        )
        for drive in DRIVES
        if drive.template in ("/key/generate", "/key/service-account/generate")
        for bad in _GRANT_CARRIER_REFUSALS[:2]
    ),
    *(
        pytest.param(
            drive,
            DefaultGrant(
                "default_internal_user_params",
                ("ana-bots",),
                200 if drive.rows_fail_independently else 400,
                "not on a user",
            ),
            id=f"{_id(drive)} default_internal_user_params",
        )
        for drive in DRIVES
        if drive.template in ("/user/new", "/management/v1/users/bulk")
    ),
)


def _without_permission(body: Mapping[str, object]) -> Mapping[str, object]:
    return {
        field_name: [_without_permission(item) if isinstance(item, Mapping) else item for item in value]
        if isinstance(value, list)
        else value
        for field_name, value in body.items()
        if field_name != "object_permission"
    }


@pytest.mark.parametrize(("drive", "default"), _DEFAULT_GRANT_CASES)
def test_a_bad_grant_from_the_configured_defaults_is_refused_before_any_write(
    database: RecordingDb, monkeypatch: pytest.MonkeyPatch, drive: Drive, default: DefaultGrant
) -> None:
    monkeypatch.setattr(litellm, default.setting, {"object_permission": {"mcp_auth_profiles": list(default.grant)}})

    status, text = _send(drive.method, drive.path, _without_permission(drive.body({})), ADMIN)

    assert (status, default.says in text, database.writes) == (default.status, True, []), text


def _keys_anywhere(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _keys_anywhere(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys_anywhere(item)


def _no_grant_cases() -> Iterator[object]:
    for drive in DRIVES:
        marks: Final = (
            pytest.mark.xfail(
                strict=True,
                reason="answers 500 on the base commit too: update_project validates the permission after "
                "jsonify_object turned it into a JSON string",
            )
            if drive.template == "/project/update"
            else ()
        )
        yield pytest.param(drive, id=_id(drive), marks=marks)


@pytest.mark.parametrize("drive", tuple(_no_grant_cases()))
def test_a_permission_without_a_grant_writes_as_before_and_answers_without_one(
    database: RecordingDb, drive: Drive
) -> None:
    status, text = _send(drive.method, drive.path, drive.body({"mcp_servers": []}), ADMIN)

    assert status == 200, text
    assert database.writes, "the no-grant request must write, or the refusal cases prove nothing"
    assert "mcp_auth_profiles" not in set(_keys_anywhere(json.loads(text)))
