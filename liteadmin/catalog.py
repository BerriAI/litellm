from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

from jsonschema import FormatChecker, ValidationError, validate
from pydantic import JsonValue

ResultKind = Literal["key", "team", "user", "budget", "spend", "log"]


def object_schema(properties: Mapping[str, JsonValue], required: tuple[str, ...] = ()) -> dict[str, JsonValue]:
    return {"type": "object", "properties": dict(properties), "required": list(required), "additionalProperties": False}


def nullable(value: JsonValue) -> JsonValue:
    return {"anyOf": [value, {"type": "null"}]}


def array(value: JsonValue, maximum: int = 50, minimum: int = 0) -> JsonValue:
    return {"type": "array", "items": value, "maxItems": maximum, "minItems": minimum}


TEXT: Final[JsonValue] = {"type": "string", "minLength": 1, "maxLength": 200}
HASH: Final[JsonValue] = {"type": "string", "pattern": "^[a-fA-F0-9]{64}$"}
OPTIONAL_TEXT: Final = nullable(TEXT)
LIMITS: Final[dict[str, JsonValue]] = {
    "max_budget": nullable({"type": "number", "minimum": 0}),
    "budget_duration": OPTIONAL_TEXT,
    "rpm_limit": nullable({"type": "integer", "minimum": 0}),
    "tpm_limit": nullable({"type": "integer", "minimum": 0}),
}
POLICY: Final[dict[str, JsonValue]] = {**LIMITS, "models": nullable(array(TEXT))}
PAGING: Final[dict[str, JsonValue]] = {
    "page": {"type": "integer", "minimum": 1},
    "page_size": {"type": "integer", "minimum": 1, "maximum": 50},
    "search": OPTIONAL_TEXT,
}
KEY: Final[dict[str, JsonValue]] = {
    **POLICY,
    "key_alias": OPTIONAL_TEXT,
    "team_id": OPTIONAL_TEXT,
    "user_id": OPTIONAL_TEXT,
    "budget_id": OPTIONAL_TEXT,
    "duration": OPTIONAL_TEXT,
}
TEAM: Final[dict[str, JsonValue]] = {**POLICY, "team_alias": OPTIONAL_TEXT, "organization_id": OPTIONAL_TEXT}
USER: Final[dict[str, JsonValue]] = {
    **POLICY,
    "user_alias": OPTIONAL_TEXT,
    "user_email": nullable({"type": "string", "format": "email", "maxLength": 200}),
    "user_role": nullable({"enum": ["proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer"]}),
}
DATES: Final[dict[str, JsonValue]] = {
    "start_date": {"type": "string", "format": "date"},
    "end_date": {"type": "string", "format": "date"},
}
MEMBER_LIMITS: Final[dict[str, JsonValue]] = {
    "max_budget_in_team": nullable({"type": "number", "minimum": 0}),
    "budget_duration": OPTIONAL_TEXT,
}


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    title: str
    path: str
    mode: Literal["read", "write", "delete"]
    kind: ResultKind
    schema: dict[str, JsonValue]
    method: Literal["GET", "POST"] = "POST"

    def arguments(self, value: dict[str, JsonValue]) -> dict[str, JsonValue] | str:
        try:
            validate(instance=value, schema=self.schema, format_checker=FormatChecker())
        except ValidationError:
            return "Invalid tool arguments. Use the tool's schema."
        start: Final = value.get("start_date")
        end: Final = value.get("end_date")
        if (
            isinstance(start, str)
            and isinstance(end, str)
            and not 0 <= (date.fromisoformat(end) - date.fromisoformat(start)).days <= 366
        ):
            return "Choose an ordered date range of at most 366 days."
        clean: Final[dict[str, JsonValue]] = {key: item for key, item in value.items() if item is not None}
        return {**clean, "auto_create_key": False} if self.name == "user_create" else clean

    def request_arguments(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        match self.name:
            case "keys_list":
                return {
                    **{key: value for key, value in arguments.items() if key != "page_size"},
                    "size": arguments["page_size"],
                    "return_full_object": True,
                }
            case "team_info":
                return {**arguments, "key_limit": 20}
            case "budgets_list":
                return {
                    **{key: value for key, value in arguments.items() if key != "search"},
                    **({"q": arguments["search"]} if "search" in arguments else {}),
                }
            case "request_logs":
                return {
                    **arguments,
                    "start_date": f"{arguments['start_date']} 00:00:00",
                    "end_date": f"{arguments['end_date']} 23:59:59",
                }
            case _:
                return arguments


OPERATIONS: Final = (
    Operation(
        "keys_list",
        "List virtual keys",
        "/key/list",
        "read",
        "key",
        object_schema({**PAGING, "team_id": OPTIONAL_TEXT, "user_id": OPTIONAL_TEXT}, ("page", "page_size")),
        "GET",
    ),
    Operation(
        "key_info", "View a virtual key", "/key/info", "read", "key", object_schema({"key": HASH}, ("key",)), "GET"
    ),
    Operation("key_create", "Create a virtual key", "/key/generate", "write", "key", object_schema(KEY)),
    Operation(
        "key_update",
        "Update a virtual key",
        "/key/update",
        "write",
        "key",
        object_schema({**KEY, "key": HASH}, ("key",)),
    ),
    Operation(
        "key_delete",
        "Delete virtual keys",
        "/key/delete",
        "delete",
        "key",
        object_schema({"keys": array(HASH, 20, 1)}, ("keys",)),
    ),
    Operation("key_block", "Block a virtual key", "/key/block", "write", "key", object_schema({"key": HASH}, ("key",))),
    Operation(
        "key_unblock", "Unblock a virtual key", "/key/unblock", "write", "key", object_schema({"key": HASH}, ("key",))
    ),
    Operation(
        "teams_list",
        "List teams",
        "/v2/team/list",
        "read",
        "team",
        object_schema({**PAGING, "organization_id": OPTIONAL_TEXT}, ("page", "page_size")),
        "GET",
    ),
    Operation(
        "team_info",
        "View a team and its members",
        "/team/info",
        "read",
        "team",
        object_schema({"team_id": TEXT}, ("team_id",)),
        "GET",
    ),
    Operation("team_create", "Create a team", "/team/new", "write", "team", object_schema(TEAM)),
    Operation(
        "team_update",
        "Update a team",
        "/team/update",
        "write",
        "team",
        object_schema({**TEAM, "team_id": TEXT}, ("team_id",)),
    ),
    Operation(
        "team_delete",
        "Delete teams",
        "/team/delete",
        "delete",
        "team",
        object_schema({"team_ids": array(TEXT, 20, 1)}, ("team_ids",)),
    ),
    Operation(
        "team_member_add",
        "Add a team member",
        "/team/member_add",
        "write",
        "team",
        object_schema(
            {
                **MEMBER_LIMITS,
                "team_id": TEXT,
                "member": object_schema({"user_id": TEXT, "role": {"enum": ["admin", "user"]}}, ("user_id", "role")),
            },
            ("team_id", "member"),
        ),
    ),
    Operation(
        "team_member_update",
        "Update a team member",
        "/team/member_update",
        "write",
        "team",
        object_schema(
            {
                **MEMBER_LIMITS,
                "rpm_limit": LIMITS["rpm_limit"],
                "tpm_limit": LIMITS["tpm_limit"],
                "team_id": TEXT,
                "user_id": TEXT,
                "role": nullable({"enum": ["admin", "user"]}),
            },
            ("team_id", "user_id"),
        ),
    ),
    Operation(
        "team_member_delete",
        "Remove a team member",
        "/team/member_delete",
        "delete",
        "team",
        object_schema({"team_id": TEXT, "user_id": TEXT}, ("team_id", "user_id")),
    ),
    Operation(
        "users_list", "List users", "/user/list", "read", "user", object_schema(PAGING, ("page", "page_size")), "GET"
    ),
    Operation(
        "user_info",
        "View a user",
        "/v2/user/info",
        "read",
        "user",
        object_schema({"user_id": TEXT}, ("user_id",)),
        "GET",
    ),
    Operation(
        "user_create",
        "Create a user without a key",
        "/user/new",
        "write",
        "user",
        object_schema({**USER, "user_id": OPTIONAL_TEXT}),
    ),
    Operation(
        "user_update",
        "Update a user",
        "/user/update",
        "write",
        "user",
        object_schema({**USER, "user_id": TEXT}, ("user_id",)),
    ),
    Operation(
        "user_delete",
        "Delete users",
        "/user/delete",
        "delete",
        "user",
        object_schema({"user_ids": array(TEXT, 20, 1)}, ("user_ids",)),
    ),
    Operation(
        "budgets_list",
        "List budgets",
        "/management/v1/budgets",
        "read",
        "budget",
        object_schema(PAGING, ("page", "page_size")),
        "GET",
    ),
    Operation(
        "budget_info",
        "View budgets",
        "/budget/info",
        "read",
        "budget",
        object_schema({"budgets": array(TEXT, 20, 1)}, ("budgets",)),
    ),
    Operation(
        "budget_create",
        "Create a budget",
        "/budget/new",
        "write",
        "budget",
        object_schema({**LIMITS, "budget_id": OPTIONAL_TEXT}),
    ),
    Operation(
        "budget_update",
        "Update a budget",
        "/budget/update",
        "write",
        "budget",
        object_schema({**LIMITS, "budget_id": TEXT}, ("budget_id",)),
    ),
    Operation(
        "budget_delete", "Delete a budget", "/budget/delete", "delete", "budget", object_schema({"id": TEXT}, ("id",))
    ),
    Operation(
        "spend_report",
        "View spend by date and group (requires an enterprise license)",
        "/global/spend/report",
        "read",
        "spend",
        object_schema(
            {
                **DATES,
                "group_by": {"enum": ["team", "customer", "api_key"]},
                "api_key": nullable(HASH),
                "team_id": OPTIONAL_TEXT,
                "internal_user_id": OPTIONAL_TEXT,
                "customer_id": OPTIONAL_TEXT,
            },
            ("start_date", "end_date", "group_by"),
        ),
        "GET",
    ),
    Operation(
        "team_spend_report",
        "View a team's spend by model and key (requires an enterprise license)",
        "/team/spend/report",
        "read",
        "spend",
        object_schema({**DATES, "team_id": TEXT}, ("start_date", "end_date", "team_id")),
        "GET",
    ),
    Operation(
        "key_spend_report",
        "View a key's spend by model (requires an enterprise license)",
        "/key/spend/report",
        "read",
        "spend",
        object_schema({**DATES, "api_key": HASH}, ("start_date", "end_date", "api_key")),
        "GET",
    ),
    Operation(
        "request_logs",
        "List request cost, timing and status, excluding prompts and responses",
        "/spend/logs/ui",
        "read",
        "log",
        object_schema(
            {
                **DATES,
                **{key: value for key, value in PAGING.items() if key != "search"},
                "request_id": OPTIONAL_TEXT,
                "team_id": OPTIONAL_TEXT,
                "user_id": OPTIONAL_TEXT,
                "model": OPTIONAL_TEXT,
                "status_filter": nullable({"enum": ["success", "failure"]}),
            },
            ("start_date", "end_date", "page", "page_size"),
        ),
        "GET",
    ),
)
