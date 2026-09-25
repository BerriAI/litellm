import pytest

from liteadmin.catalog import OPERATIONS
from liteadmin.results import project_result


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("key_info", {"key": "sk-raw-key"}),
        ("team_update", {"team_id": "team-1", "max_budget": -1}),
        ("user_create", {"user_email": "not-an-email"}),
        ("user_create", {"user_id": "user-1", "auto_create_key": True}),
        ("keys_list", {"page": 1, "page_size": 51}),
        ("team_delete", {"team_ids": []}),
        ("spend_report", {"start_date": "2026-09-26", "end_date": "2026-09-25", "group_by": "team"}),
        ("spend_report", {"start_date": "2024-01-01", "end_date": "2026-09-25", "group_by": "team"}),
    ],
)
def test_invalid_administrative_arguments_are_rejected(name, arguments):
    operation = next(item for item in OPERATIONS if item.name == name)
    assert isinstance(operation.arguments(arguments), str)


def test_new_users_never_implicitly_generate_a_key():
    operation = next(item for item in OPERATIONS if item.name == "user_create")
    assert operation.arguments({"user_id": "user-1", "models": None}) == {"user_id": "user-1", "auto_create_key": False}


def test_request_logs_keep_only_operational_fields():
    assert project_result(
        "log",
        {
            "data": [
                {
                    "request_id": "r-1",
                    "spend": 0.12,
                    "prompt": "private",
                    "messages": [{"content": "private"}],
                    "response": "secret",
                    "api_key": "sk-private",
                    "cache_hit": True,
                }
            ],
            "total": 1,
        },
        (),
    ) == {"data": [{"request_id": "r-1", "spend": 0.12, "cache_hit": True}], "total": 1}


def test_nested_team_keys_keep_hashes_but_never_raw_credentials():
    key_hash = "a" * 64
    assert project_result(
        "team",
        {
            "team_info": {
                "team_id": "team-1",
                "members_with_roles": [{"user_id": "user-1", "role": "admin", "password": "secret"}],
            },
            "keys": [{"token": key_hash, "key": "sk-private", "key_alias": "Bearer secret-token"}],
        },
        (),
    ) == {
        "team_info": {"team_id": "team-1", "members_with_roles": [{"user_id": "user-1", "role": "admin"}]},
        "keys": [{"token": key_hash, "key_alias": "[redacted]"}],
    }


def test_result_projection_bounds_strings_rows_and_total_output():
    result = project_result(
        "user", {"users": [{"user_id": str(index), "user_alias": "a" * 500} for index in range(60)]}, ()
    )
    assert isinstance(result, dict)
    assert len(result["users"]) == 50
    assert len(result["users"][0]["user_alias"]) == 400
    huge = project_result("log", {"data": [{"request_id": "x" * 400, "model": "y" * 400} for _ in range(50)]}, ())
    assert huge == {"notice": "The result is too large to summarize safely. Use a smaller page or narrower filters."}
