import json

import pytest

from litellm.proxy.admin_mcp.results import ResultError, ResultStore, contains_credential, encode, page, select


def rebuild(item, token, path=""):
    """Use only the public index/continuations, without knowing field names."""
    first = page(item, token, path=path, limit=3, max_chars=257)
    result = (
        {}
        if first["type"] == "object"
        else []
        if first["type"] == "array"
        else ""
        if first["type"] == "string"
        else first["value"]
    )
    current = first
    while True:
        if first["type"] in {"object", "array"}:
            for entry in current["entries"]:
                value = entry["value"] if entry["complete"] else rebuild(item, token, entry["read"]["path"])
                if isinstance(result, dict):
                    result[entry["key"]] = value
                else:
                    result.append(value)
        elif first["type"] == "string":
            result += current["value"]
        if not current.get("next"):
            return result
        args = dict(current["next"])
        args.pop("result_id")
        current = page(item, token, **args)


def test_index_and_pages_reconstruct_every_value_without_prior_field_knowledge():
    original = {
        "user_info": {"role": "proxy_admin", "max_budget": None, "spend": 0, "blocked": False},
        "keys": [
            {"id": n, "rare_setting": {"model/a~b": n}, "metadata": {"new_field": "x" * 17100}} for n in range(17)
        ],
        "a/b~c": {"": "🎉e\u0301" * 2700, "empty_object": {}, "empty_array": [], "empty_string": ""},
        "0": [True, False, None, 0, -1, 1.25],
    }
    store = ResultStore()
    token = store.put("get_user", original, "alice", "user-1")
    item = store.get(token, "alice", "user-1")
    first = page(item, token)
    profile = next(e for e in first["entries"] if e["key"] == "user_info")
    assert profile["complete"] and profile["value"] == original["user_info"]
    assert rebuild(item, token) == original
    assert json.loads(item.raw) == original
    assert len(encode(first)) < len(item.raw) / 4


def test_record_previews_include_all_small_fields_including_uncommon_ones():
    value = [
        {
            "id": "team-1",
            "custom_policy": False,
            "budget": None,
            "spend": 0,
            "new_config": {"special": True},
            "keys": ["x" * 500 for _ in range(100)],
        }
    ]
    store = ResultStore()
    token = store.put("list_teams", value, "alice", "user-1")
    result = page(store.get(token, "alice", "user-1"), token)
    entry = result["entries"][0]
    assert not entry["complete"]
    assert entry["preview"] == {k: v for k, v in value[0].items() if k != "keys"}
    assert entry["omitted_fields"] == ["keys"]


def test_page_boundary_moves_whole_records_to_next_page_instead_of_fragmenting():
    value = [{"id": n, "data": "x" * 7000} for n in range(7)]
    store = ResultStore()
    token = store.put("list_keys", value, "alice", "user-1")
    item = store.get(token, "alice", "user-1")
    current, seen, calls = page(item, token), [], 0
    while True:
        calls += 1
        for entry in current["entries"]:
            assert entry["complete"]
            seen.append(entry["value"])
        if not current["next"]:
            break
        args = {k: v for k, v in current["next"].items() if k != "result_id"}
        current = page(item, token, **args)
    assert seen == value and calls == 4


def test_snapshot_is_immutable_and_bound_to_both_credential_and_principal():
    value = {"data": [1, 2]}
    store = ResultStore()
    token = store.put("list_keys", value, "alice-key", "user-1")
    value["data"].append(3)
    assert json.loads(store.get(token, "alice-key", "user-1").raw) == {"data": [1, 2]}
    for credential, principal in [("bob-key", "user-1"), ("alice-key", "user-2")]:
        with pytest.raises(ResultError, match="unavailable"):
            store.get(token, credential, principal)


def test_expiration_capacity_and_clear_do_not_evict_unexpired_results():
    now = [100]
    store = ResultStore(ttl=10, max_bytes=100, max_per_owner=1, clock=lambda: now[0], wall_clock=lambda: 1000)
    token = store.put("get_user", {"a": 1}, "alice", "user-1")
    assert store.put("get_user", {"b": 2}, "alice", "user-1") is None
    assert store.put("get_user", {"a": "x" * 100}, "bob", "user-2") is None
    assert store.get(token, "alice", "user-1")
    now[0] = 111
    with pytest.raises(ResultError, match="expired"):
        store.get(token, "alice", "user-1")
    replacement = store.put("get_user", {"a": 1}, "alice", "user-1")
    assert json.loads(store.get(replacement, "alice", "user-1").raw) == {"a": 1}
    store.clear()
    with pytest.raises(ResultError, match="unavailable"):
        store.get(replacement, "alice", "user-1")


@pytest.mark.parametrize(
    "source,value",
    [
        ("create_key", {"key": "custom-key-without-prefix"}),
        ("regenerate_key", {"key": "new"}),
        ("regenerate_key_by_id", {"key": "new"}),
        ("create_service_account_key", {"key": "new"}),
        ("get_key", {"key": "custom-key-without-prefix", "info": {}}),
        ("list_request_logs", {"history": "use sk-something here"}),
        ("get_key", {"metadata": json.dumps({"key": "custom-secret"})}),
        ("list_models", {"data": [{"model_info": {"key": "sk-sensitive"}}]}),
        ("get_model", {"data": [{"model_info": {"key": "sk-sensitive"}}]}),
        ("list_models", {"data": [{"model_info": {"key": json.dumps({"key": "custom-secret"})}}]}),
        ("list_models", {"data": [{"model_info": {"key": "openai/example", "metadata": {"key": "custom-secret"}}}]}),
        ("list_models", {"data": [{"key": "custom-secret"}]}),
        ("list_models", {"model_info": {"key": "custom-secret"}}),
        ("list_request_logs", {"data": [{"model_info": {"key": "custom-secret"}}]}),
    ],
)
def test_credentials_are_not_retained_in_result_store(source, value):
    store = ResultStore()
    assert store.put(source, value, "alice", "user-1") is None


@pytest.mark.parametrize(
    "path", ["not-a-pointer", "/bad~2escape", "/array/-1", "/array/01", "/array/4", "/missing", "/scalar/x"]
)
def test_invalid_paths_fail_explicitly(path):
    with pytest.raises(ResultError):
        select({"array": [1], "scalar": 0}, path)


def test_empty_values_and_pagination_edges():
    store = ResultStore()
    for value in ([], {}, "", False, 0, None):
        token = store.put("get_user", value, "alice", "user-1")
        item = store.get(token, "alice", "user-1")
        assert page(item, token)["complete"]
        assert rebuild(item, token) == value
        with pytest.raises(ResultError):
            page(item, token, offset=1)


def test_deep_result_paths_and_credential_scan_do_not_use_python_call_stack():
    value = {"password": "private-token"}
    for _ in range(1200):
        value = {"nested": value}
    assert select(value, "/nested" * 1200 + "/password") == "private-token"
    assert contains_credential(value)


@pytest.mark.parametrize(
    "payload",
    [
        "[" * 250 + '{"password":"provider-secret"}' + "]" * 250,
        '  {"password":"provider-secret"',
        '[{"password":"provider-secret"}',
    ],
)
def test_unparseable_json_containers_are_never_retained(payload):
    assert contains_credential({"metadata": payload})
    assert ResultStore().put("get_user", {"metadata": payload}, "alice", "user-1") is None
