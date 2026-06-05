from litellm.proxy.auth.v2.route_map import match_route


def test_model_routes_map_to_resource_and_action():
    assert match_route("/model/new").resource == "model"
    assert match_route("/model/new").action == "write"
    assert match_route("/model/update").action == "write"
    assert match_route("/model/delete").action == "delete"
    assert match_route("/model/info").action == "read"


def test_update_and_delete_carry_id_fields():
    assert match_route("/model/update").id_fields == ["model_id", "id"]
    assert match_route("/model/delete").id_fields == ["model_id", "id"]


def test_create_has_no_id_field():
    assert match_route("/model/new").id_fields == []


def test_trailing_slash_is_normalized():
    assert match_route("/model/info/").resource == "model"


def test_ungoverned_routes_return_none():
    # These are loud-open in slice 1 and must not be governed yet.
    for route in ("/chat/completions", "/key/generate", "/team/new", "/v1/models", "/"):
        assert match_route(route) is None
