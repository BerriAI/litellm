from litellm.proxy.auth.v2.route_map import is_inference_route, match_route


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


def test_team_routes_map_to_team_resource():
    assert match_route("/team/new").resource == "team"
    assert match_route("/team/new").action == "write"
    assert match_route("/team/update").action == "write"
    assert match_route("/team/delete").action == "delete"
    assert match_route("/team/info").action == "read"
    assert match_route("/team/delete").id_fields == ["team_id", "id"]


def test_trailing_slash_is_normalized():
    assert match_route("/model/info/").resource == "model"


def test_inference_routes_are_detected():
    for route in (
        "/chat/completions",
        "/v1/chat/completions",
        "/embeddings",
        "/v1/embeddings",
        "/completions",
        "/responses",
    ):
        assert is_inference_route(route) is True


def test_non_inference_routes_are_not_inference():
    for route in ("/model/new", "/team/info", "/key/generate", "/"):
        assert is_inference_route(route) is False


def test_inference_routes_are_not_control_plane_governed():
    # Inference is data-plane (model attribute), not in the RBAC route map.
    assert match_route("/chat/completions") is None


def test_key_user_org_resources_are_governed():
    assert match_route("/key/generate") == match_route("/key/generate")
    assert match_route("/key/generate").resource == "key"
    assert match_route("/key/delete").action == "delete"
    assert match_route("/key/info").action == "read"
    assert match_route("/user/new").resource == "user"
    assert match_route("/user/delete").action == "delete"
    assert match_route("/organization/update").resource == "organization"


def test_membership_changes_are_the_manage_action():
    assert match_route("/team/member_add").resource == "team"
    assert match_route("/team/member_add").action == "manage"
    assert match_route("/team/member_delete").action == "manage"
    assert match_route("/organization/member_add").action == "manage"


def test_ungoverned_routes_return_none():
    # Genuinely not yet owned by v2: loud-open.
    for route in ("/v1/models", "/health", "/"):
        assert match_route(route) is None
