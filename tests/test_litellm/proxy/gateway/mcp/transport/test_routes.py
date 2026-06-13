from litellm.proxy.gateway.mcp.transport.routes import RouteTarget, parse_route


def test_aggregated_mcp_yields_no_server():
    assert parse_route("/mcp") == RouteTarget(server=None)


def test_aggregated_mcp_trailing_slash():
    assert parse_route("/mcp/") == RouteTarget(server=None)


def test_single_server_mcp():
    assert parse_route("/github/mcp") == RouteTarget(server="github")


def test_single_server_mcp_trailing_slash():
    assert parse_route("/github/mcp/") == RouteTarget(server="github")


def test_root_is_not_an_mcp_endpoint():
    assert parse_route("/") is None


def test_unknown_top_level_path():
    assert parse_route("/foo") is None


def test_too_many_segments():
    assert parse_route("/a/b/c/mcp") is None


def test_server_without_mcp_suffix():
    assert parse_route("/github/tools") is None


def test_valid_sep986_server_name_is_accepted():
    assert parse_route("/git-hub.v2_1/mcp") == RouteTarget(server="git-hub.v2_1")


def test_malformed_server_segment_is_rejected():
    assert parse_route("/foo bar/mcp") is None
    assert parse_route("/has/slash/mcp") is None
