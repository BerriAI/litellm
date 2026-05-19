"""Tests for /openapi-public.json filtering (S5-02)."""

from litellm.proxy.capability_endpoints.openapi_public import _filter_openapi


def _mock_schema():
    return {
        "openapi": "3.1.0",
        "info": {"title": "litellm proxy", "version": "1.0"},
        "components": {
            "schemas": {
                "ChatCompletionRequest": {"type": "object"},
                "PrivateAdminPayload": {"type": "object"},  # never referenced
            }
        },
        "paths": {
            "/v1/chat/completions": {
                "post": {
                    "operationId": "chat_completions",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ChatCompletionRequest"
                                }
                            }
                        }
                    },
                }
            },
            "/v1/capabilities": {"get": {"operationId": "get_capabilities"}},
            "/v1/agents": {
                "get": {"operationId": "list_agents"},
                "post": {
                    "operationId": "create_agent"
                },  # admin-only — should be DROPPED
                "delete": {
                    "operationId": "delete_agent"
                },  # admin-only — should be DROPPED
            },
            "/v1/xct-skills/{skill_id}": {
                "get": {"operationId": "get_skill"},
                "patch": {"operationId": "patch_skill"},  # write — DROPPED
                "delete": {"operationId": "delete_skill"},  # write — DROPPED
            },
            "/v1/xct-skills/{skill_id}/publish": {
                "post": {"operationId": "publish_skill"}
            },  # admin-only path entirely — DROPPED
            "/v1/xct-apps": {"post": {"operationId": "create_app"}},  # ENTIRELY admin
            "/key/generate": {"post": {"operationId": "generate_key"}},  # not allowed
            "/oauth/token": {"post": {"operationId": "oauth_token"}},
        },
    }


def test_filter_keeps_public_data_plane():
    filtered = _filter_openapi(_mock_schema())
    paths = set(filtered["paths"].keys())
    assert "/v1/chat/completions" in paths
    assert "/v1/capabilities" in paths
    assert "/oauth/token" in paths


def test_filter_drops_admin_only_paths():
    filtered = _filter_openapi(_mock_schema())
    paths = filtered["paths"]
    assert "/v1/xct-apps" not in paths  # never publicly accessible
    assert "/key/generate" not in paths
    assert "/v1/xct-skills/{skill_id}/publish" not in paths


def test_filter_drops_write_methods_on_restricted_paths():
    filtered = _filter_openapi(_mock_schema())
    agents = filtered["paths"]["/v1/agents"]
    # Only GET kept.
    assert "get" in agents
    assert "post" not in agents
    assert "delete" not in agents
    # Skills detail: GET kept, PATCH/DELETE dropped.
    skill = filtered["paths"]["/v1/xct-skills/{skill_id}"]
    assert "get" in skill
    assert "patch" not in skill
    assert "delete" not in skill


def test_filter_prunes_unreferenced_component_schemas():
    filtered = _filter_openapi(_mock_schema())
    schemas = filtered["components"]["schemas"]
    # Referenced by /v1/chat/completions → kept
    assert "ChatCompletionRequest" in schemas
    # Not referenced by any kept path → pruned
    assert "PrivateAdminPayload" not in schemas


def test_filter_info_carries_xct_marker():
    filtered = _filter_openapi(_mock_schema())
    assert filtered["info"]["title"] == "xct-litellm public API"
    assert "x-xct-doc" in filtered["info"]


def test_filter_preserves_servers_if_set():
    schema = _mock_schema()
    schema["servers"] = [{"url": "https://litellm.xct.test"}]
    filtered = _filter_openapi(schema)
    assert filtered["servers"] == [{"url": "https://litellm.xct.test"}]
