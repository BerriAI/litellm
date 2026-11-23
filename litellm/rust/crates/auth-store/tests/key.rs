//! The key's database read against a Prisma-migrated Postgres. The row and entry literals are the
//! ones `tests/unit/redis/stores/test_auth_cache_key.py` runs Python's own models and key writer
//! on, so this crate and Python are pinned to the same bytes. Change both together.

use std::{str::FromStr, sync::Arc};

use litellm_auth_store::KeyDatabase;
use litellm_db::{Endpoint, Observer, PoolLimits, Pools, QueryEvent, SslMode};
use litellm_db_testing::MigratedPostgres;
use litellm_gateway_auth_core::KeyHash;
use litellm_redis_store::IntoValue;
use litellm_repositories::{Database, KeyRepository, ObjectPermissionRepository, Side};
use litellm_secret_value::SecretValue;
use rstest::rstest;
use sqlx::{PgPool, postgres::PgConnectOptions};

const BARE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const TEAM_KEY: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const FULL: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const ROTATED: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const STAMP: &str = "1763546400.5";

const BARE_ROW: &str = concat!(
    r#"{"token":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","#,
    r#""key_name":"sk-...abcd","key_alias":null,"soft_budget_cooldown":false,"spend":0.0,"#,
    r#""total_spend":0.0,"expires":null,"models":[],"aliases":{},"config":{},"router_settings":{},"#,
    r#""user_id":null,"team_id":null,"agent_id":null,"project_id":null,"permissions":{},"#,
    r#""max_parallel_requests":null,"metadata":{},"blocked":null,"tpm_limit":null,"rpm_limit":null,"#,
    r#""tpd_limit":null,"max_budget":null,"soft_budget":null,"budget_duration":null,"#,
    r#""budget_reset_at":null,"allowed_cache_controls":[],"allowed_routes":[],"key_type":null,"#,
    r#""access_group_ids":[],"model_spend":{},"model_max_budget":{},"budget_fallbacks":{},"#,
    r#""budget_id":null,"org_id":null,"object_permission_id":null,"#,
    r#""created_at":"2026-11-19T10:00:00.123+00:00","created_by":"default_user_id","#,
    r#""updated_at":"2026-11-19T10:00:00+00:00","updated_by":"default_user_id","#,
    r#""settings_updated_at":null,"last_active":null,"rotation_count":0,"auto_rotate":false,"#,
    r#""rotation_interval":null,"last_rotation_at":null,"key_rotation_at":null,"budget_limits":null,"#,
    r#""team_spend":null,"team_alias":null,"team_tpm_limit":null,"team_rpm_limit":null,"#,
    r#""team_tpd_limit":null,"team_max_budget":null,"team_soft_budget":null,"#,
    r#""team_model_max_budget":null,"team_models":[],"team_blocked":false,"team_model_aliases":null,"#,
    r#""team_member":null,"team_metadata":null,"team_object_permission_id":null,"#,
    r#""team_member_spend":null,"team_member_tpm_limit":null,"team_member_rpm_limit":null,"#,
    r#""organization_alias":null,"organization_max_budget":null,"organization_tpm_limit":null,"#,
    r#""organization_rpm_limit":null,"organization_metadata":null,"project_alias":null,"#,
    r#""last_refreshed_at":1763546400.5}"#,
);
const BARE_ENTRY: &str = concat!(
    r#"{"key_id": null, "token": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","#,
    r#" "key_name": "sk-...abcd", "key_alias": null, "spend": 0.0, "total_spend": 0.0,"#,
    r#" "max_budget": null, "expires": null, "models": [], "aliases": {}, "config": {},"#,
    r#" "user_id": null, "team_id": null, "agent_id": null, "project_id": null,"#,
    r#" "max_parallel_requests": null, "metadata": {}, "tpm_limit": null, "rpm_limit": null,"#,
    r#" "tpd_limit": null, "budget_duration": null, "budget_reset_at": null,"#,
    r#" "allowed_cache_controls": [], "allowed_routes": [], "key_type": null, "permissions": {},"#,
    r#" "model_spend": {}, "model_max_budget": {}, "budget_fallbacks": {},"#,
    r#" "soft_budget_cooldown": false, "blocked": null, "litellm_budget_table": null,"#,
    r#" "budget_id": null, "org_id": null, "created_at": "2026-11-19T10:00:00.123000Z","#,
    r#" "created_by": "default_user_id", "updated_at": "2026-11-19T10:00:00Z","#,
    r#" "updated_by": "default_user_id", "settings_updated_at": null, "last_active": null,"#,
    r#" "object_permission_id": null, "object_permission": null, "access_group_ids": [],"#,
    r#" "rotation_count": 0, "auto_rotate": false, "rotation_interval": null, "last_rotation_at": null,"#,
    r#" "key_rotation_at": null, "router_settings": {}, "budget_limits": null, "team_spend": null,"#,
    r#" "team_alias": null, "team_tpm_limit": null, "team_rpm_limit": null, "team_tpd_limit": null,"#,
    r#" "team_max_budget": null, "team_soft_budget": null, "team_model_max_budget": null,"#,
    r#" "team_models": [], "team_blocked": false, "soft_budget": null, "team_model_aliases": null,"#,
    r#" "team_member": null, "team_metadata": null, "team_object_permission_id": null,"#,
    r#" "team_member_spend": null, "team_member_tpm_limit": null, "team_member_rpm_limit": null,"#,
    r#" "end_user_id": null, "end_user_tpm_limit": null, "end_user_rpm_limit": null,"#,
    r#" "end_user_tpd_limit": null, "end_user_max_budget": null, "end_user_model_max_budget": null,"#,
    r#" "end_user_model_tpm_limit": null, "end_user_model_rpm_limit": null, "customer_hierarchy": null,"#,
    r#" "organization_alias": null, "organization_max_budget": null, "organization_tpm_limit": null,"#,
    r#" "organization_rpm_limit": null, "organization_metadata": null, "project_alias": null,"#,
    r#" "project_metadata": null, "last_refreshed_at": 1763546400.5, "api_key": null,"#,
    r#" "user_role": null, "allowed_model_region": null, "parent_otel_span": null,"#,
    r#" "rpm_limit_per_model": null, "tpm_limit_per_model": null, "user_tpm_limit": null,"#,
    r#" "user_rpm_limit": null, "user_email": null, "user_spend": null, "user_max_budget": null,"#,
    r#" "user_model_max_budget": null, "request_route": null, "is_session_token": false, "user": null,"#,
    r#" "created_by_user": null, "end_user_object_permission": null, "team_object_permission": null,"#,
    r#" "jwt_claims": null}"#,
);
const TEAM_ROW: &str = concat!(
    r#"{"token":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","#,
    r#""key_name":"sk-...wxyz","key_alias":"team key","soft_budget_cooldown":false,"#,
    r#""spend":0.30000000000000004,"total_spend":0.0,"expires":"2030-01-02T03:04:05.678+00:00","#,
    r#""models":["m1","gpt-4o"],"aliases":{},"config":{},"router_settings":null,"user_id":"kv-user","#,
    r#""team_id":"kv-team","agent_id":null,"project_id":null,"permissions":{},"#,
    r#""max_parallel_requests":3,"metadata":{"a": [1, 2.5], "note": "é"},"blocked":null,"tpm_limit":77,"#,
    r#""rpm_limit":5,"tpd_limit":null,"max_budget":9.5,"soft_budget":4.5,"budget_duration":null,"#,
    r#""budget_reset_at":"2030-01-02T00:00:00+00:00","allowed_cache_controls":null,"#,
    r#""allowed_routes":null,"key_type":null,"access_group_ids":[],"model_spend":{},"#,
    r#""model_max_budget":{"gpt-4o": 1},"budget_fallbacks":{},"budget_id":"kv-budget","org_id":null,"#,
    r#""object_permission_id":"kv-permission","created_at":"2026-11-19T10:00:00.123+00:00","#,
    r#""created_by":null,"updated_at":"2026-11-19T10:00:00+00:00","updated_by":null,"#,
    r#""settings_updated_at":null,"last_active":null,"rotation_count":null,"auto_rotate":null,"#,
    r#""rotation_interval":null,"last_rotation_at":null,"key_rotation_at":null,"budget_limits":null,"#,
    r#""team_spend":2.25,"team_alias":"Team KV","team_tpm_limit":null,"team_rpm_limit":null,"#,
    r#""team_tpd_limit":null,"team_max_budget":12.0,"team_soft_budget":null,"team_model_max_budget":{},"#,
    r#""team_models":["gpt-4o"],"team_blocked":true,"team_model_aliases":null,"#,
    r#""team_member":{"role": "admin", "joined": "2026", "user_id": "kv-user"},"#,
    r#""team_metadata":{"tier": "gold"},"team_object_permission_id":null,"team_member_spend":null,"#,
    r#""team_member_tpm_limit":null,"team_member_rpm_limit":null,"organization_alias":null,"#,
    r#""organization_max_budget":null,"organization_tpm_limit":null,"organization_rpm_limit":null,"#,
    r#""organization_metadata":null,"project_alias":null,"last_refreshed_at":1763546400.5}"#,
);
const TEAM_ENTRY: &str = concat!(
    r#"{"key_id": null, "token": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","#,
    r#" "key_name": "sk-...wxyz", "key_alias": "team key", "spend": 0.30000000000000004,"#,
    r#" "total_spend": 0.0, "max_budget": 9.5, "expires": "2030-01-02T03:04:05.678+00:00","#,
    r#" "models": ["m1", "gpt-4o"], "aliases": {}, "config": {}, "user_id": "kv-user","#,
    r#" "team_id": "kv-team", "agent_id": null, "project_id": null, "max_parallel_requests": 3,"#,
    r#" "metadata": {"a": [1, 2.5], "note": "\u00e9"}, "tpm_limit": 77, "rpm_limit": 5,"#,
    r#" "tpd_limit": null, "budget_duration": null, "budget_reset_at": "2030-01-02T00:00:00Z","#,
    r#" "allowed_cache_controls": [], "allowed_routes": [], "key_type": null, "permissions": {},"#,
    r#" "model_spend": {}, "model_max_budget": {"gpt-4o": 1}, "budget_fallbacks": {},"#,
    r#" "soft_budget_cooldown": false, "blocked": null, "litellm_budget_table": null,"#,
    r#" "budget_id": "kv-budget", "org_id": null, "created_at": "2026-11-19T10:00:00.123000Z","#,
    r#" "created_by": null, "updated_at": "2026-11-19T10:00:00Z", "updated_by": null,"#,
    r#" "settings_updated_at": null, "last_active": null, "object_permission_id": "kv-permission","#,
    r#" "object_permission": {"object_permission_id": "kv-permission", "mcp_servers": ["server-a"],"#,
    r#" "mcp_access_groups": [], "mcp_tool_permissions": {"server-a": ["search"]},"#,
    r#" "vector_stores": ["store-a"], "agents": [], "agent_access_groups": [], "models": [],"#,
    r#" "mcp_toolsets": [], "blocked_tools": [], "search_tools": [], "mcp_tool_search_enabled": null,"#,
    r#" "skills": [], "apps": null}, "access_group_ids": [], "rotation_count": 0, "auto_rotate": false,"#,
    r#" "rotation_interval": null, "last_rotation_at": null, "key_rotation_at": null,"#,
    r#" "router_settings": null, "budget_limits": null, "team_spend": 2.25, "team_alias": "Team KV","#,
    r#" "team_tpm_limit": null, "team_rpm_limit": null, "team_tpd_limit": null,"#,
    r#" "team_max_budget": 12.0, "team_soft_budget": null, "team_model_max_budget": {},"#,
    r#" "team_models": ["gpt-4o"], "team_blocked": true, "soft_budget": 4.5,"#,
    r#" "team_model_aliases": null, "team_member": {"user_id": "kv-user", "user_email": null,"#,
    r#" "role": "admin"}, "team_metadata": {"tier": "gold"}, "team_object_permission_id": null,"#,
    r#" "team_member_spend": null, "team_member_tpm_limit": null, "team_member_rpm_limit": null,"#,
    r#" "end_user_id": null, "end_user_tpm_limit": null, "end_user_rpm_limit": null,"#,
    r#" "end_user_tpd_limit": null, "end_user_max_budget": null, "end_user_model_max_budget": null,"#,
    r#" "end_user_model_tpm_limit": null, "end_user_model_rpm_limit": null, "customer_hierarchy": null,"#,
    r#" "organization_alias": null, "organization_max_budget": null, "organization_tpm_limit": null,"#,
    r#" "organization_rpm_limit": null, "organization_metadata": null, "project_alias": null,"#,
    r#" "project_metadata": null, "last_refreshed_at": 1763546400.5, "api_key": null,"#,
    r#" "user_role": null, "allowed_model_region": null, "parent_otel_span": null,"#,
    r#" "rpm_limit_per_model": null, "tpm_limit_per_model": null, "user_tpm_limit": null,"#,
    r#" "user_rpm_limit": null, "user_email": null, "user_spend": null, "user_max_budget": null,"#,
    r#" "user_model_max_budget": null, "request_route": null, "is_session_token": false, "user": null,"#,
    r#" "created_by_user": null, "end_user_object_permission": null, "team_object_permission": null,"#,
    r#" "jwt_claims": null}"#,
);
const PERMISSION_ROW: &str = concat!(
    r#"{"object_permission_id":"kv-permission","mcp_servers":["server-a"],"mcp_access_groups":[],"#,
    r#""mcp_tool_permissions":{"server-a": ["search"]},"vector_stores":["store-a"],"agents":[],"#,
    r#""agent_access_groups":[],"models":[],"blocked_tools":[],"mcp_toolsets":[],"search_tools":[],"#,
    r#""mcp_tool_search_enabled":null,"skills":[]}"#,
);
const PERMISSION_ENTRY: &str = concat!(
    r#"{"object_permission_id": "kv-permission", "mcp_servers": ["server-a"], "mcp_access_groups": [],"#,
    r#" "mcp_tool_permissions": {"server-a": ["search"]}, "vector_stores": ["store-a"], "agents": [],"#,
    r#" "agent_access_groups": [], "models": [], "mcp_toolsets": [], "blocked_tools": [],"#,
    r#" "search_tools": [], "mcp_tool_search_enabled": null, "skills": [], "apps": null}"#,
);
const FULL_ROW: &str = concat!(
    r#"{"token":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","#,
    r#""key_name":"sk-...full","key_alias":"full key","soft_budget_cooldown":true,"spend":1.25,"#,
    r#""total_spend":2.5,"expires":"2031-02-03T04:05:06+00:00","models":["k-model"],"#,
    r#""aliases":{"alias": "k-model"},"config":{"cfg": 1},"#,
    r#""router_settings":{"routing_strategy": "simple-shuffle"},"user_id":"full-user","#,
    r#""team_id":"full-team","agent_id":"full-agent","project_id":"full-project","#,
    r#""permissions":{"perm": true},"max_parallel_requests":7,"metadata":{"meta": [1]},"blocked":false,"#,
    r#""tpm_limit":101,"rpm_limit":102,"tpd_limit":103,"max_budget":104.5,"soft_budget":5.5,"#,
    r#""budget_duration":"30d","budget_reset_at":"2031-03-01T00:00:00.500+00:00","#,
    r#""allowed_cache_controls":["no-cache"],"allowed_routes":["/v1/messages"],"key_type":"llm_api","#,
    r#""access_group_ids":["group-a"],"model_spend":{"k-model": 0.5},"#,
    r#""model_max_budget":{"k-model": {"budget_limit": 2}},"#,
    r#""budget_fallbacks":{"k-model": ["fallback"]},"budget_id":"full-key-budget","org_id":"full-org","#,
    r#""object_permission_id":"full-key-permission","created_at":"2026-01-01T01:01:01.001+00:00","#,
    r#""created_by":"creator","updated_at":"2026-02-02T02:02:02.002+00:00","updated_by":"updater","#,
    r#""settings_updated_at":"2026-03-03T03:03:03+00:00","last_active":"2026-04-04T04:04:04.400+00:00","#,
    r#""rotation_count":3,"auto_rotate":true,"rotation_interval":"90d","#,
    r#""last_rotation_at":"2026-05-05T05:05:05+00:00","key_rotation_at":"2026-08-08T08:08:08+00:00","#,
    r#""budget_limits":[{"model": "k-model", "budget": 1}],"team_spend":3.5,"team_alias":"Full Team","#,
    r#""team_tpm_limit":61,"team_rpm_limit":62,"team_tpd_limit":63,"team_max_budget":51.5,"#,
    r#""team_soft_budget":52.5,"team_model_max_budget":{"t-model": {"budget_limit": 3}},"#,
    r#""team_models":["t-model"],"team_blocked":false,"team_model_aliases":{"fast": "t-model"},"#,
    r#""team_member":{"role": "user", "user_id": "full-user", "user_email": "user@example.com"},"#,
    r#""team_metadata":{"team": true},"team_object_permission_id":"full-team-permission","#,
    r#""team_member_spend":4.5,"team_member_tpm_limit":21,"team_member_rpm_limit":22,"#,
    r#""organization_alias":"Full Org","organization_max_budget":43.5,"organization_tpm_limit":31,"#,
    r#""organization_rpm_limit":32,"organization_metadata":{"org": true},"#,
    r#""project_alias":"Full Project","last_refreshed_at":1763546400.5}"#,
);
const FULL_ENTRY: &str = concat!(
    r#"{"key_id": null, "token": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","#,
    r#" "key_name": "sk-...full", "key_alias": "full key", "spend": 1.25, "total_spend": 2.5,"#,
    r#" "max_budget": 104.5, "expires": "2031-02-03T04:05:06+00:00", "models": ["k-model"],"#,
    r#" "aliases": {"alias": "k-model"}, "config": {"cfg": 1}, "user_id": "full-user","#,
    r#" "team_id": "full-team", "agent_id": "full-agent", "project_id": "full-project","#,
    r#" "max_parallel_requests": 7, "metadata": {"meta": [1]}, "tpm_limit": 101, "rpm_limit": 102,"#,
    r#" "tpd_limit": 103, "budget_duration": "30d", "budget_reset_at": "2031-03-01T00:00:00.500000Z","#,
    r#" "allowed_cache_controls": ["no-cache"], "allowed_routes": ["/v1/messages"],"#,
    r#" "key_type": "llm_api", "permissions": {"perm": true}, "model_spend": {"k-model": 0.5},"#,
    r#" "model_max_budget": {"k-model": {"budget_limit": 2}},"#,
    r#" "budget_fallbacks": {"k-model": ["fallback"]}, "soft_budget_cooldown": true, "blocked": false,"#,
    r#" "litellm_budget_table": null, "budget_id": "full-key-budget", "org_id": "full-org","#,
    r#" "created_at": "2026-01-01T01:01:01.001000Z", "created_by": "creator","#,
    r#" "updated_at": "2026-02-02T02:02:02.002000Z", "updated_by": "updater","#,
    r#" "settings_updated_at": "2026-03-03T03:03:03Z", "last_active": "2026-04-04T04:04:04.400000Z","#,
    r#" "object_permission_id": "full-key-permission","#,
    r#" "object_permission": {"object_permission_id": "full-key-permission", "mcp_servers": ["server"],"#,
    r#" "mcp_access_groups": ["mcp-group"], "mcp_tool_permissions": {"server": ["tool"]},"#,
    r#" "vector_stores": ["store"], "agents": ["agent"], "agent_access_groups": ["agent-group"],"#,
    r#" "models": ["p-model"], "mcp_toolsets": ["toolset"], "blocked_tools": ["blocked-tool"],"#,
    r#" "search_tools": ["search-tool"], "mcp_tool_search_enabled": true, "skills": ["skill"],"#,
    r#" "apps": null}, "access_group_ids": ["group-a"], "rotation_count": 3, "auto_rotate": true,"#,
    r#" "rotation_interval": "90d", "last_rotation_at": "2026-05-05T05:05:05Z","#,
    r#" "key_rotation_at": "2026-08-08T08:08:08Z","#,
    r#" "router_settings": {"routing_strategy": "simple-shuffle"},"#,
    r#" "budget_limits": [{"model": "k-model", "budget": 1}], "team_spend": 3.5,"#,
    r#" "team_alias": "Full Team", "team_tpm_limit": 61, "team_rpm_limit": 62, "team_tpd_limit": 63,"#,
    r#" "team_max_budget": 51.5, "team_soft_budget": 52.5,"#,
    r#" "team_model_max_budget": {"t-model": {"budget_limit": 3}}, "team_models": ["t-model"],"#,
    r#" "team_blocked": false, "soft_budget": 5.5, "team_model_aliases": {"fast": "t-model"},"#,
    r#" "team_member": {"user_id": "full-user", "user_email": "user@example.com", "role": "user"},"#,
    r#" "team_metadata": {"team": true}, "team_object_permission_id": "full-team-permission","#,
    r#" "team_member_spend": 4.5, "team_member_tpm_limit": 21, "team_member_rpm_limit": 22,"#,
    r#" "end_user_id": null, "end_user_tpm_limit": null, "end_user_rpm_limit": null,"#,
    r#" "end_user_tpd_limit": null, "end_user_max_budget": null, "end_user_model_max_budget": null,"#,
    r#" "end_user_model_tpm_limit": null, "end_user_model_rpm_limit": null, "customer_hierarchy": null,"#,
    r#" "organization_alias": "Full Org", "organization_max_budget": 43.5,"#,
    r#" "organization_tpm_limit": 31, "organization_rpm_limit": 32,"#,
    r#" "organization_metadata": {"org": true}, "project_alias": "Full Project","#,
    r#" "project_metadata": null, "last_refreshed_at": 1763546400.5, "api_key": null,"#,
    r#" "user_role": null, "allowed_model_region": null, "parent_otel_span": null,"#,
    r#" "rpm_limit_per_model": null, "tpm_limit_per_model": null, "user_tpm_limit": null,"#,
    r#" "user_rpm_limit": null, "user_email": null, "user_spend": null, "user_max_budget": null,"#,
    r#" "user_model_max_budget": null, "request_route": null, "is_session_token": false, "user": null,"#,
    r#" "created_by_user": null, "end_user_object_permission": null, "team_object_permission": null,"#,
    r#" "jwt_claims": null}"#,
);
const FULL_PERMISSION_ROW: &str = concat!(
    r#"{"object_permission_id":"full-key-permission","mcp_servers":["server"],"#,
    r#""mcp_access_groups":["mcp-group"],"mcp_tool_permissions":{"server": ["tool"]},"#,
    r#""vector_stores":["store"],"agents":["agent"],"agent_access_groups":["agent-group"],"#,
    r#""models":["p-model"],"blocked_tools":["blocked-tool"],"mcp_toolsets":["toolset"],"#,
    r#""search_tools":["search-tool"],"mcp_tool_search_enabled":true,"skills":["skill"]}"#,
);
const FULL_PERMISSION_ENTRY: &str = concat!(
    r#"{"object_permission_id": "full-key-permission", "mcp_servers": ["server"],"#,
    r#" "mcp_access_groups": ["mcp-group"], "mcp_tool_permissions": {"server": ["tool"]},"#,
    r#" "vector_stores": ["store"], "agents": ["agent"], "agent_access_groups": ["agent-group"],"#,
    r#" "models": ["p-model"], "mcp_toolsets": ["toolset"], "blocked_tools": ["blocked-tool"],"#,
    r#" "search_tools": ["search-tool"], "mcp_tool_search_enabled": true, "skills": ["skill"],"#,
    r#" "apps": null}"#,
);

struct NoEvents;

impl Observer for NoEvents {
    fn record(&self, _event: QueryEvent) {}
}

fn database(postgres: &MigratedPostgres) -> Database {
    let options = PgConnectOptions::from_str(postgres.url()).unwrap();
    let endpoint = Endpoint {
        host: options.get_host().to_owned(),
        port: options.get_port(),
        socket: None,
        username: "postgres".to_owned(),
        password: Some(SecretValue::new("postgres")),
        database: "postgres".to_owned(),
        ssl_mode: SslMode::Disable,
        ssl_root_cert: None,
        ssl_client: None,
        application_name: None,
        options: Vec::new(),
        statement_cache_capacity: 100,
        pool: PoolLimits {
            max_connections: 2,
            acquire_timeout_seconds: Some(10.0),
            idle_timeout_seconds: None,
            socket_timeout_seconds: None,
        },
    };
    Database::new(
        Pools::new(&endpoint, None).unwrap(),
        Arc::new(NoEvents),
        None,
        endpoint.names_statements(),
    )
}

fn key_database(postgres: &MigratedPostgres) -> KeyDatabase {
    KeyDatabase::new(
        Arc::new(KeyRepository::new(database(postgres))),
        Arc::new(ObjectPermissionRepository::new(database(postgres))),
    )
}

async fn execute(pool: &PgPool, sql: &'static str) {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture statements are outside the compile-checked query set"
    )]
    let statement = sqlx::query(sql);
    statement.execute(pool).await.unwrap();
}

/// A key as `/key/generate` stores it when asked for nothing.
async fn insert_bare_key(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_VerificationToken" (token, key_name, created_at, created_by, updated_at, updated_by)
           VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'sk-...abcd',
                   '2026-11-19 10:00:00.123', 'default_user_id', '2026-11-19 10:00:00', 'default_user_id')"#,
    )
    .await;
}

/// A key in a team, with a member entry that holds more than `Member` keeps, a budget row, an
/// object permission, and nulls in the columns whose model default is not null.
async fn insert_team_key(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, tpm_limit, soft_budget, model_max_budget, created_by, updated_by)
           VALUES ('kv-budget', 9.5, 77, 4.5, '{"gpt-4o": 1}', 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, models, blocked, max_budget, spend, members_with_roles, metadata)
           VALUES ('kv-team', 'Team KV', ARRAY['gpt-4o'], true, 12.0, 2.25,
                   '[{"user_id": "kv-user", "role": "admin", "joined": "2026"}]', '{"tier": "gold"}')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id, mcp_servers, vector_stores, mcp_tool_permissions)
           VALUES ('kv-permission', ARRAY['server-a'], ARRAY['store-a'], '{"server-a": ["search"]}')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_VerificationToken"
               (token, key_name, key_alias, spend, expires, models, metadata, user_id, team_id, budget_id,
                rpm_limit, max_parallel_requests, object_permission_id, budget_reset_at,
                allowed_cache_controls, allowed_routes, rotation_count, auto_rotate, router_settings,
                created_at, updated_at)
           VALUES ('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'sk-...wxyz', 'team key', 0.30000000000000004,
                   '2030-01-02 03:04:05.678', ARRAY['m1', 'gpt-4o'], '{"a": [1, 2.5], "note": "é"}', 'kv-user', 'kv-team', 'kv-budget',
                   5, 3, 'kv-permission', '2030-01-02 00:00:00',
                   NULL, NULL, NULL, NULL, NULL,
                   '2026-11-19 10:00:00.123', '2026-11-19 10:00:00')"#,
    )
    .await;
}

/// A key with every column of its row set, each to a value no other column holds, in a team, an
/// organization and a project that set theirs too.
async fn insert_full_key(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, soft_budget, tpm_limit, rpm_limit, tpd_limit, model_max_budget, created_by, updated_by)
           VALUES ('full-key-budget', 41.5, 5.5, 11, 12, 13, '{"b-model": {"budget_limit": 1}}', 'test', 'test'),
                  ('full-member-budget', 42.5, NULL, 21, 22, NULL, '{}', 'test', 'test'),
                  ('full-org-budget', 43.5, NULL, 31, 32, NULL, '{}', 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, metadata, created_by, updated_by)
           VALUES ('full-org', 'Full Org', 'full-org-budget', '{"org": true}', 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ModelTable" (id, aliases, created_by, updated_by)
           VALUES (9001, '{"fast": "t-model"}', 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ObjectPermissionTable"
               (object_permission_id, mcp_servers, mcp_access_groups, mcp_tool_permissions, vector_stores, agents,
                agent_access_groups, models, blocked_tools, mcp_toolsets, search_tools, mcp_tool_search_enabled, skills)
           VALUES ('full-team-permission', ARRAY['team-server'], NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
                  ('full-key-permission', ARRAY['server'], ARRAY['mcp-group'], '{"server": ["tool"]}', ARRAY['store'], ARRAY['agent'],
                   ARRAY['agent-group'], ARRAY['p-model'], ARRAY['blocked-tool'], ARRAY['toolset'], ARRAY['search-tool'], true, ARRAY['skill'])"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamTable"
               (team_id, team_alias, organization_id, object_permission_id, models, blocked, spend, max_budget, soft_budget,
                tpm_limit, rpm_limit, tpd_limit, model_max_budget, members_with_roles, metadata, model_id)
           VALUES ('full-team', 'Full Team', 'full-org', 'full-team-permission', ARRAY['t-model'], false, 3.5, 51.5, 52.5,
                   61, 62, 63, '{"t-model": {"budget_limit": 3}}',
                   '[{"user_id": "full-user", "user_email": "user@example.com", "role": "user"}]', '{"team": true}', 9001)"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, budget_id)
           VALUES ('full-user', 'full-team', 4.5, 'full-member-budget')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ProjectTable" (project_id, project_alias, team_id, models, created_by, updated_by)
           VALUES ('full-project', 'Full Project', 'full-team', ARRAY[]::text[], 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_VerificationToken"
               (token, key_name, key_alias, soft_budget_cooldown, spend, total_spend, expires, models, aliases, config,
                router_settings, user_id, team_id, agent_id, project_id, permissions, max_parallel_requests, metadata,
                blocked, tpm_limit, rpm_limit, tpd_limit, max_budget, budget_duration, budget_reset_at,
                allowed_cache_controls, allowed_routes, key_type, access_group_ids, model_spend, model_max_budget,
                budget_fallbacks, budget_id, organization_id, object_permission_id, created_at, created_by, updated_at,
                updated_by, settings_updated_at, last_active, rotation_count, auto_rotate, rotation_interval,
                last_rotation_at, key_rotation_at, budget_limits)
           VALUES ('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', 'sk-...full', 'full key', true, 1.25, 2.5,
                   '2031-02-03 04:05:06', ARRAY['k-model'], '{"alias": "k-model"}', '{"cfg": 1}',
                   '{"routing_strategy": "simple-shuffle"}', 'full-user', 'full-team', 'full-agent', 'full-project',
                   '{"perm": true}', 7, '{"meta": [1]}',
                   false, 101, 102, 103, 104.5, '30d', '2031-03-01 00:00:00.5',
                   ARRAY['no-cache'], ARRAY['/v1/messages'], 'llm_api', ARRAY['group-a'], '{"k-model": 0.5}',
                   '{"k-model": {"budget_limit": 2}}',
                   '{"k-model": ["fallback"]}', 'full-key-budget', 'full-org', 'full-key-permission',
                   '2026-01-01 01:01:01.001', 'creator', '2026-02-02 02:02:02.002',
                   'updater', '2026-03-03 03:03:03', '2026-04-04 04:04:04.4', 3, true, '90d',
                   '2026-05-05 05:05:05', '2026-08-08 08:08:08', '[{"model": "k-model", "budget": 1}]')"#,
    )
    .await;
}

fn hash(token: &str) -> KeyHash {
    KeyHash::new(token)
}

/// `json` with the time it was read or cached replaced by [`STAMP`], the one field no two runs
/// agree on.
fn stamped(json: &str) -> String {
    const FIELD: &str = "\"last_refreshed_at\":";
    let Some(start) = json.find(FIELD) else {
        return json.to_owned();
    };
    let value = start + FIELD.len() + json[start + FIELD.len()..].starts_with(' ') as usize;
    let end = json[value..]
        .find([',', '}'])
        .map_or(json.len(), |end| value + end);
    format!("{}{STAMP}{}", &json[..value], &json[end..])
}

fn dumped(value: impl IntoValue) -> String {
    litellm_python_compat::json::dumps(&value.into_value()).unwrap()
}

#[rstest]
#[case::as_generated(BARE, BARE_ROW, BARE_ENTRY)]
#[case::in_a_team(TEAM_KEY, TEAM_ROW, TEAM_ENTRY)]
#[case::every_column_set(FULL, FULL_ROW, FULL_ENTRY)]
#[tokio::test(flavor = "multi_thread")]
async fn a_key_read_from_the_database_is_the_entry_python_caches(
    #[case] token: &str,
    #[case] row: &str,
    #[case] cached: &str,
) {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_bare_key(postgres.pool()).await;
    insert_team_key(postgres.pool()).await;
    insert_full_key(postgres.pool()).await;
    let keys = KeyRepository::new(database(&postgres));

    let found = keys.find(token, Side::Read).await.unwrap().unwrap();
    let read = key_database(&postgres)
        .read(&hash(token), Side::Read)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(
        (
            stamped(&serde_json::to_string(&found).unwrap()).as_str(),
            stamped(&dumped(read)).as_str(),
        ),
        (row, cached)
    );
}

#[rstest]
#[case::its_defaults(TEAM_KEY, "kv-permission", PERMISSION_ROW, PERMISSION_ENTRY)]
#[case::every_column_set(
    FULL,
    "full-key-permission",
    FULL_PERMISSION_ROW,
    FULL_PERMISSION_ENTRY
)]
#[tokio::test(flavor = "multi_thread")]
async fn a_key_s_object_permission_is_nested_as_python_dumps_it(
    #[case] token: &str,
    #[case] id: &str,
    #[case] row: &str,
    #[case] nested: &str,
) {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_team_key(postgres.pool()).await;
    insert_full_key(postgres.pool()).await;
    let permissions = ObjectPermissionRepository::new(database(&postgres));

    let found = permissions
        .find_by_object_permission_id(id, Side::Read)
        .await
        .unwrap()
        .unwrap();
    let read = key_database(&postgres)
        .read(&hash(token), Side::Read)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(
        (
            serde_json::to_string(&found).unwrap().as_str(),
            read.object_permission.map(dumped).as_deref(),
        ),
        (row, Some(nested))
    );
}

/// Holds every read of the object permission table until it is committed. A key read has its
/// row by then and is waiting for the permission the row points at.
async fn stall_permission_reads(pool: &PgPool) -> sqlx::Transaction<'static, sqlx::Postgres> {
    let mut stalled = pool.begin().await.unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture statements are outside the compile-checked query set"
    )]
    let lock =
        sqlx::query(r#"LOCK TABLE "LiteLLM_ObjectPermissionTable" IN ACCESS EXCLUSIVE MODE"#);
    lock.execute(&mut *stalled).await.unwrap();
    stalled
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn the_key_row_alone_is_read_while_the_object_permission_table_does_not_answer() {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_team_key(postgres.pool()).await;
    let _stalled = stall_permission_reads(postgres.pool()).await;

    let key = key_database(&postgres)
        .key(&hash(TEAM_KEY), Side::Read)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(
        (
            key.object_permission_id.as_deref(),
            key.object_permission.is_none()
        ),
        (Some("kv-permission"), true)
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_rotated_hash_in_its_grace_period_is_the_active_key() {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_bare_key(postgres.pool()).await;
    execute(
        postgres.pool(),
        r#"INSERT INTO "LiteLLM_DeprecatedVerificationToken" (id, token, active_token_id, revoke_at)
           VALUES ('rotation', 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                   'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', NOW() AT TIME ZONE 'UTC' + INTERVAL '1 hour')"#,
    )
    .await;

    let read = key_database(&postgres)
        .read(&hash(ROTATED), Side::Read)
        .await
        .unwrap();

    assert_eq!(read.and_then(|key| key.token).as_deref(), Some(BARE));
}

/// A key whose object permission holds its tool map as JSON text, which the model refuses.
async fn insert_key_with_an_unreadable_permission(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id, mcp_servers, mcp_tool_permissions)
           VALUES ('legacy-permission', ARRAY['server-a'], '"{\"server-a\": [\"search\"]}"'::jsonb)"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_VerificationToken" (token, object_permission_id)
           VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'legacy-permission')"#,
    )
    .await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_key_whose_object_permission_the_model_refuses_is_read_without_it() {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_key_with_an_unreadable_permission(postgres.pool()).await;

    let read = key_database(&postgres)
        .read(&hash(BARE), Side::Read)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(
        (read.object_permission_id.as_deref(), read.object_permission),
        (Some("legacy-permission"), None)
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_writer_read_fails_on_an_object_permission_the_model_refuses() {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_key_with_an_unreadable_permission(postgres.pool()).await;

    let read = key_database(&postgres)
        .read(&hash(BARE), Side::Writer)
        .await;

    assert!(
        matches!(
            read,
            Err(litellm_auth_store::Error::Row {
                column: "mcp_tool_permissions"
            })
        ),
        "{:?}",
        read.err()
    );
}

#[rstest]
#[case::metadata_that_is_a_list(
    r#"UPDATE "LiteLLM_VerificationToken" SET metadata = '[]'"#,
    "metadata"
)]
#[case::fallbacks_that_are_not_lists(
    r#"UPDATE "LiteLLM_VerificationToken" SET budget_fallbacks = '{"m": "n"}'"#,
    "budget_fallbacks"
)]
#[case::budget_limits_that_are_no_list(
    r#"UPDATE "LiteLLM_VerificationToken" SET budget_limits = '{"m": 1}'"#,
    "budget_limits"
)]
#[tokio::test(flavor = "multi_thread")]
async fn a_key_row_the_model_refuses_fails_the_read(
    #[case] corrupt: &'static str,
    #[case] column: &str,
) {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_bare_key(postgres.pool()).await;
    execute(postgres.pool(), corrupt).await;

    let read = key_database(&postgres).read(&hash(BARE), Side::Read).await;

    assert!(
        matches!(&read, Err(litellm_auth_store::Error::Row { column: refused }) if *refused == column),
        "{:?}",
        read.err()
    );
}
