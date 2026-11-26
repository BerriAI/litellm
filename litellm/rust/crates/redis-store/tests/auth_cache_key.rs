//! The auth cache's typed key and object permission entries on real Redis. The entry literals
//! are the ones `tests/unit/redis/stores/test_auth_cache_key.py` checks against Python's own
//! models and key writer, and `auth-store/tests/key.rs` finds after a database read.
//! Change all three together.

mod support;

use std::time::Duration;

use jiff::Timestamp;
use litellm_python_compat::datetime::IsoDateTime;
use litellm_redis_client::commands;
use litellm_redis_store::{
    FromValue, IntoValue,
    auth_cache::{
        AuthCache, AuthObject, CachedKey, CachedMember, CachedObjectPermission, Lists, MemberRole,
        Settings,
    },
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, remote};

const TTL: Duration = Duration::from_secs(60);
const TOKEN: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const REWRITTEN: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

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
const PERMISSION_ENTRY: &str = concat!(
    r#"{"object_permission_id": "kv-permission", "mcp_servers": ["server-a"], "mcp_access_groups": [],"#,
    r#" "mcp_tool_permissions": {"server-a": ["search"]}, "vector_stores": ["store-a"], "agents": [],"#,
    r#" "agent_access_groups": [], "models": [], "mcp_toolsets": [], "blocked_tools": [],"#,
    r#" "search_tools": [], "mcp_tool_search_enabled": null, "skills": [], "apps": null}"#,
);

fn utc(text: &str) -> Option<IsoDateTime> {
    Some(IsoDateTime::utc(
        text.parse::<Timestamp>().unwrap().as_microsecond(),
    ))
}

#[rstest]
#[tokio::test]
async fn a_key_entry_python_wrote_is_read_typed_and_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let cache = AuthCache::new(Some(remote(connect(&deployment))), Settings::default());
    raw.send(commands::set(TOKEN, TEAM_ENTRY, TTL))
        .await
        .unwrap();

    let held = cache
        .get(None, &AuthObject::Key(TOKEN.to_owned()))
        .await
        .unwrap()
        .unwrap();
    let key = CachedKey::from_value(&held).unwrap();
    cache
        .set(
            None,
            &AuthObject::Key(REWRITTEN.to_owned()),
            key.clone().into_value(),
            Some(TTL),
        )
        .await
        .unwrap();

    assert_eq!(
        (
            key.token.as_deref(),
            key.spend,
            key.expires.as_deref(),
            key.budget_reset_at,
            key.created_at,
            key.rpm_limit,
            key.allowed_routes,
            key.team_member,
            key.object_permission
                .and_then(|permission| permission.mcp_servers),
            key.last_refreshed_at,
            key.unread,
            raw.send(commands::get::<String>(REWRITTEN))
                .await
                .unwrap()
                .as_deref(),
        ),
        (
            Some(TOKEN),
            0.30000000000000004,
            Some("2030-01-02T03:04:05.678+00:00"),
            utc("2030-01-02T00:00:00Z"),
            utc("2026-11-19T10:00:00.123Z"),
            Some(5),
            Some(Vec::new()),
            Some(CachedMember {
                user_id: Some("kv-user".to_owned()),
                user_email: None,
                role: MemberRole::Admin,
                unread: Vec::new(),
            }),
            Some(vec!["server-a".to_owned()]),
            Some(1_763_546_400.5),
            Vec::new(),
            Some(TEAM_ENTRY),
        )
    );
}

#[rstest]
#[tokio::test]
async fn an_object_permission_entry_python_wrote_is_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let cache = AuthCache::new(Some(remote(connect(&deployment))), Settings::default());
    let read = AuthObject::ObjectPermission("kv-permission".to_owned());
    let rewritten = AuthObject::ObjectPermission("rewritten".to_owned());
    raw.send(commands::set(&read.key(), PERMISSION_ENTRY, TTL))
        .await
        .unwrap();

    let held = cache.get(None, &read).await.unwrap().unwrap();
    let permission = CachedObjectPermission::from_value(&held).unwrap();
    cache
        .set(None, &rewritten, permission.clone().into_value(), Some(TTL))
        .await
        .unwrap();

    assert_eq!(
        (
            permission.vector_stores,
            permission.mcp_tool_permissions,
            permission.apps,
            raw.send(commands::get::<String>(&rewritten.key()))
                .await
                .unwrap()
                .as_deref(),
        ),
        (
            Some(vec!["store-a".to_owned()]),
            Some(Lists(vec![(
                "server-a".to_owned(),
                vec!["search".to_owned()]
            )])),
            None,
            Some(PERMISSION_ENTRY),
        )
    );
}
