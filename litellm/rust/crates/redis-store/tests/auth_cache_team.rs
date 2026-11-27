//! The auth cache's typed team entry on real Redis. The entry literals are the ones
//! `tests/unit/redis/stores/test_auth_cache_team.py` checks against Python's own model and team
//! writer and reader, and `auth-store/tests/team.rs` finds after a database read.
//! Change all three together.

mod support;

use litellm_redis_store::auth_cache::{AuthObject, TeamEntry};
use litellm_redis_testing::Topology;
use rstest::rstest;
use support::auth_cache::round_trip;

const CREATED_ENTRY: &str = concat!(
    r#"{"team_alias": null, "team_id": "created", "organization_id": null, "admins": [],"#,
    r#" "members": [], "members_with_roles": [], "team_member_permissions": [],"#,
    r#" "metadata": {}, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,"#,
    r#" "max_budget": null, "soft_budget": null, "budget_duration": null,"#,
    r#" "budget_limits": null, "models": [], "blocked": false, "router_settings": {},"#,
    r#" "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,"#,
    r#" "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,"#,
    r#" "model_spend": {}, "model_max_budget": {}, "policies": [],"#,
    r#" "allow_team_guardrail_config": false, "litellm_model_table": null,"#,
    r#" "object_permission": null, "object_permission_id": null,"#,
    r#" "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z","#,
    r#" "last_refreshed_at": null}"#,
);
const STAMPED_ENTRY: &str = concat!(
    r#"{"team_alias": null, "team_id": "stamped", "organization_id": null, "admins": [],"#,
    r#" "members": [], "members_with_roles": [], "team_member_permissions": [],"#,
    r#" "metadata": {}, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,"#,
    r#" "max_budget": null, "soft_budget": null, "budget_duration": null,"#,
    r#" "budget_limits": null, "models": [], "blocked": false, "router_settings": {},"#,
    r#" "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,"#,
    r#" "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,"#,
    r#" "model_spend": {}, "model_max_budget": {}, "policies": [],"#,
    r#" "allow_team_guardrail_config": false, "litellm_model_table": null,"#,
    r#" "object_permission": null, "object_permission_id": null,"#,
    r#" "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z","#,
    r#" "last_refreshed_at": 1764547200.5}"#,
);
const NULLS_ENTRY: &str = concat!(
    r#"{"team_alias": null, "team_id": "nulls", "organization_id": null, "admins": [],"#,
    r#" "members": [], "members_with_roles": [], "team_member_permissions": [],"#,
    r#" "metadata": null, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,"#,
    r#" "max_budget": null, "soft_budget": null, "budget_duration": null,"#,
    r#" "budget_limits": null, "models": [], "blocked": false, "router_settings": {},"#,
    r#" "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,"#,
    r#" "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,"#,
    r#" "model_spend": null, "model_max_budget": null, "policies": [],"#,
    r#" "allow_team_guardrail_config": false, "litellm_model_table": null,"#,
    r#" "object_permission": null, "object_permission_id": null,"#,
    r#" "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00Z","#,
    r#" "last_refreshed_at": null}"#,
);
const TEXT_ENTRY: &str = concat!(
    r#"{"team_alias": null, "team_id": "text", "organization_id": null, "admins": [],"#,
    r#" "members": [], "members_with_roles": [], "team_member_permissions": [],"#,
    r#" "metadata": {"guardrails": ["pii"]}, "tpm_limit": null, "rpm_limit": null,"#,
    r#" "tpd_limit": null, "max_budget": null, "soft_budget": null, "budget_duration": null,"#,
    r#" "budget_limits": [{"budget_duration": "1d", "max_budget": 1.0, "reset_at": null}],"#,
    r#" "models": [], "blocked": false, "router_settings": {"num_retries": 2},"#,
    r#" "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,"#,
    r#" "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,"#,
    r#" "model_spend": {}, "model_max_budget": {"claude": {"budget_limit": 2.0}},"#,
    r#" "policies": [], "allow_team_guardrail_config": false, "litellm_model_table": null,"#,
    r#" "object_permission": null, "object_permission_id": null,"#,
    r#" "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00Z","#,
    r#" "last_refreshed_at": null}"#,
);
const FULL_ENTRY: &str = concat!(
    r#"{"team_alias": "Full", "team_id": "full", "organization_id": "org","#,
    r#" "admins": ["admin"], "members": ["member"],"#,
    r#" "members_with_roles": [{"user_id": "admin", "user_email": null, "role": "admin"},"#,
    r#" {"user_id": null, "user_email": "m@example.com", "role": "user"}],"#,
    r#" "team_member_permissions": ["/key/generate"], "metadata": {"guardrails": ["pii"]},"#,
    r#" "tpm_limit": 1000, "rpm_limit": 10, "tpd_limit": 100000, "max_budget": 10.5,"#,
    r#" "soft_budget": 5.25, "budget_duration": "30d","#,
    r#" "budget_limits": [{"budget_duration": "1d", "max_budget": 1.0,"#,
    r#" "reset_at": "2026-12-01T00:00:00Z", "window_start": "2026-11-01T00:00:00Z"},"#,
    r#" {"budget_duration": "30d", "max_budget": 2.5, "reset_at": null}],"#,
    r#" "models": ["claude", "all-proxy-models"],"#,
    r#" "blocked": true, "router_settings": {"num_retries": 2},"#,
    r#" "access_group_ids": ["group"], "default_team_member_models": ["claude"],"#,
    r#" "spend": 1.5, "max_parallel_requests": 3, "budget_reset_at": "2026-12-01T00:00:00Z","#,
    r#" "model_id": 7, "model_spend": {"claude": 1.5},"#,
    r#" "model_max_budget": {"claude": {"budget_limit": 2.0}}, "policies": ["policy"],"#,
    r#" "allow_team_guardrail_config": true, "litellm_model_table": {"id": 7,"#,
    r#" "model_aliases": {"gpt": "claude"}, "created_by": "test", "updated_by": "test","#,
    r#" "team": null}, "object_permission": {"object_permission_id": "permission","#,
    r#" "mcp_servers": [], "mcp_access_groups": [], "mcp_tool_permissions": null,"#,
    r#" "vector_stores": [], "agents": [], "agent_access_groups": [], "models": [],"#,
    r#" "mcp_toolsets": [], "blocked_tools": [], "search_tools": [],"#,
    r#" "mcp_tool_search_enabled": null, "skills": [], "apps": null},"#,
    r#" "object_permission_id": "permission","#,
    r#" "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z","#,
    r#" "last_refreshed_at": null}"#,
);

const DEFAULTS_ENTRY: &str = concat!(
    r#"{"team_alias": null, "team_id": "defaults", "organization_id": "org", "admins": [], "mem"#,
    r#"bers": [], "members_with_roles": [], "team_member_permissions": ["/key/generate"], "meta"#,
    r#"data": {}, "tpm_limit": 100, "rpm_limit": 10, "tpd_limit": null, "max_budget": 10.0, "so"#,
    r#"ft_budget": null, "budget_duration": "30d", "budget_limits": null, "models": [], "blocke"#,
    r#"d": false, "router_settings": {}, "access_group_ids": ["wanted"], "default_team_member_m"#,
    r#"odels": [], "spend": 0.0, "max_parallel_requests": null, "budget_reset_at": "2026-12-01T"#,
    r#"00:00:00Z", "model_id": null, "model_spend": {}, "model_max_budget": {}, "policies": [],"#,
    r#" "allow_team_guardrail_config": false, "litellm_model_table": null, "object_permission":"#,
    r#" null, "object_permission_id": null, "updated_at": "2026-11-27T08:00:00Z", "created_at":"#,
    r#" "2026-11-27T08:00:00Z", "last_refreshed_at": null}"#,
);

#[rstest]
#[case::as_created("created", CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ENTRY)]
#[case::json_nulls("nulls", NULLS_ENTRY)]
#[case::json_text("text", TEXT_ENTRY)]
#[case::stamped_by_the_host_that_cached_it("stamped", STAMPED_ENTRY)]
#[case::upserted_with_defaults("defaults", DEFAULTS_ENTRY)]
#[tokio::test]
async fn a_team_entry_python_wrote_is_read_typed_and_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] team_id: &str,
    #[case] entry: &str,
) {
    let (team, written) = round_trip::<TeamEntry>(
        topology,
        AuthObject::Team(team_id.to_owned()),
        AuthObject::Team("rewritten".to_owned()),
        entry,
    )
    .await;

    assert_eq!(
        (team.team_id.as_str(), team.unread, written.as_deref()),
        (team_id, Vec::new(), Some(entry))
    );
}
