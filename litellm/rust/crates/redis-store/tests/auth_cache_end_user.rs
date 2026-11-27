//! The auth cache's typed end user entry on real Redis. The entry literals are the ones
//! `tests/unit/redis/stores/test_auth_cache_end_user.py` checks against Python's own model and end
//! user reader, and `auth-store/tests/end_user.rs` finds after a database read, less the one with
//! the default end-user budget merged in, which only the host that caches the end user writes.
//! Change all three together.

mod support;

use litellm_redis_store::auth_cache::{AuthObject, EndUserEntry};
use litellm_redis_testing::Topology;
use rstest::rstest;
use support::auth_cache::round_trip;

const CREATED_ENTRY: &str = concat!(
    r#"{"user_id": "created", "blocked": false, "alias": null, "spend": 0.0,"#,
    r#" "allowed_model_region": null, "default_model": null, "models": [], "budget_id": null,"#,
    r#" "litellm_budget_table": null, "object_permission_id": null, "object_permission": null}"#,
);
const FULL_ENTRY: &str = concat!(
    r#"{"user_id": "full", "blocked": true, "alias": "Full", "spend": 1.5,"#,
    r#" "allowed_model_region": "eu", "default_model": "claude", "models": ["claude", "gpt"],"#,
    r#" "budget_id": "budget", "litellm_budget_table": {"budget_id": "budget","#,
    r#" "soft_budget": 8.0, "max_budget": 10.5, "rollover_max_budget": 20.0,"#,
    r#" "max_parallel_requests": 3, "tpm_limit": 1000, "rpm_limit": 10, "tpd_limit": 100000,"#,
    r#" "model_max_budget": {"claude": {"max_budget": 1}}, "model_tpm_limit": {"gpt": 7,"#,
    r#" "claude": 100}, "model_rpm_limit": {"claude": 5}, "budget_duration": "30d","#,
    r#" "allowed_models": ["claude"], "temp_budget_increase": 1.5,"#,
    r#" "temp_budget_expiry": "2026-11-20T12:30:00.250000Z"},"#,
    r#" "object_permission_id": "permission","#,
    r#" "object_permission": {"object_permission_id": "permission", "mcp_servers": ["server"],"#,
    r#" "mcp_access_groups": ["group"], "mcp_tool_permissions": {"server": ["tool"]},"#,
    r#" "vector_stores": ["store"], "agents": ["agent"], "agent_access_groups": ["agents"],"#,
    r#" "models": ["claude"], "mcp_toolsets": ["toolset"], "blocked_tools": ["shell"],"#,
    r#" "search_tools": ["search"], "mcp_tool_search_enabled": true, "skills": ["skill"],"#,
    r#" "apps": null}}"#,
);
const BARE_PERMISSION_ENTRY: &str = concat!(
    r#"{"user_id": "bare_permission", "blocked": false, "alias": null, "spend": 0.0,"#,
    r#" "allowed_model_region": null, "default_model": null, "models": [], "budget_id": null,"#,
    r#" "litellm_budget_table": null, "object_permission_id": "bare","#,
    r#" "object_permission": {"object_permission_id": "bare", "mcp_servers": [],"#,
    r#" "mcp_access_groups": [], "mcp_tool_permissions": null, "vector_stores": [],"#,
    r#" "agents": [], "agent_access_groups": [], "models": [], "mcp_toolsets": [],"#,
    r#" "blocked_tools": [], "search_tools": [], "mcp_tool_search_enabled": null, "skills": [],"#,
    r#" "apps": null}}"#,
);
const MERGED_ENTRY: &str = concat!(
    r#"{"user_id": "created", "blocked": false, "alias": null, "spend": 0.0,"#,
    r#" "allowed_model_region": null, "default_model": null, "models": [], "budget_id": null,"#,
    r#" "litellm_budget_table": {"budget_id": "default", "soft_budget": null,"#,
    r#" "max_budget": 5.0, "max_parallel_requests": null, "tpm_limit": null, "rpm_limit": null,"#,
    r#" "tpd_limit": null, "model_max_budget": null, "model_tpm_limit": null,"#,
    r#" "model_rpm_limit": null, "budget_duration": null, "allowed_models": null,"#,
    r#" "temp_budget_increase": null, "temp_budget_expiry": null},"#,
    r#" "object_permission_id": null, "object_permission": null}"#,
);

#[rstest]
#[case::as_created("created", CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ENTRY)]
#[case::a_permission_with_null_arrays("bare_permission", BARE_PERMISSION_ENTRY)]
#[case::the_default_budget_merged_in("created", MERGED_ENTRY)]
#[tokio::test]
async fn an_end_user_entry_python_wrote_is_read_typed_and_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] end_user_id: &str,
    #[case] entry: &str,
) {
    let (end_user, written) = round_trip::<EndUserEntry>(
        topology,
        AuthObject::EndUser(end_user_id.to_owned()),
        AuthObject::EndUser("rewritten".to_owned()),
        entry,
    )
    .await;

    assert_eq!(
        (
            end_user.user_id.as_str(),
            end_user.unread,
            written.as_deref()
        ),
        (end_user_id, Vec::new(), Some(entry))
    );
}
