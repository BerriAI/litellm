//! The auth cache's typed user entry on real Redis, under the bare user id. The entry literals
//! are the ones `tests/unit/redis/stores/test_auth_cache_user.py` checks against Python's own model
//! and user reader, and `auth-store/tests/user.rs` finds after a database read. Change all three
//! together.

mod support;

use litellm_redis_store::auth_cache::{AuthObject, UserEntry};
use litellm_redis_testing::Topology;
use rstest::rstest;
use support::auth_cache::round_trip;

const CREATED_ENTRY: &str = concat!(
    r#"{"user_id": "created", "user_alias": null, "team_id": null, "sso_user_id": null,"#,
    r#" "organization_id": null, "object_permission_id": null, "password_reset_required": null,"#,
    r#" "last_breach_check_at": null, "teams": [], "user_role": null, "max_budget": null,"#,
    r#" "budget_limits": null, "spend": 0.0, "user_email": null, "models": [], "metadata": {},"#,
    r#" "max_parallel_requests": null, "tpm_limit": null, "rpm_limit": null,"#,
    r#" "budget_duration": null, "budget_reset_at": null, "allowed_cache_controls": [],"#,
    r#" "policies": [], "model_spend": {}, "model_max_budget": {},"#,
    r#" "created_at": "2026-11-01T00:00:00Z", "updated_at": "2026-11-01T00:00:00Z","#,
    r#" "organization_memberships": [], "object_permission": null}"#,
);
const NULLS_ENTRY: &str = concat!(
    r#"{"user_id": "nulls", "user_alias": null, "team_id": null, "sso_user_id": null,"#,
    r#" "organization_id": null, "object_permission_id": null, "password_reset_required": null,"#,
    r#" "last_breach_check_at": null, "teams": [], "user_role": null, "max_budget": null,"#,
    r#" "budget_limits": null, "spend": 0.0, "user_email": null, "models": [],"#,
    r#" "metadata": null, "max_parallel_requests": null, "tpm_limit": null, "rpm_limit": null,"#,
    r#" "budget_duration": null, "budget_reset_at": null, "allowed_cache_controls": [],"#,
    r#" "policies": [], "model_spend": null, "model_max_budget": null,"#,
    r#" "created_at": "2026-11-01T00:00:00Z", "updated_at": "2026-11-01T00:00:00Z","#,
    r#" "organization_memberships": [], "object_permission": null}"#,
);
const FULL_ENTRY: &str = concat!(
    r#"{"user_id": "full", "user_alias": "Full", "team_id": "team", "sso_user_id": "sso","#,
    r#" "organization_id": "org-a", "object_permission_id": "permission","#,
    r#" "password_reset_required": true, "last_breach_check_at": "2026-11-20T12:30:00.250000Z","#,
    r#" "teams": ["team", "other-team"], "user_role": "internal_user", "max_budget": 10.5,"#,
    r#" "rollover_max_budget": 20.0, "rollover_credit": 2.5,"#,
    r#" "budget_limits": [{"budget_duration": "1d", "max_budget": 5.0,"#,
    r#" "reset_at": "2026-12-01T00:00:00Z", "window_start": "2026-11-30T00:00:00Z"}],"#,
    r#" "spend": 1.5, "user_email": "full@example.com", "models": ["claude", "gpt"],"#,
    r#" "metadata": {"guardrails": ["pii"]}, "max_parallel_requests": 3, "tpm_limit": 1000,"#,
    r#" "rpm_limit": 10, "budget_duration": "30d", "budget_reset_at": "2026-12-01T00:00:00Z","#,
    r#" "allowed_cache_controls": ["no-cache"], "policies": ["policy"],"#,
    r#" "model_spend": {"claude": 1.5}, "model_max_budget": {"claude": {"max_budget": 1}},"#,
    r#" "created_at": "2026-11-01T00:00:00Z", "updated_at": "2026-11-02T00:00:00Z","#,
    r#" "organization_memberships": [{"user_id": "full", "organization_id": "org-a","#,
    r#" "user_role": null, "spend": 0.0, "budget_id": null,"#,
    r#" "created_at": "2026-11-03T00:00:00Z", "updated_at": "2026-11-03T00:00:00Z","#,
    r#" "user": null, "litellm_budget_table": null, "user_email": null}, {"user_id": "full","#,
    r#" "organization_id": "org-b", "user_role": "org_admin", "spend": 0.5,"#,
    r#" "budget_id": "budget", "created_at": "2026-11-03T00:00:00Z","#,
    r#" "updated_at": "2026-11-04T00:00:00.500000Z", "user": null,"#,
    r#" "litellm_budget_table": null, "user_email": null}], "object_permission": null}"#,
);

#[rstest]
#[case::as_created("created", CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ENTRY)]
#[case::json_nulls("nulls", NULLS_ENTRY)]
#[tokio::test]
async fn a_user_entry_python_wrote_is_read_typed_and_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] user_id: &str,
    #[case] entry: &str,
) {
    let (user, written) = round_trip::<UserEntry>(
        topology,
        AuthObject::User(user_id.to_owned()),
        AuthObject::User("rewritten".to_owned()),
        entry,
    )
    .await;

    assert_eq!((user.unread, written.as_deref()), (Vec::new(), Some(entry)));
}
