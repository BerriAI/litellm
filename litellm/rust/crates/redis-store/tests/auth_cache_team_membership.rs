//! The auth cache's typed team membership entry on real Redis. The entry literals are the ones
//! `tests/unit/redis/stores/test_auth_cache_team_membership.py` checks against Python's own model
//! and membership readers, and `auth-store/tests/team_membership.rs` finds after a database read.
//! Change all three together.

mod support;

use litellm_redis_store::auth_cache::{AuthObject, TeamMembershipEntry};
use litellm_redis_testing::Topology;
use rstest::rstest;
use support::auth_cache::round_trip;

const CREATED_ENTRY: &str = concat!(
    r#"{"user_id": "member", "team_id": "created", "budget_id": null, "spend": 0.0,"#,
    r#" "total_spend": 0.0, "litellm_budget_table": null}"#,
);
const BARE_BUDGET_ENTRY: &str = concat!(
    r#"{"user_id": "member", "team_id": "bare_budget", "budget_id": "bare", "spend": 0.0,"#,
    r#" "total_spend": 0.0, "litellm_budget_table": {"budget_id": "bare", "soft_budget": null,"#,
    r#" "max_budget": null, "max_parallel_requests": null, "tpm_limit": null,"#,
    r#" "rpm_limit": null, "tpd_limit": null, "model_max_budget": null,"#,
    r#" "model_tpm_limit": null, "model_rpm_limit": null, "budget_duration": null,"#,
    r#" "allowed_models": [], "temp_budget_increase": null, "temp_budget_expiry": null,"#,
    r#" "budget_reset_at": null, "created_at": "2026-11-01T00:00:00Z"}}"#,
);
const FULL_ENTRY: &str = concat!(
    r#"{"user_id": "member", "team_id": "full", "budget_id": "budget", "spend": 1.5,"#,
    r#" "total_spend": 3.0, "rollover_credit": 2.5,"#,
    r#" "litellm_budget_table": {"budget_id": "budget", "soft_budget": 8.0, "max_budget": 10.5,"#,
    r#" "rollover_max_budget": 20.0, "max_parallel_requests": 3, "tpm_limit": 1000,"#,
    r#" "rpm_limit": 10, "tpd_limit": 100000,"#,
    r#" "model_max_budget": {"claude": {"max_budget": 1}}, "model_tpm_limit": {"gpt": 7,"#,
    r#" "claude": 100}, "model_rpm_limit": {"claude": 5}, "budget_duration": "30d","#,
    r#" "allowed_models": ["claude"], "temp_budget_increase": 1.5,"#,
    r#" "temp_budget_expiry": "2026-11-20T12:30:00.250000Z","#,
    r#" "budget_reset_at": "2026-12-01T00:00:00Z", "created_at": "2026-11-01T00:00:00Z"}}"#,
);

#[rstest]
#[case::as_created("created", CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ENTRY)]
#[case::a_budget_with_its_defaults("bare_budget", BARE_BUDGET_ENTRY)]
#[tokio::test]
async fn a_team_membership_entry_python_wrote_is_read_typed_and_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] team_id: &str,
    #[case] entry: &str,
) {
    let (membership, written) = round_trip::<TeamMembershipEntry>(
        topology,
        AuthObject::TeamMembership {
            team_id: team_id.to_owned(),
            user_id: "member".to_owned(),
        },
        AuthObject::TeamMembership {
            team_id: "rewritten".to_owned(),
            user_id: "member".to_owned(),
        },
        entry,
    )
    .await;

    assert_eq!(
        (
            membership.team_id.as_str(),
            membership.unread,
            written.as_deref()
        ),
        (team_id, Vec::new(), Some(entry))
    );
}
