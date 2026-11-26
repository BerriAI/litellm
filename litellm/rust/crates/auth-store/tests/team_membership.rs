//! A team membership's database read against a Prisma-migrated Postgres. Each row literal is what
//! the repository hands the Python proxy, and each entry literal is what Python caches for that
//! row. `tests/unit/redis/stores/test_auth_cache_team_membership.py` checks both against Python's
//! own model and membership readers, and `redis-store/tests/auth_cache_team_membership.rs`
//! round-trips the entries through Redis. Change all three together.

mod support;

use std::sync::Arc;

use litellm_auth_store::{Error, TeamMembershipDatabase};
use litellm_db_testing::MigratedPostgres;
use litellm_repositories::{Side, TeamMembershipRepository};
use rstest::{fixture, rstest};
use sqlx::PgPool;

use support::{database, dumped, execute, insert_budgets};

const CREATED_ROW: &str = concat!(
    r#"{"user_id":"member","team_id":"created","budget_id":null,"spend":0.0,"total_spend":0.0,"#,
    r#""litellm_budget_table":null}"#,
);
const CREATED_ENTRY: &str = concat!(
    r#"{"user_id": "member", "team_id": "created", "budget_id": null, "spend": 0.0,"#,
    r#" "total_spend": 0.0, "litellm_budget_table": null}"#,
);
const BARE_BUDGET_ROW: &str = concat!(
    r#"{"user_id":"member","team_id":"bare_budget","budget_id":"bare","spend":0.0,"#,
    r#""total_spend":0.0,"litellm_budget_table":{"budget_id":"bare","name":null,"#,
    r#""max_budget":null,"soft_budget":null,"max_parallel_requests":null,"tpm_limit":null,"#,
    r#""rpm_limit":null,"tpd_limit":null,"model_max_budget":null,"model_tpm_limit":null,"#,
    r#""model_rpm_limit":null,"budget_duration":null,"budget_limits":null,"cap_policy":null,"#,
    r#""budget_reset_at":null,"budget_reset_count":0,"allowed_models":[],"#,
    r#""temp_budget_increase":null,"temp_budget_expiry":null,"#,
    r#""created_at":"2026-11-01T00:00:00+00:00","created_by":"test","#,
    r#""updated_at":"2026-11-01T00:00:00+00:00","updated_by":"test"}}"#,
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
const FULL_ROW: &str = concat!(
    r#"{"user_id":"member","team_id":"full","budget_id":"budget","spend":1.5,"#,
    r#""rollover_credit":2.5,"total_spend":3.0,"litellm_budget_table":{"budget_id":"budget","#,
    r#""name":"Budget","max_budget":10.5,"rollover_max_budget":20.0,"soft_budget":8.0,"#,
    r#""max_parallel_requests":3,"tpm_limit":1000,"rpm_limit":10,"tpd_limit":100000,"#,
    r#""model_max_budget":{"claude": {"max_budget": 1}},"model_tpm_limit":{"gpt": "7","#,
    r#" "claude": 100.0},"model_rpm_limit":{"claude": "5"},"budget_duration":"30d","#,
    r#""budget_limits":[],"cap_policy":"hard","budget_reset_at":"2026-12-01T00:00:00+00:00","#,
    r#""budget_reset_count":2,"allowed_models":["claude"],"temp_budget_increase":1.5,"#,
    r#""temp_budget_expiry":"2026-11-20T12:30:00.250+00:00","#,
    r#""created_at":"2026-11-01T00:00:00+00:00","created_by":"test","#,
    r#""updated_at":"2026-11-02T00:00:00+00:00","updated_by":"test"}}"#,
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

struct Memberships {
    postgres: MigratedPostgres,
    rows: Arc<TeamMembershipRepository>,
    database: TeamMembershipDatabase,
}

/// A migrated Postgres holding a membership as a member add stores it without a budget, one with
/// every column set, its budget's too, and one whose budget has only its defaults.
#[fixture]
async fn memberships() -> Memberships {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_budgets(postgres.pool()).await;
    insert_memberships(postgres.pool()).await;
    let rows = Arc::new(TeamMembershipRepository::new(database(&postgres)));
    Memberships {
        database: TeamMembershipDatabase::new(rows.clone()),
        rows,
        postgres,
    }
}

async fn insert_memberships(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id) VALUES ('member', 'created')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id)
           VALUES ('member', 'bare_budget', 'bare')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id, spend, rollover_credit, total_spend)
           VALUES ('member', 'full', 'budget', 1.5, 2.5, 3.0)"#,
    )
    .await;
}

#[rstest]
#[case::as_created("created", CREATED_ROW, CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ROW, FULL_ENTRY)]
#[case::a_budget_with_its_defaults("bare_budget", BARE_BUDGET_ROW, BARE_BUDGET_ENTRY)]
#[tokio::test(flavor = "multi_thread")]
async fn a_team_membership_read_from_the_database_is_the_entry_python_caches(
    #[future(awt)] memberships: Memberships,
    #[case] team_id: &str,
    #[case] row: &str,
    #[case] cached: &str,
) {
    let found = memberships
        .rows
        .find_with_budget("member", team_id, Side::Read)
        .await
        .unwrap()
        .unwrap();
    let read = memberships
        .database
        .team_membership("member", team_id, Side::Read)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(
        (
            serde_json::to_string(&found).unwrap().as_str(),
            dumped(read).as_str()
        ),
        (row, cached)
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_user_who_is_not_a_member_is_not_found(#[future(awt)] memberships: Memberships) {
    let read = memberships
        .database
        .team_membership("member", "gone", Side::Read)
        .await
        .unwrap();

    assert_eq!(read, None);
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_budget_the_model_refuses_fails_the_read_under_the_field_it_sits_in(
    #[future(awt)] memberships: Memberships,
) {
    execute(
        memberships.postgres.pool(),
        r#"UPDATE "LiteLLM_BudgetTable" SET budget_limits = '[{"budget_duration": "1d", "max_budget": 1}]'"#,
    )
    .await;

    let read = memberships
        .database
        .team_membership("member", "full", Side::Read)
        .await;

    assert!(
        matches!(
            &read,
            Err(Error::Row {
                column: "budget_limits",
                within: Some("litellm_budget_table")
            })
        ),
        "{:?}",
        read.err()
    );
}
