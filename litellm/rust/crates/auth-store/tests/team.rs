//! A team's database read against a Prisma-migrated Postgres. Each row literal is what the
//! repository hands the Python proxy, and each entry literal is what
//! `LiteLLM_TeamTableCachedObj(**row).model_dump(mode="json")` gave for that row on 2026-11-23,
//! less `last_refreshed_at`, which the host that caches the team stamps.

mod support;

use std::sync::Arc;

use litellm_auth_store::{Error, TeamDatabase};
use litellm_db_testing::MigratedPostgres;
use litellm_repositories::{Side, TeamRepository};
use rstest::rstest;
use sqlx::PgPool;

use support::{database, dumped, execute};

const CREATED_ROW: &str = concat!(
    r#"{"team_id":"created","team_alias":null,"organization_id":null,"admins":[],"#,
    r#""members":[],"members_with_roles":{},"metadata":{},"max_budget":null,"#,
    r#""soft_budget":null,"spend":0.0,"models":[],"max_parallel_requests":null,"#,
    r#""tpm_limit":null,"rpm_limit":null,"tpd_limit":null,"budget_duration":null,"#,
    r#""budget_reset_at":null,"blocked":false,"created_at":"2026-11-19T10:00:00.123+00:00","#,
    r#""updated_at":"2026-11-19T10:00:00+00:00","model_spend":{},"model_max_budget":{},"#,
    r#""router_settings":{},"team_member_permissions":[],"access_group_ids":[],"policies":[],"#,
    r#""default_team_member_models":[],"budget_limits":null,"model_id":null,"#,
    r#""allow_team_guardrail_config":false,"object_permission_id":null,"#,
    r#""litellm_model_table":null}"#,
);
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
const FULL_ROW: &str = concat!(
    r#"{"team_id":"full","team_alias":"Full","organization_id":"org","admins":["admin"],"#,
    r#""members":["member"],"members_with_roles":[{"role": "admin", "extra": 1,"#,
    r#" "user_id": "admin"}, {"role": "user", "user_email": "m@example.com"}],"#,
    r#""metadata":{"guardrails": ["pii"]},"max_budget":10.5,"soft_budget":5.25,"spend":1.5,"#,
    r#""models":["claude","all-proxy-models"],"max_parallel_requests":3,"tpm_limit":1000,"#,
    r#""rpm_limit":10,"tpd_limit":100000,"budget_duration":"30d","#,
    r#""budget_reset_at":"2026-12-01T00:00:00+00:00","blocked":true,"#,
    r#""created_at":"2026-11-19T10:00:00.123+00:00","updated_at":"2026-11-19T10:00:00+00:00","#,
    r#""model_spend":{"claude": 1.5},"model_max_budget":{"claude": {"budget_limit": 2.0}},"#,
    r#""router_settings":{"num_retries": 2},"team_member_permissions":["/key/generate"],"#,
    r#""access_group_ids":["group"],"policies":["policy"],"#,
    r#""default_team_member_models":["claude"],"budget_limits":[{"max_budget": 1,"#,
    r#" "window_start": "2026-11-01T00:00:00+00:00", "budget_duration": "1d"},"#,
    r#" {"max_budget": 2.5, "budget_duration": "30d"}],"model_id":7,"#,
    r#""allow_team_guardrail_config":true,"object_permission_id":"permission","#,
    r#""litellm_model_table":{"id": 7, "created_by": "test", "updated_by": "test","#,
    r#" "model_aliases": {"gpt": "claude"}}}"#,
);
const FULL_ENTRY: &str = concat!(
    r#"{"team_alias": "Full", "team_id": "full", "organization_id": "org","#,
    r#" "admins": ["admin"], "members": ["member"],"#,
    r#" "members_with_roles": [{"user_id": "admin", "user_email": null, "role": "admin"},"#,
    r#" {"user_id": null, "user_email": "m@example.com", "role": "user"}],"#,
    r#" "team_member_permissions": ["/key/generate"], "metadata": {"guardrails": ["pii"]},"#,
    r#" "tpm_limit": 1000, "rpm_limit": 10, "tpd_limit": 100000, "max_budget": 10.5,"#,
    r#" "soft_budget": 5.25, "budget_duration": "30d","#,
    r#" "budget_limits": [{"budget_duration": "1d", "max_budget": 1.0, "reset_at": null,"#,
    r#" "window_start": "2026-11-01T00:00:00Z"}, {"budget_duration": "30d","#,
    r#" "max_budget": 2.5, "reset_at": null}], "models": ["claude", "all-proxy-models"],"#,
    r#" "blocked": true, "router_settings": {"num_retries": 2},"#,
    r#" "access_group_ids": ["group"], "default_team_member_models": ["claude"],"#,
    r#" "spend": 1.5, "max_parallel_requests": 3, "budget_reset_at": "2026-12-01T00:00:00Z","#,
    r#" "model_id": 7, "model_spend": {"claude": 1.5},"#,
    r#" "model_max_budget": {"claude": {"budget_limit": 2.0}}, "policies": ["policy"],"#,
    r#" "allow_team_guardrail_config": true, "litellm_model_table": {"id": 7,"#,
    r#" "model_aliases": {"gpt": "claude"}, "created_by": "test", "updated_by": "test","#,
    r#" "team": null}, "object_permission": null, "object_permission_id": "permission","#,
    r#" "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z","#,
    r#" "last_refreshed_at": null}"#,
);

fn teams(postgres: &MigratedPostgres) -> TeamRepository {
    TeamRepository::new(database(postgres))
}

/// A team as `/team/new` stores it when asked for nothing, and one with every column set.
async fn insert_teams(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamTable" (team_id, created_at, updated_at)
           VALUES ('created', '2026-11-19 10:00:00.123', '2026-11-19 10:00:00')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_BudgetTable" (budget_id, created_by, updated_by)
           VALUES ('budget', 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_OrganizationTable"
               (organization_id, organization_alias, budget_id, created_by, updated_by)
           VALUES ('org', 'Org', 'budget', 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id) VALUES ('permission')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ModelTable" (id, aliases, created_by, updated_by)
           VALUES (7, '{"gpt": "claude"}', 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamTable" (
               team_id, team_alias, organization_id, object_permission_id, admins, members,
               members_with_roles, metadata, max_budget, soft_budget, spend, models,
               max_parallel_requests, tpm_limit, rpm_limit, tpd_limit, budget_duration,
               budget_reset_at, blocked, created_at, updated_at, model_spend, model_max_budget,
               router_settings, team_member_permissions, access_group_ids, policies,
               default_team_member_models, budget_limits, model_id, allow_team_guardrail_config
           ) VALUES (
               'full', 'Full', 'org', 'permission', ARRAY['admin'], ARRAY['member'],
               '[{"role": "admin", "user_id": "admin", "extra": 1}, {"role": "user", "user_email": "m@example.com"}]',
               '{"guardrails": ["pii"]}', 10.5, 5.25, 1.5, ARRAY['claude', 'all-proxy-models'],
               3, 1000, 10, 100000, '30d',
               '2026-12-01 00:00:00', true, '2026-11-19 10:00:00.123', '2026-11-19 10:00:00',
               '{"claude": 1.5}', '{"claude": {"budget_limit": 2.0}}',
               '{"num_retries": 2}', ARRAY['/key/generate'], ARRAY['group'], ARRAY['policy'],
               ARRAY['claude'], '[{"budget_duration": "1d", "max_budget": 1, "window_start": "2026-11-01T00:00:00+00:00"}, {"budget_duration": "30d", "max_budget": 2.5}]', 7, true
           )"#,
    )
    .await;
}

#[rstest]
#[case::as_created("created", CREATED_ROW, CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ROW, FULL_ENTRY)]
#[tokio::test(flavor = "multi_thread")]
async fn a_team_read_from_the_database_is_the_entry_python_caches(
    #[case] team_id: &str,
    #[case] row: &str,
    #[case] cached: &str,
) {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_teams(postgres.pool()).await;

    let found = teams(&postgres)
        .find_with_model_aliases(team_id, Side::Read)
        .await
        .unwrap()
        .unwrap();
    let read = TeamDatabase::new(Arc::new(teams(&postgres)))
        .team(team_id, Side::Read)
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
async fn a_team_no_row_has_is_not_found() {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_teams(postgres.pool()).await;

    let read = TeamDatabase::new(Arc::new(teams(&postgres)))
        .team("gone", Side::Read)
        .await
        .unwrap();

    assert_eq!(read, None);
}

#[rstest]
#[case::a_roster_that_is_no_list(
    r#"UPDATE "LiteLLM_TeamTable" SET members_with_roles = '{"admin": "a"}'"#,
    "members_with_roles"
)]
#[case::a_member_with_no_role(
    r#"UPDATE "LiteLLM_TeamTable" SET members_with_roles = '[{"user_id": "a"}]'"#,
    "members_with_roles"
)]
#[case::a_member_nobody_can_name(
    r#"UPDATE "LiteLLM_TeamTable" SET members_with_roles = '[{"role": "user"}]'"#,
    "members_with_roles"
)]
#[case::metadata_that_is_a_list(r#"UPDATE "LiteLLM_TeamTable" SET metadata = '[1]'"#, "metadata")]
#[tokio::test(flavor = "multi_thread")]
async fn a_team_row_the_model_refuses_fails_the_read(
    #[case] corrupt: &'static str,
    #[case] column: &str,
) {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_teams(postgres.pool()).await;
    execute(postgres.pool(), corrupt).await;

    let refused = TeamDatabase::new(Arc::new(teams(&postgres)))
        .team("created", Side::Read)
        .await;

    assert!(
        matches!(refused, Err(Error::Row { column: named }) if named == column),
        "{refused:?}"
    );
}
