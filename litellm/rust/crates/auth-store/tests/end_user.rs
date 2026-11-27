//! An end user's database read against a Prisma-migrated Postgres. Each row literal is what the
//! repository hands the Python proxy, and each entry literal is what Python caches for that row.
//! `tests/unit/redis/stores/test_auth_cache_end_user.py` checks both against Python's own model and
//! end user reader, and `redis-store/tests/auth_cache_end_user.rs` round-trips the entries through
//! Redis. Change all three together.

mod support;

use std::sync::Arc;

use litellm_auth_store::{EndUserDatabase, Error};
use litellm_db_testing::MigratedPostgres;
use litellm_repositories::EndUserRepository;
use rstest::{fixture, rstest};
use sqlx::PgPool;

use support::{database, dumped, execute, insert_budgets};

const CREATED_ROW: &str = concat!(
    r#"{"user_id":"created","alias":null,"spend":0.0,"allowed_model_region":null,"#,
    r#""default_model":null,"models":[],"budget_id":null,"litellm_budget_table":null,"#,
    r#""object_permission_id":null,"object_permission":null,"blocked":false}"#,
);
const CREATED_ENTRY: &str = concat!(
    r#"{"user_id": "created", "blocked": false, "alias": null, "spend": 0.0,"#,
    r#" "allowed_model_region": null, "default_model": null, "models": [], "budget_id": null,"#,
    r#" "litellm_budget_table": null, "object_permission_id": null, "object_permission": null}"#,
);
const FULL_ROW: &str = concat!(
    r#"{"user_id":"full","alias":"Full","spend":1.5,"allowed_model_region":"eu","#,
    r#""default_model":"claude","models":["claude","gpt"],"budget_id":"budget","#,
    r#""litellm_budget_table":{"budget_id":"budget","name":"Budget","max_budget":10.5,"#,
    r#""rollover_max_budget":20.0,"soft_budget":8.0,"max_parallel_requests":3,"tpm_limit":1000,"#,
    r#""rpm_limit":10,"tpd_limit":100000,"model_max_budget":{"claude": {"max_budget": 1}},"#,
    r#""model_tpm_limit":{"gpt": "7", "claude": 100.0},"model_rpm_limit":{"claude": "5"},"#,
    r#""budget_duration":"30d","budget_limits":[],"cap_policy":"hard","#,
    r#""budget_reset_at":"2026-12-01T00:00:00+00:00","budget_reset_count":2,"#,
    r#""allowed_models":["claude"],"temp_budget_increase":1.5,"#,
    r#""temp_budget_expiry":"2026-11-20T12:30:00.250+00:00","#,
    r#""created_at":"2026-11-01T00:00:00+00:00","created_by":"test","#,
    r#""updated_at":"2026-11-02T00:00:00+00:00","updated_by":"test"},"#,
    r#""object_permission_id":"permission","#,
    r#""object_permission":{"object_permission_id":"permission","mcp_servers":["server"],"#,
    r#""mcp_access_groups":["group"],"mcp_tool_permissions":{"server": ["tool"]},"#,
    r#""vector_stores":["store"],"agents":["agent"],"agent_access_groups":["agents"],"#,
    r#""models":["claude"],"blocked_tools":["shell"],"mcp_toolsets":["toolset"],"#,
    r#""search_tools":["search"],"mcp_tool_search_enabled":true,"skills":["skill"]},"#,
    r#""blocked":true}"#,
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
const BARE_PERMISSION_ROW: &str = concat!(
    r#"{"user_id":"bare_permission","alias":null,"spend":0.0,"allowed_model_region":null,"#,
    r#""default_model":null,"models":[],"budget_id":null,"litellm_budget_table":null,"#,
    r#""object_permission_id":"bare","object_permission":{"object_permission_id":"bare","#,
    r#""mcp_servers":[],"mcp_access_groups":[],"mcp_tool_permissions":null,"vector_stores":[],"#,
    r#""agents":[],"agent_access_groups":[],"models":[],"blocked_tools":[],"mcp_toolsets":[],"#,
    r#""search_tools":[],"mcp_tool_search_enabled":null,"skills":[]},"blocked":false}"#,
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

struct EndUsers {
    postgres: MigratedPostgres,
    rows: Arc<EndUserRepository>,
    database: EndUserDatabase,
}

/// A migrated Postgres holding an end user as a request first records it, one with every column
/// set, its budget's and object permission's too, and one whose object permission has every
/// array column NULL.
#[fixture]
async fn end_users() -> EndUsers {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_budgets(postgres.pool()).await;
    insert_end_users(postgres.pool()).await;
    let rows = Arc::new(EndUserRepository::new(database(&postgres)));
    EndUsers {
        database: EndUserDatabase::new(rows.clone()),
        rows,
        postgres,
    }
}

async fn insert_end_users(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_EndUserTable" (user_id) VALUES ('created')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ObjectPermissionTable" (
               object_permission_id, mcp_servers, mcp_access_groups, mcp_tool_permissions,
               vector_stores, agents, agent_access_groups, models, blocked_tools, mcp_toolsets,
               search_tools, mcp_tool_search_enabled, skills, apps
           ) VALUES (
               'permission', ARRAY['server'], ARRAY['group'], '{"server": ["tool"]}',
               ARRAY['store'], ARRAY['agent'], ARRAY['agents'], ARRAY['claude'], ARRAY['shell'],
               ARRAY['toolset'], ARRAY['search'], true, ARRAY['skill'], ARRAY['app']
           )"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_EndUserTable" (
               user_id, alias, spend, allowed_model_region, default_model, models, budget_id,
               object_permission_id, blocked
           ) VALUES (
               'full', 'Full', 1.5, 'eu', 'claude', ARRAY['claude', 'gpt'], 'budget', 'permission', true
           )"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ObjectPermissionTable" (
               object_permission_id, mcp_servers, mcp_access_groups, vector_stores, agents,
               agent_access_groups, models, blocked_tools, mcp_toolsets, search_tools, skills
           ) VALUES ('bare', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, object_permission_id)
           VALUES ('bare_permission', 'bare')"#,
    )
    .await;
}

#[rstest]
#[case::as_created("created", CREATED_ROW, CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ROW, FULL_ENTRY)]
#[case::a_permission_with_null_arrays(
    "bare_permission",
    BARE_PERMISSION_ROW,
    BARE_PERMISSION_ENTRY
)]
#[tokio::test(flavor = "multi_thread")]
async fn an_end_user_read_from_the_database_is_the_entry_python_caches(
    #[future(awt)] end_users: EndUsers,
    #[case] end_user_id: &str,
    #[case] row: &str,
    #[case] cached: &str,
) {
    let found = end_users
        .rows
        .find_with_budget_and_object_permission(end_user_id)
        .await
        .unwrap()
        .unwrap();
    let read = end_users
        .database
        .end_user(end_user_id)
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
async fn an_end_user_no_row_has_is_not_found(#[future(awt)] end_users: EndUsers) {
    let read = end_users.database.end_user("gone").await.unwrap();

    assert_eq!(read, None);
}

#[rstest]
#[case::a_region_the_model_does_not_take(
    r#"UPDATE "LiteLLM_EndUserTable" SET allowed_model_region = 'ap'"#,
    "allowed_model_region",
    None
)]
#[case::a_budget_with_concurrent_windows(
    r#"UPDATE "LiteLLM_BudgetTable" SET budget_limits = '[{"budget_duration": "1d", "max_budget": 1}]'"#,
    "budget_limits",
    Some("litellm_budget_table")
)]
#[case::a_permission_the_model_refuses(
    r#"UPDATE "LiteLLM_ObjectPermissionTable" SET mcp_tool_permissions = '{"server": "tool"}'"#,
    "mcp_tool_permissions",
    Some("object_permission")
)]
#[tokio::test(flavor = "multi_thread")]
async fn an_end_user_row_the_model_refuses_fails_the_read(
    #[future(awt)] end_users: EndUsers,
    #[case] corrupt: &'static str,
    #[case] column: &str,
    #[case] within: Option<&str>,
) {
    execute(end_users.postgres.pool(), corrupt).await;

    let refused = end_users.database.end_user("full").await;

    assert!(
        matches!(&refused, Err(Error::Row { column: named, within: under }) if *named == column && *under == within),
        "{:?}",
        refused.err()
    );
}
