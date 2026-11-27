//! A user's database read against a Prisma-migrated Postgres. Each row literal is what the
//! repository hands the Python proxy, and each entry literal is what Python caches for that row
//! under the bare user id. `tests/unit/redis/stores/test_auth_cache_user.py` checks both against
//! Python's own model and user reader, and `redis-store/tests/auth_cache_user.rs` round-trips the
//! entries through Redis. Change all three together.

mod support;

use std::sync::Arc;

use litellm_auth_store::{Error, SignedIn, UserDatabase};
use litellm_db_testing::MigratedPostgres;
use litellm_repositories::{Side, SignIn, SsoLink, UserRepository};
use rstest::{fixture, rstest};
use sqlx::PgPool;

use support::{database, dumped, execute, insert_budgets};

const CREATED_ROW: &str = concat!(
    r#"{"user_id":"created","user_alias":null,"team_id":null,"sso_user_id":null,"#,
    r#""organization_id":null,"object_permission_id":null,"password_reset_required":null,"#,
    r#""last_breach_check_at":null,"teams":[],"user_role":null,"max_budget":null,"#,
    r#""budget_limits":null,"spend":0.0,"user_email":null,"models":[],"metadata":{},"#,
    r#""max_parallel_requests":null,"tpm_limit":null,"rpm_limit":null,"budget_duration":null,"#,
    r#""budget_reset_at":null,"allowed_cache_controls":[],"policies":[],"model_spend":{},"#,
    r#""model_max_budget":{},"created_at":"2026-11-01T00:00:00+00:00","#,
    r#""updated_at":"2026-11-01T00:00:00+00:00","organization_memberships":[]}"#,
);
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
const NULLS_ROW: &str = concat!(
    r#"{"user_id":"nulls","user_alias":null,"team_id":null,"sso_user_id":null,"#,
    r#""organization_id":null,"object_permission_id":null,"password_reset_required":null,"#,
    r#""last_breach_check_at":null,"teams":[],"user_role":null,"max_budget":null,"#,
    r#""budget_limits":null,"spend":0.0,"user_email":null,"models":[],"metadata":null,"#,
    r#""max_parallel_requests":null,"tpm_limit":null,"rpm_limit":null,"budget_duration":null,"#,
    r#""budget_reset_at":null,"allowed_cache_controls":[],"policies":[],"model_spend":null,"#,
    r#""model_max_budget":null,"created_at":"2026-11-01T00:00:00+00:00","#,
    r#""updated_at":"2026-11-01T00:00:00+00:00","organization_memberships":[]}"#,
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
const FULL_ROW: &str = concat!(
    r#"{"user_id":"full","user_alias":"Full","team_id":"team","sso_user_id":"sso","#,
    r#""organization_id":"org-a","object_permission_id":"permission","#,
    r#""password_reset_required":true,"last_breach_check_at":"2026-11-20T12:30:00.250+00:00","#,
    r#""teams":["team","other-team"],"user_role":"internal_user","max_budget":10.5,"#,
    r#""rollover_max_budget":20.0,"rollover_credit":2.5,"#,
    r#""budget_limits":[{"later": 1, "reset_at": "2026-12-01T00:00:00+00:00", "max_budget": 5,"#,
    r#" "window_start": "2026-11-30T00:00:00+00:00", "budget_duration": "1d"}],"spend":1.5,"#,
    r#""user_email":"full@example.com","models":["claude","gpt"],"#,
    r#""metadata":{"guardrails": ["pii"]},"max_parallel_requests":3,"tpm_limit":1000,"#,
    r#""rpm_limit":10,"budget_duration":"30d","budget_reset_at":"2026-12-01T00:00:00+00:00","#,
    r#""allowed_cache_controls":["no-cache"],"policies":["policy"],"#,
    r#""model_spend":{"claude": 1.5},"model_max_budget":{"claude": {"max_budget": 1}},"#,
    r#""created_at":"2026-11-01T00:00:00+00:00","updated_at":"2026-11-02T00:00:00+00:00","#,
    r#""organization_memberships":[{"user_id":"full","organization_id":"org-a","#,
    r#""user_role":null,"spend":0.0,"budget_id":null,"created_at":"2026-11-03T00:00:00+00:00","#,
    r#""updated_at":"2026-11-03T00:00:00+00:00"},{"user_id":"full","organization_id":"org-b","#,
    r#""user_role":"org_admin","spend":0.5,"budget_id":"budget","#,
    r#""created_at":"2026-11-03T00:00:00+00:00","updated_at":"2026-11-04T00:00:00.500+00:00"}]}"#,
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

struct Users {
    postgres: MigratedPostgres,
    rows: Arc<UserRepository>,
    database: UserDatabase,
}

/// A user inserted with only its id, one with every column set and a membership in two
/// organizations, the later one by id inserted first, and one holding a JSON `null` in each JSON
/// column.
#[fixture]
async fn users() -> Users {
    let postgres = MigratedPostgres::start().await.unwrap();
    insert_budgets(postgres.pool()).await;
    insert_users(postgres.pool()).await;
    let rows = Arc::new(UserRepository::new(database(&postgres)));
    Users {
        database: UserDatabase::new(rows.clone()),
        rows,
        postgres,
    }
}

async fn insert_users(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id) VALUES ('permission')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, created_by, updated_by)
           VALUES ('org-a', 'A', 'budget', 'test', 'test'), ('org-b', 'B', 'budget', 'test', 'test')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_UserTable" (user_id, created_at, updated_at)
           VALUES ('created', '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_UserTable" (
               user_id, user_alias, team_id, sso_user_id, organization_id, object_permission_id, password,
               password_reset_required, last_breach_check_at, teams, user_role, max_budget,
               rollover_max_budget, rollover_credit, budget_limits, spend, user_email, models, metadata,
               max_parallel_requests, tpm_limit, rpm_limit, budget_duration, budget_reset_at,
               allowed_cache_controls, policies, model_spend, model_max_budget, created_at, updated_at
           ) VALUES (
               'full', 'Full', 'team', 'sso', 'org-a', 'permission', 'hash',
               true, '2026-11-20 12:30:00.25', ARRAY['team', 'other-team'], 'internal_user', 10.5,
               20.0, 2.5,
               '[{"budget_duration": "1d", "max_budget": 5, "reset_at": "2026-12-01T00:00:00+00:00", "window_start": "2026-11-30T00:00:00+00:00", "later": 1}]',
               1.5, 'full@example.com', ARRAY['claude', 'gpt'], '{"guardrails": ["pii"]}',
               3, 1000, 10, '30d', '2026-12-01 00:00:00',
               ARRAY['no-cache'], ARRAY['policy'], '{"claude": 1.5}', '{"claude": {"max_budget": 1}}',
               '2026-11-01 00:00:00', '2026-11-02 00:00:00'
           )"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_UserTable" (user_id, budget_limits, metadata, model_spend, model_max_budget, created_at, updated_at)
           VALUES ('nulls', 'null', 'null', 'null', 'null', '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_OrganizationMembership" (user_id, organization_id, user_role, spend, budget_id, created_at, updated_at)
           VALUES ('full', 'org-b', 'org_admin', 0.5, 'budget', '2026-11-03 00:00:00', '2026-11-04 00:00:00.5')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_OrganizationMembership" (user_id, organization_id, created_at, updated_at)
           VALUES ('full', 'org-a', '2026-11-03 00:00:00', '2026-11-03 00:00:00')"#,
    )
    .await;
}

#[rstest]
#[case::as_created("created", CREATED_ROW, CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ROW, FULL_ENTRY)]
#[case::json_nulls("nulls", NULLS_ROW, NULLS_ENTRY)]
#[tokio::test(flavor = "multi_thread")]
async fn a_user_read_from_the_database_is_the_entry_python_caches(
    #[future(awt)] users: Users,
    #[case] user_id: &str,
    #[case] row: &str,
    #[case] cached: &str,
) {
    let sign_in = SignIn::by_id(user_id);
    let found = users
        .rows
        .resolve(&sign_in, Side::Read)
        .await
        .unwrap()
        .unwrap();
    let read = users
        .database
        .user(user_id, Side::Read)
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
async fn a_user_no_row_has_is_not_found(#[future(awt)] users: Users) {
    let read = users.database.user("gone", Side::Read).await.unwrap();

    assert_eq!(read, None);
}

#[rstest]
#[case::metadata_that_is_a_list(
    r#"UPDATE "LiteLLM_UserTable" SET metadata = '[1]'"#,
    "metadata",
    None
)]
#[case::metadata_text_the_model_does_not_parse(
    r#"UPDATE "LiteLLM_UserTable" SET metadata = '"{}"'"#,
    "metadata",
    None
)]
#[case::model_spend_that_is_a_list(
    r#"UPDATE "LiteLLM_UserTable" SET model_spend = '[1]'"#,
    "model_spend",
    None
)]
#[case::model_max_budget_that_is_text(
    r#"UPDATE "LiteLLM_UserTable" SET model_max_budget = '"{}"'"#,
    "model_max_budget",
    None
)]
#[case::a_budget_window_without_its_max_budget(
    r#"UPDATE "LiteLLM_UserTable" SET budget_limits = '[{"budget_duration": "1d"}]'"#,
    "budget_limits",
    None
)]
#[case::a_membership_without_its_spend(
    r#"UPDATE "LiteLLM_OrganizationMembership" SET spend = NULL WHERE organization_id = 'org-b'"#,
    "spend",
    Some("organization_memberships")
)]
#[case::a_membership_without_its_creation_time(
    r#"UPDATE "LiteLLM_OrganizationMembership" SET created_at = NULL WHERE organization_id = 'org-a'"#,
    "created_at",
    Some("organization_memberships")
)]
#[case::a_membership_without_its_update_time(
    r#"UPDATE "LiteLLM_OrganizationMembership" SET updated_at = NULL WHERE organization_id = 'org-b'"#,
    "updated_at",
    Some("organization_memberships")
)]
#[tokio::test(flavor = "multi_thread")]
async fn a_user_row_the_model_refuses_fails_the_read(
    #[future(awt)] users: Users,
    #[case] corrupt: &'static str,
    #[case] column: &str,
    #[case] within: Option<&str>,
) {
    execute(users.postgres.pool(), corrupt).await;

    let refused = users.database.user("full", Side::Read).await;

    assert!(
        matches!(&refused, Err(Error::Row { column: named, within: under }) if *named == column && *under == within),
        "{:?}",
        refused.err()
    );
}

fn sign_in(
    user_id: &str,
    sso_user_id: Option<&str>,
    user_email: Option<&str>,
    links_any_user: bool,
) -> SignIn {
    SignIn {
        user_id: user_id.to_owned(),
        sso_user_id: sso_user_id.map(str::to_owned),
        user_email: user_email.map(str::to_owned),
        email_verified: None,
        links_any_user,
    }
}

async fn stored_sso_user_id(pool: &PgPool, user_id: &str) -> Option<String> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Assertion reads are outside the compile-checked query set"
    )]
    let read =
        sqlx::query_scalar(r#"SELECT sso_user_id FROM "LiteLLM_UserTable" WHERE user_id = $1"#);
    read.bind(user_id).fetch_one(pool).await.unwrap()
}

/// Python caches the row it read before linking its own id as its SSO id, so the entry still has
/// none while the row now holds it.
#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_sign_in_that_links_its_own_id_is_the_entry_python_caches_for_the_row_as_read(
    #[future(awt)] users: Users,
) {
    let signed_in = users
        .database
        .sign_in(
            &sign_in("created", Some("created"), None, false),
            Side::Read,
        )
        .await
        .unwrap()
        .unwrap();
    let stored = stored_sso_user_id(users.postgres.pool(), "created").await;

    assert_eq!(
        (
            dumped(signed_in.user).as_str(),
            signed_in.sso_link,
            signed_in.owes_email,
            stored.as_deref()
        ),
        (CREATED_ENTRY, SsoLink::Linked, false, Some("created"))
    );
}

/// `full` holds `sso` and `full@example.com`, `created` holds neither, and `emailed` holds only
/// `emailed@example.com`.
#[rstest]
#[case::by_id_alone(
    sign_in("full", None, None, false),
    "full",
    Some("sso"),
    SsoLink::Unchanged,
    false
)]
#[case::by_id_with_an_email_the_row_lacks(
    sign_in("created", None, Some("created@example.com"), false),
    "created",
    None,
    SsoLink::Unchanged,
    true
)]
#[case::by_id_with_the_email_the_row_holds(
    sign_in("full", None, Some("full@example.com"), false),
    "full",
    Some("sso"),
    SsoLink::Unchanged,
    false
)]
#[case::by_id_under_links_any_user_leaves_its_own_id_unlinked(
    sign_in("created", Some("created"), None, true),
    "created",
    None,
    SsoLink::Unchanged,
    false
)]
#[case::by_sso_id(
    sign_in("new", Some("sso"), None, false),
    "full",
    Some("sso"),
    SsoLink::Unchanged,
    false
)]
#[case::by_email_links_the_sso_id_in_line(
    sign_in("new", Some("sso-new"), Some("Emailed@Example.com"), false),
    "emailed",
    Some("sso-new"),
    SsoLink::Linked,
    false
)]
#[case::by_email_under_links_any_user_links_in_the_background(
    sign_in("new", Some("sso-new"), Some("full@example.com"), true),
    "full",
    Some("sso"),
    SsoLink::Pending,
    false
)]
#[tokio::test(flavor = "multi_thread")]
async fn a_sign_in_reads_the_user_it_resolves_to_with_what_it_linked_and_owes(
    #[future(awt)] users: Users,
    #[case] signing_in: SignIn,
    #[case] user_id: &str,
    #[case] sso_user_id: Option<&str>,
    #[case] sso_link: SsoLink,
    #[case] owes_email: bool,
) {
    execute(
        users.postgres.pool(),
        r#"INSERT INTO "LiteLLM_UserTable" (user_id, user_email, created_at, updated_at)
           VALUES ('emailed', 'emailed@example.com', '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
    )
    .await;

    let SignedIn {
        user,
        sso_link: linked,
        owes_email: owed,
    } = users
        .database
        .sign_in(&signing_in, Side::Read)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(
        (
            user.user_id.as_str(),
            user.sso_user_id.as_deref(),
            linked,
            owed
        ),
        (user_id, sso_user_id, sso_link, owes_email)
    );
}

#[rstest]
#[case::by_id_alone(sign_in("gone", None, None, false))]
#[case::by_an_sso_id_and_an_email_no_row_holds(sign_in(
    "gone",
    Some("sso-gone"),
    Some("gone@example.com"),
    false
))]
#[tokio::test(flavor = "multi_thread")]
async fn a_sign_in_no_row_answers_is_not_found(
    #[future(awt)] users: Users,
    #[case] signing_in: SignIn,
) {
    let read = users
        .database
        .sign_in(&signing_in, Side::Read)
        .await
        .unwrap();

    assert!(read.is_none());
}
