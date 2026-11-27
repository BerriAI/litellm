//! A team membership's database read against a Prisma-migrated Postgres. Each row literal is what
//! the repository hands the Python proxy, and each entry literal is what Python caches for that
//! row. `tests/unit/redis/stores/test_auth_cache_team_membership.py` checks both against Python's
//! own model and membership readers, and `redis-store/tests/auth_cache_team_membership.rs`
//! round-trips the entries through Redis. Change all three together.

mod support;

use std::sync::Arc;

use litellm_auth_store::{Error, MemberAdded, MemberRemoved, TeamMembershipDatabase};
use litellm_db_testing::MigratedPostgres;
use litellm_repositories::{
    JoinBudget, MappedClaim, MemberBudget, NewCappedMemberBudget, RosterRole, Side,
    TeamMembershipRepository,
};
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

/// `roster` lists `other` and then `leaver`, whose row lists `roster` and `elsewhere` among its
/// teams and who holds a key in each, the one in `roster` mapped to a JWT claim. `stale` lists
/// `roster` among its teams with no roster entry, and `joiner` is in no team.
async fn insert_roster(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles, created_at, updated_at)
           VALUES ('roster', '[{"user_id": "other", "user_email": null, "role": "admin"},
                               {"user_id": "leaver", "user_email": "leaver@example.com", "role": "user"}]',
                   '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                  ('elsewhere', '[]', '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles, default_team_member_models, created_at, updated_at)
           VALUES ('dt-capped', '[]', ARRAY['claude'], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                  ('dt-plain', '[]', ARRAY[]::text[], '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_UserTable" (user_id, user_email, teams, created_at, updated_at)
           VALUES ('leaver', 'leaver@example.com', ARRAY['roster', 'elsewhere'], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                  ('stale', NULL, ARRAY['roster'], '2026-11-01 00:00:00', '2026-11-01 00:00:00'),
                  ('joiner', 'joiner@example.com', ARRAY[]::text[], '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id) VALUES ('leaver', 'roster'), ('leaver', 'elsewhere')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_VerificationToken"
               (token, user_id, team_id, key_alias, spend, allowed_cache_controls, allowed_routes, policies, access_group_ids)
           VALUES ('leaver-roster-key', 'leaver', 'roster', 'kept alias', 1.5, NULL, NULL, NULL, NULL),
                  ('leaver-elsewhere-key', 'leaver', 'elsewhere', NULL, 0, '{}', '{}', '{}', '{}')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_JWTKeyMapping" (id, jwt_claim_name, jwt_claim_value, token, jwt_issuer)
           VALUES ('mapping', 'sub', 'leaver-sub', 'leaver-roster-key', 'https://issuer.example.com')"#,
    )
    .await;
}

#[fixture]
async fn rosters(#[future(awt)] memberships: Memberships) -> Memberships {
    insert_roster(memberships.postgres.pool()).await;
    memberships
}

fn user_ids(roster: &[litellm_redis_store::auth_cache::MemberEntry]) -> Vec<Option<&str>> {
    roster
        .iter()
        .map(|member| member.user_id.as_deref())
        .collect()
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_member_add_returns_the_roster_as_stored_with_the_new_member_last(
    #[future(awt)] rosters: Memberships,
) {
    let added = rosters
        .database
        .add_member("joiner", "roster", MemberBudget::None)
        .await
        .unwrap();
    let membership = rosters
        .database
        .team_membership("joiner", "roster", Side::Writer)
        .await
        .unwrap();

    let MemberAdded::Added { roster } = added else {
        panic!("the user was added: {added:?}");
    };
    assert_eq!(
        (
            dumped(roster.last().cloned().unwrap()).as_str(),
            user_ids(&roster),
            membership.map(|membership| membership.team_id)
        ),
        (
            r#"{"user_id": "joiner", "user_email": "joiner@example.com", "role": "user"}"#,
            vec![Some("other"), Some("leaver"), Some("joiner")],
            Some("roster".to_owned())
        )
    );
}

#[rstest]
#[case::already_on_the_roster("leaver", "roster", MemberAdded::AlreadyMember)]
#[case::a_team_no_row_has("joiner", "gone", MemberAdded::TeamMissing)]
#[case::a_user_no_row_has("gone", "roster", MemberAdded::UserMissing)]
#[tokio::test(flavor = "multi_thread")]
async fn a_member_add_that_writes_nothing_says_why(
    #[future(awt)] rosters: Memberships,
    #[case] user_id: &str,
    #[case] team_id: &str,
    #[case] expected: MemberAdded,
) {
    let added = rosters
        .database
        .add_member(user_id, team_id, MemberBudget::None)
        .await
        .unwrap();

    assert_eq!(added, expected);
}

/// A removal returns what the host evicts and audits: the rosters before and after as `Member`
/// dumps them, the hashes of the member's deleted keys and the JWT claims mapped to them.
#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_member_removal_takes_the_user_off_the_team_with_its_keys_in_the_team(
    #[future(awt)] rosters: Memberships,
) {
    let removed = rosters
        .database
        .remove_member("leaver", "roster")
        .await
        .unwrap();

    let MemberRemoved::Removed {
        roster_before,
        roster_after,
        deleted_tokens,
        mapped_claims,
    } = removed
    else {
        panic!("the user was removed: {removed:?}");
    };
    assert_eq!(
        (
            user_ids(&roster_before),
            user_ids(&roster_after),
            deleted_tokens,
            mapped_claims,
        ),
        (
            vec![Some("other"), Some("leaver")],
            vec![Some("other")],
            vec!["leaver-roster-key".to_owned()],
            vec![MappedClaim {
                jwt_issuer: "https://issuer.example.com".to_owned(),
                jwt_claim_name: "sub".to_owned(),
                jwt_claim_value: "leaver-sub".to_owned(),
            }],
        )
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_member_removal_of_a_team_only_the_users_row_lists_leaves_the_roster_as_it_was(
    #[future(awt)] rosters: Memberships,
) {
    let removed = rosters
        .database
        .remove_member("stale", "roster")
        .await
        .unwrap();

    let MemberRemoved::Removed {
        roster_before,
        roster_after,
        ..
    } = removed
    else {
        panic!("the stale team was cleared: {removed:?}");
    };
    assert_eq!(
        (user_ids(&roster_after), roster_before == roster_after),
        (vec![Some("other"), Some("leaver")], true)
    );
}

#[rstest]
#[case::a_user_the_team_does_not_list("joiner", "roster", MemberRemoved::NotMember)]
#[case::a_team_no_row_has("leaver", "gone", MemberRemoved::TeamMissing)]
#[tokio::test(flavor = "multi_thread")]
async fn a_member_removal_that_writes_nothing_says_why(
    #[future(awt)] rosters: Memberships,
    #[case] user_id: &str,
    #[case] team_id: &str,
    #[case] expected: MemberRemoved,
) {
    let removed = rosters
        .database
        .remove_member(user_id, team_id)
        .await
        .unwrap();

    assert_eq!(removed, expected);
}

/// A default team join returns the roster with the entry's role, as `Member` dumps it.
#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn a_default_team_join_returns_the_roster_with_the_role_it_gave(
    #[future(awt)] rosters: Memberships,
) {
    let capped = NewCappedMemberBudget {
        budget_id: "joiner-capped".to_owned(),
        max_budget: 3.0,
        allowed_models: Some(vec!["claude".to_owned()]),
        created_by: "default_user_id".to_owned(),
        updated_by: "default_user_id".to_owned(),
    };

    let added = rosters
        .database
        .add_member_as(
            "joiner",
            "dt-capped",
            RosterRole::Admin,
            JoinBudget::Capped(&capped),
        )
        .await
        .unwrap();

    let MemberAdded::Added { roster } = added else {
        panic!("the user was added: {added:?}");
    };
    assert_eq!(
        roster.into_iter().map(dumped).collect::<Vec<_>>(),
        vec![
            r#"{"user_id": "joiner", "user_email": "joiner@example.com", "role": "admin"}"#
                .to_owned()
        ]
    );
}

/// `broken` lists `leaver` with no role, which `Member` refuses, and `leaver` holds a key in it.
async fn insert_broken_roster(pool: &PgPool) {
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles, created_at, updated_at)
           VALUES ('broken', '[{"user_id": "leaver"}]', '2026-11-01 00:00:00', '2026-11-01 00:00:00')"#,
    )
    .await;
    execute(
        pool,
        r#"INSERT INTO "LiteLLM_VerificationToken" (token, user_id, team_id) VALUES ('leaver-broken-key', 'leaver', 'broken')"#,
    )
    .await;
}

#[rstest]
#[case::an_add("joiner", true)]
#[case::a_removal("leaver", false)]
#[tokio::test(flavor = "multi_thread")]
async fn a_roster_the_model_refuses_fails_the_write_as_the_column_it_refuses(
    #[future(awt)] rosters: Memberships,
    #[case] user_id: &str,
    #[case] adding: bool,
) {
    let pool = rosters.postgres.pool();
    insert_broken_roster(pool).await;

    let refused = match adding {
        true => rosters
            .database
            .add_member(user_id, "broken", MemberBudget::None)
            .await
            .err(),
        false => rosters
            .database
            .remove_member(user_id, "broken")
            .await
            .err(),
    };

    assert!(
        matches!(
            &refused,
            Some(Error::Row {
                column: "role",
                within: Some("members_with_roles")
            })
        ),
        "{refused:?}"
    );
}
