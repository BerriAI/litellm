//! `LiteLLM_UserTable` reads and the conditional writes the user repository makes while resolving a sign-in

use serde::{Deserialize, Serialize};
use sqlx::{
    PgExecutor,
    types::{Json, chrono::NaiveDateTime},
};

use litellm_db::Error;

use super::repository::NewUserDefaults;
use crate::{RawJson, utc_timestamp};

/// One `LiteLLM_UserTable` row without its password hash, with its organization memberships
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Serialize)]
pub struct User {
    pub(super) user_id: String,
    pub(super) user_alias: Option<String>,
    pub(super) team_id: Option<String>,
    pub(super) sso_user_id: Option<String>,
    pub(super) organization_id: Option<String>,
    pub(super) object_permission_id: Option<String>,
    pub(super) password_reset_required: Option<bool>,
    #[serde(with = "utc_timestamp::option")]
    pub(super) last_breach_check_at: Option<NaiveDateTime>,
    pub(super) teams: Vec<String>,
    pub(super) user_role: Option<String>,
    pub(super) max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) rollover_max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) rollover_credit: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) temp_budget_increase: Option<f64>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "utc_timestamp::option"
    )]
    pub(super) temp_budget_expiry: Option<NaiveDateTime>,
    pub(super) budget_limits: Option<RawJson>,
    pub(super) spend: f64,
    pub(super) user_email: Option<String>,
    pub(super) models: Vec<String>,
    pub(super) metadata: RawJson,
    pub(super) max_parallel_requests: Option<i32>,
    pub(super) tpm_limit: Option<i64>,
    pub(super) rpm_limit: Option<i64>,
    pub(super) budget_duration: Option<String>,
    #[serde(with = "utc_timestamp::option")]
    pub(super) budget_reset_at: Option<NaiveDateTime>,
    pub(super) allowed_cache_controls: Vec<String>,
    pub(super) policies: Vec<String>,
    pub(super) model_spend: RawJson,
    pub(super) model_max_budget: RawJson,
    #[serde(with = "utc_timestamp::option")]
    pub(super) created_at: Option<NaiveDateTime>,
    #[serde(with = "utc_timestamp::option")]
    pub(super) updated_at: Option<NaiveDateTime>,
    pub(super) organization_memberships: Json<Vec<OrganizationMembership>>,
}

/// One `LiteLLM_OrganizationMembership` row, as a user row includes it
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Deserialize, Serialize)]
pub(super) struct OrganizationMembership {
    pub user_id: String,
    pub organization_id: String,
    pub user_role: Option<String>,
    pub spend: Option<f64>,
    pub budget_id: Option<String>,
    #[serde(serialize_with = "utc_timestamp::option::serialize")]
    pub created_at: Option<NaiveDateTime>,
    #[serde(serialize_with = "utc_timestamp::option::serialize")]
    pub updated_at: Option<NaiveDateTime>,
}

/// The user whose id is `user_id`, else the one whose SSO id is `sso_user_id`
pub(super) async fn find_user(
    executor: impl PgExecutor<'_>,
    user_id: Option<&str>,
    sso_user_id: Option<&str>,
) -> Result<Option<User>, Error> {
    Ok(
        sqlx::query_file_as!(User, "src/user/queries/find_user.sql", user_id, sso_user_id)
            .fetch_optional(executor)
            .await?,
    )
}

/// A user whose email matches `user_email` ignoring case, the lowest id first; with `unlinked_only`, only one no
/// sign-in has linked yet
pub(super) async fn find_user_by_email(
    executor: impl PgExecutor<'_>,
    user_email: &str,
    unlinked_only: bool,
) -> Result<Option<User>, Error> {
    Ok(sqlx::query_file_as!(
        User,
        "src/user/queries/find_user_by_email.sql",
        user_email,
        unlinked_only
    )
    .fetch_optional(executor)
    .await?)
}

/// Whether a user other than `user_id` holds `user_email`, ignoring case
pub(super) async fn email_belongs_to_another_user(
    executor: impl PgExecutor<'_>,
    user_email: &str,
    user_id: &str,
) -> Result<bool, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/user/queries/email_belongs_to_another_user.sql",
        user_email,
        user_id
    )
    .fetch_one(executor)
    .await?)
}

/// Links `sso_user_id` to `user_id` only while the row has none; false when it already had one, the row is
/// missing, or another row holds that sign-in id
pub(super) async fn link_null_sso_user_id(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    sso_user_id: &str,
) -> Result<bool, Error> {
    let result = sqlx::query_file!(
        "src/user/queries/link_null_sso_user_id.sql",
        user_id,
        sso_user_id
    )
    .execute(executor)
    .await;
    match result {
        Ok(done) => Ok(done.rows_affected() == 1),
        Err(sqlx::Error::Database(error)) if error.is_unique_violation() => Ok(false),
        Err(error) => Err(error.into()),
    }
}

pub(super) async fn set_sso_user_id(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    sso_user_id: &str,
) -> Result<(), Error> {
    sqlx::query_file!("src/user/queries/set_sso_user_id.sql", user_id, sso_user_id)
        .execute(executor)
        .await?;
    Ok(())
}

/// Sets `user_email` on `user_id` while it has none, unless another user holds it and `links_any_user` is off.
/// The email now stored, or `None` when another user holds it or the row is gone
pub(super) async fn backfill_user_email(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    user_email: &str,
    links_any_user: bool,
) -> Result<Option<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/user/queries/backfill_user_email.sql",
        user_id,
        user_email,
        links_any_user
    )
    .fetch_optional(executor)
    .await?)
}

/// A user to create. `email_may_be_shared` stores the email even when another user holds it
pub(super) struct NewUser<'a> {
    pub user_id: &'a str,
    pub user_email: Option<&'a str>,
    pub email_may_be_shared: bool,
    pub sso_user_id: Option<&'a str>,
    pub defaults: &'a NewUserDefaults,
}

/// Sets the role of `user_id`; the row as stored, or `None` when no row has that id
pub(super) async fn set_user_role(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    user_role: &str,
) -> Result<Option<User>, Error> {
    Ok(sqlx::query_file_as!(
        User,
        "src/user/queries/set_user_role.sql",
        user_id,
        user_role
    )
    .fetch_optional(executor)
    .await?)
}

/// Inserts `new_user`; the row as stored, or `None` when its id or SSO id is already taken
pub(super) async fn insert_user(
    executor: impl PgExecutor<'_>,
    new_user: &NewUser<'_>,
) -> Result<Option<User>, Error> {
    let defaults = new_user.defaults;
    Ok(sqlx::query_file_as!(
        User,
        "src/user/queries/insert_user.sql",
        new_user.user_id,
        new_user.user_email,
        new_user.email_may_be_shared,
        new_user.sso_user_id,
        defaults.user_role.as_deref(),
        defaults.max_budget,
        defaults.budget_duration.as_deref(),
        defaults.budget_reset_at.and_then(utc_timestamp::naive_utc),
        defaults.models.as_deref(),
        defaults.budget_limits.as_ref(),
        defaults.tpm_limit,
        defaults.rpm_limit,
        defaults.max_parallel_requests,
        defaults.user_alias.as_deref(),
        defaults.team_id.as_deref(),
        defaults.organization_id.as_deref(),
        defaults.object_permission_id.as_deref(),
        defaults.password_reset_required,
        defaults.metadata.as_ref(),
        defaults.allowed_cache_controls.as_deref(),
        defaults.policies.as_deref(),
        defaults.model_max_budget.as_ref()
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use litellm_db::Statements;
    use sqlx::PgConnection;

    use crate::Inserted;
    use crate::Side;
    use crate::database::Connections;
    use crate::test_support::{OneTransaction, database_pool, insert_budget, leave_utc};
    use crate::user::repository::{NewUserDefaults, SignIn, insert_once, resolve_once};

    async fn insert_user(
        tx: &mut Transaction<'_, Postgres>,
        user_id: &str,
        user_email: Option<&str>,
        sso_user_id: Option<&str>,
    ) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_UserTable"
                (user_id, user_email, sso_user_id, password, models, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, 'hash', ARRAY['gpt-4o'], '{"team": "a"}', '2026-11-01 00:00:00',
                    '2026-11-01 00:00:00')
            "#,
        )
        .bind(user_id)
        .bind(user_email)
        .bind(sso_user_id)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    async fn insert_membership(
        tx: &mut Transaction<'_, Postgres>,
        user_id: &str,
    ) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            WITH budget AS (
                INSERT INTO "LiteLLM_BudgetTable" (budget_id, created_by, updated_by)
                VALUES ('user-repository-org-budget', 'db-query-test', 'db-query-test')
            ), organization AS (
                INSERT INTO "LiteLLM_OrganizationTable"
                    (organization_id, organization_alias, budget_id, created_by, updated_by)
                VALUES ('user-repository-org', 'org', 'user-repository-org-budget', 'db-query-test',
                        'db-query-test')
            )
            INSERT INTO "LiteLLM_OrganizationMembership"
                (user_id, organization_id, user_role, created_at, updated_at)
            VALUES ($1, 'user-repository-org', 'org_admin', '2026-11-03 00:00:00',
                    '2026-11-03 00:00:00')
            "#,
        )
        .bind(user_id)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    fn user_ids(found: Option<super::User>) -> Option<String> {
        found.map(|user| user.user_id)
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn user_by_id_carries_memberships_omits_password_and_marks_timestamps_utc(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(
            &mut tx,
            "user-repository-by-id",
            Some("a@example.com"),
            None,
        )
        .await?;
        insert_membership(&mut tx, "user-repository-by-id").await?;

        let user = super::find_user(&mut *tx, Some("user-repository-by-id"), None)
            .await?
            .ok_or("user row missing")?;

        assert_eq!(
            serde_json::to_value(user)?,
            serde_json::json!({
                "user_id": "user-repository-by-id",
                "user_alias": null,
                "team_id": null,
                "sso_user_id": null,
                "organization_id": null,
                "object_permission_id": null,
                "password_reset_required": null,
                "last_breach_check_at": null,
                "teams": [],
                "user_role": null,
                "max_budget": null,
                "budget_limits": null,
                "spend": 0.0,
                "user_email": "a@example.com",
                "models": ["gpt-4o"],
                "metadata": {"team": "a"},
                "max_parallel_requests": null,
                "tpm_limit": null,
                "rpm_limit": null,
                "budget_duration": null,
                "budget_reset_at": null,
                "allowed_cache_controls": [],
                "policies": [],
                "model_spend": {},
                "model_max_budget": {},
                "created_at": "2026-11-01T00:00:00+00:00",
                "updated_at": "2026-11-01T00:00:00+00:00",
                "organization_memberships": [{
                    "user_id": "user-repository-by-id",
                    "organization_id": "user-repository-org",
                    "user_role": "org_admin",
                    "spend": 0.0,
                    "budget_id": null,
                    "created_at": "2026-11-03T00:00:00+00:00",
                    "updated_at": "2026-11-03T00:00:00+00:00",
                }],
            })
        );
        Ok(())
    }

    #[rstest]
    #[case::id_wins_over_sso_id(
        Some("user-repository-a"),
        Some("sso-b"),
        Some("user-repository-a")
    )]
    #[case::sso_id_when_no_id_matches(
        Some("user-repository-missing"),
        Some("sso-b"),
        Some("user-repository-b")
    )]
    #[case::sso_id_alone(None, Some("sso-b"), Some("user-repository-b"))]
    #[case::neither(Some("user-repository-missing"), Some("sso-missing"), None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn user_by_id_or_sso_id(
        #[case] user_id: Option<&str>,
        #[case] sso_user_id: Option<&str>,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(&mut tx, "user-repository-a", None, None).await?;
        insert_user(&mut tx, "user-repository-b", None, Some("sso-b")).await?;

        let found = super::find_user(&mut *tx, user_id, sso_user_id).await?;

        assert_eq!(user_ids(found).as_deref(), expected);
        Ok(())
    }

    #[rstest]
    #[case::exact("Mixed@Example.com", false, Some("user-repository-email"))]
    #[case::ignores_case("mixed@EXAMPLE.com", false, Some("user-repository-email"))]
    #[case::underscore_is_literal("mixed@example.co_", false, None)]
    #[case::percent_is_literal("mixed@example.%", false, None)]
    #[case::backslash_is_literal("mixed@example\\.com", false, None)]
    #[case::unlinked_only_skips_linked("mixed@example.com", true, None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn user_by_email(
        #[case] lookup: &str,
        #[case] unlinked_only: bool,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(
            &mut tx,
            "user-repository-email",
            Some("Mixed@Example.com"),
            Some("sso-linked"),
        )
        .await?;

        let found = super::find_user_by_email(&mut *tx, lookup, unlinked_only).await?;

        assert_eq!(user_ids(found).as_deref(), expected);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn user_by_email_takes_the_lowest_id_of_a_shared_email(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(
            &mut tx,
            "user-repository-shared-b",
            Some("Shared@Example.com"),
            None,
        )
        .await?;
        insert_user(
            &mut tx,
            "user-repository-shared-a",
            Some("shared@example.com"),
            None,
        )
        .await?;

        let found = super::find_user_by_email(&mut *tx, "SHARED@example.com", false).await?;

        assert_eq!(user_ids(found).as_deref(), Some("user-repository-shared-a"));
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn sso_link_only_fills_an_empty_slot(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(&mut tx, "user-repository-link-a", None, None).await?;
        insert_user(&mut tx, "user-repository-link-b", None, Some("sso-held")).await?;

        let linked =
            super::link_null_sso_user_id(&mut *tx, "user-repository-link-a", "sso-new").await?;
        let relinked =
            super::link_null_sso_user_id(&mut *tx, "user-repository-link-a", "sso-other").await?;
        let found = super::find_user(&mut *tx, None, Some("sso-new")).await?;

        assert_eq!(
            (linked, relinked, user_ids(found)),
            (true, false, Some("user-repository-link-a".to_owned()))
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn sso_link_to_a_held_id_is_refused(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(&mut tx, "user-repository-held-a", None, None).await?;
        insert_user(&mut tx, "user-repository-held-b", None, Some("sso-held")).await?;

        assert!(
            !super::link_null_sso_user_id(&mut *tx, "user-repository-held-a", "sso-held").await?
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn email_ownership_ignores_case_and_the_user_itself(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(
            &mut tx,
            "user-repository-owner",
            Some("Owner@Example.com"),
            None,
        )
        .await?;

        let taken = super::email_belongs_to_another_user(
            &mut *tx,
            "owner@example.com",
            "user-repository-other",
        )
        .await?;
        let own = super::email_belongs_to_another_user(
            &mut *tx,
            "owner@example.com",
            "user-repository-owner",
        )
        .await?;

        assert_eq!((taken, own), (true, false));
        Ok(())
    }

    #[rstest]
    #[case::fills_a_null_email(
        "user-repository-null",
        "new@example.com",
        false,
        Some("new@example.com")
    )]
    #[case::keeps_a_stored_email(
        "user-repository-set",
        "new@example.com",
        false,
        Some("set@example.com")
    )]
    #[case::refuses_an_email_another_user_holds(
        "user-repository-null",
        "Owner@example.com",
        false,
        None
    )]
    #[case::returns_nothing_for_a_stored_email_another_user_holds(
        "user-repository-set",
        "owner@example.com",
        false,
        None
    )]
    #[case::links_any_user_takes_a_held_email(
        "user-repository-null",
        "owner@example.com",
        true,
        Some("owner@example.com")
    )]
    #[case::missing_row("user-repository-missing", "new@example.com", false, None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn email_backfill(
        #[case] user_id: &str,
        #[case] user_email: &str,
        #[case] links_any_user: bool,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(&mut tx, "user-repository-null", None, None).await?;
        insert_user(
            &mut tx,
            "user-repository-set",
            Some("set@example.com"),
            None,
        )
        .await?;
        insert_user(
            &mut tx,
            "user-repository-owner",
            Some("owner@example.com"),
            None,
        )
        .await?;

        let stored =
            super::backfill_user_email(&mut *tx, user_id, user_email, links_any_user).await?;

        assert_eq!(stored.as_deref(), expected);
        Ok(())
    }

    fn sign_in(
        user_id: &str,
        sso_user_id: Option<&str>,
        user_email: Option<&str>,
        email_verified: Option<bool>,
        links_any_user: bool,
    ) -> SignIn {
        SignIn {
            user_id: user_id.to_owned(),
            sso_user_id: sso_user_id.map(str::to_owned),
            user_email: user_email.map(str::to_owned),
            email_verified,
            links_any_user,
        }
    }

    /// Rows: `own` has no SSO id and no email, `linked` holds `sso-linked`, `unlinked` has `shared@example.com`
    /// and no SSO id, `taken` has `taken@example.com` and holds `sso-taken`
    async fn insert_sign_in_rows(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        insert_user(tx, "own", None, None).await?;
        insert_user(tx, "linked", None, Some("sso-linked")).await?;
        insert_user(tx, "unlinked", Some("shared@example.com"), None).await?;
        insert_user(tx, "taken", Some("taken@example.com"), Some("sso-taken")).await
    }

    #[rstest]
    #[case::by_id_links_its_own_id(
        sign_in("own", Some("own"), None, None, false),
        Some("own"),
        Some("own"),
        None
    )]
    #[case::by_id_leaves_a_foreign_sso_id(
        sign_in("own", Some("sso-x"), None, None, false),
        Some("own"),
        None,
        None
    )]
    #[case::by_id_under_links_any_user_does_not_link(
        sign_in("own", Some("own"), None, None, true),
        Some("own"),
        None,
        None
    )]
    #[case::by_sso_id(
        sign_in("new", Some("sso-linked"), Some("shared@example.com"), None, false),
        Some("linked"),
        Some("sso-linked"),
        None
    )]
    #[case::by_email_links_the_sso_id(
        sign_in("new", Some("sso-new"), Some("Shared@Example.com"), Some(true), false),
        Some("unlinked"),
        Some("sso-new"),
        None
    )]
    #[case::unverified_email_is_skipped(
        sign_in("new", Some("sso-new"), Some("shared@example.com"), Some(false), false),
        None,
        None,
        None
    )]
    #[case::unverified_email_under_links_any_user(
        sign_in("new", None, Some("shared@example.com"), Some(false), true),
        Some("unlinked"),
        None,
        None
    )]
    #[case::linked_email_is_skipped(
        sign_in("new", Some("sso-new"), Some("taken@example.com"), None, false),
        None,
        None,
        None
    )]
    #[case::email_without_sso_id(
        sign_in("new", None, Some("taken@example.com"), None, false),
        Some("taken"),
        Some("sso-taken"),
        None
    )]
    #[case::links_any_user_links_later(sign_in("new", Some("sso-new"), Some("taken@example.com"), None, true), Some("taken"), Some("sso-taken"), Some(("taken", "sso-new")))]
    #[case::no_email(sign_in("new", Some("sso-new"), None, None, false), None, None, None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn resolve(
        #[case] sign_in: SignIn,
        #[case] expected: Option<&str>,
        #[case] stored_sso_user_id: Option<&str>,
        #[case] link_later: Option<(&str, &str)>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_sign_in_rows(&mut tx).await?;

        let resolved = resolve_once(&mut OneTransaction(&mut tx), &sign_in, Side::Read).await?;
        let stored = match expected {
            Some(user_id) => super::find_user(&mut *tx, Some(user_id), None)
                .await?
                .and_then(|user| user.sso_user_id),
            None => None,
        };

        assert_eq!(
            (user_ids(resolved.user), resolved.link_later, stored),
            (
                expected.map(str::to_owned),
                link_later.map(|(user_id, sso)| (user_id.to_owned(), sso.to_owned())),
                stored_sso_user_id.map(str::to_owned),
            )
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_sign_in_that_loses_the_link_race_gets_the_user_that_won(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(&mut tx, "race-a", Some("race@example.com"), None).await?;
        insert_user(&mut tx, "race-b", None, None).await?;
        let sign_in = sign_in(
            "new",
            Some("sso-race"),
            Some("race@example.com"),
            None,
            false,
        );

        let mut racing = RaceAfterEmailRead {
            tx: &mut tx,
            winner: "race-b",
            sso_user_id: "sso-race",
            reads: 0,
        };
        let resolved = resolve_once(&mut racing, &sign_in, Side::Read).await?;

        assert_eq!(user_ids(resolved.user).as_deref(), Some("race-b"));
        Ok(())
    }

    /// One transaction for every step that, once the email lookup has read, links `sso_user_id` to `winner`
    /// before the resolve tries to link it, the way a concurrent sign-in would
    struct RaceAfterEmailRead<'t, 'c> {
        tx: &'t mut Transaction<'c, Postgres>,
        winner: &'static str,
        sso_user_id: &'static str,
        reads: usize,
    }

    impl Connections for RaceAfterEmailRead<'_, '_> {
        async fn with<T>(
            &mut self,
            side: Side,
            step: impl for<'e> AsyncFnOnce(
                Statements<&'e mut PgConnection>,
            ) -> Result<T, litellm_db::Error>,
        ) -> Result<T, litellm_db::Error> {
            if side == Side::Writer && self.reads == 2 {
                super::link_null_sso_user_id(&mut **self.tx, self.winner, self.sso_user_id).await?;
            }
            self.reads += 1;
            // The losing link fails on the unique SSO id, which aborts a transaction unless a savepoint holds it
            let mut savepoint = sqlx::Connection::begin(&mut **self.tx).await?;
            let result = step(Statements::new(&mut *savepoint, false)).await;
            savepoint.rollback().await?;
            result
        }
    }

    /// How many seconds the row's `created_at` and `updated_at` are from now in UTC
    async fn seconds_from_utc_now(
        tx: &mut Transaction<'_, Postgres>,
        user_id: &str,
    ) -> Result<(f64, f64), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        sqlx::query_as(
            r#"
            SELECT abs(extract(epoch FROM created_at - (now() AT TIME ZONE 'UTC')))::float8,
                   abs(extract(epoch FROM updated_at - (now() AT TIME ZONE 'UTC')))::float8
            FROM "LiteLLM_UserTable"
            WHERE user_id = $1
            "#,
        )
        .bind(user_id)
        .fetch_one(&mut **tx)
        .await
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn set_role_returns_the_stored_row_and_stamps_it_in_utc(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(
            &mut tx,
            "user-repository-role",
            Some("role@example.com"),
            None,
        )
        .await?;
        insert_user(&mut tx, "user-repository-bystander", None, None).await?;
        insert_membership(&mut tx, "user-repository-role").await?;
        leave_utc(&mut tx).await?;

        let updated = super::set_user_role(&mut *tx, "user-repository-role", "proxy_admin")
            .await?
            .ok_or("no row came back")?;

        let stored = super::find_user(&mut *tx, Some("user-repository-role"), None).await?;
        let bystander = super::find_user(&mut *tx, Some("user-repository-bystander"), None)
            .await?
            .ok_or("bystander row missing")?;
        let (_, updated_at_drift) = seconds_from_utc_now(&mut tx, "user-repository-role").await?;
        assert_eq!(
            (
                updated.user_role.as_deref(),
                updated.organization_memberships.len(),
                bystander.user_role,
            ),
            (Some("proxy_admin"), 1, None)
        );
        assert_eq!(Some(updated), stored);
        assert!(
            updated_at_drift < 1.0,
            "updated_at is {updated_at_drift}s from UTC now"
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn set_role_on_a_missing_user_writes_nothing(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(&mut tx, "user-repository-bystander", None, None).await?;

        let updated =
            super::set_user_role(&mut *tx, "user-repository-missing", "proxy_admin").await?;

        let bystander = super::find_user(&mut *tx, Some("user-repository-bystander"), None)
            .await?
            .ok_or("bystander row missing")?;
        assert_eq!((updated, bystander.user_role), (None, None));
        Ok(())
    }

    fn configured_defaults() -> Result<NewUserDefaults, jiff::Error> {
        Ok(NewUserDefaults {
            user_role: Some("internal_user_viewer".to_owned()),
            max_budget: Some(12.5),
            budget_duration: Some("30d".to_owned()),
            budget_reset_at: Some("2026-12-01T00:00:00.250Z".parse()?),
            models: Some(vec!["gpt-4o".to_owned()]),
            budget_limits: Some(serde_json::json!([{"budget_duration": "1d", "max_budget": 1.0}])),
            tpm_limit: Some(1000),
            rpm_limit: Some(60),
            max_parallel_requests: Some(4),
            user_alias: Some("new hire".to_owned()),
            team_id: Some("team-default".to_owned()),
            organization_id: Some("org-default".to_owned()),
            object_permission_id: Some("permission-default".to_owned()),
            password_reset_required: Some(true),
            metadata: Some(serde_json::json!({"source": "jwt"})),
            allowed_cache_controls: Some(vec!["no-cache".to_owned()]),
            policies: Some(vec!["policy-default".to_owned()]),
            model_max_budget: Some(serde_json::json!({"gpt-4o": 5.0})),
        })
    }

    async fn insert_rows_the_defaults_point_at(
        tx: &mut Transaction<'_, Postgres>,
    ) -> Result<(), sqlx::Error> {
        insert_budget(tx, "budget-default").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            WITH permission AS (
                INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id) VALUES ('permission-default')
            )
            INSERT INTO "LiteLLM_OrganizationTable"
                (organization_id, organization_alias, budget_id, created_by, updated_by)
            VALUES ('org-default', 'default', 'budget-default', 'db-query-test', 'db-query-test')
            "#,
        )
        .execute(&mut **tx)
        .await?;
        Ok(())
    }

    fn without_timestamps(user: &super::User) -> Result<serde_json::Value, serde_json::Error> {
        let mut row = serde_json::to_value(user)?;
        if let Some(columns) = row.as_object_mut() {
            columns.remove("created_at");
            columns.remove("updated_at");
        }
        Ok(row)
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn insert_stores_the_sign_in_and_the_defaults_stamped_in_utc(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        leave_utc(&mut tx).await?;
        insert_rows_the_defaults_point_at(&mut tx).await?;
        let sign_in = sign_in("new", Some("sso-new"), Some("New@Example.com"), None, false);

        let inserted = insert_once(
            &mut OneTransaction(&mut tx),
            &sign_in,
            &configured_defaults()?,
        )
        .await?;

        let Inserted::Created(user) = inserted else {
            return Err(format!("expected a created user, got {inserted:?}").into());
        };
        let (created_at_drift, updated_at_drift) = seconds_from_utc_now(&mut tx, "new").await?;
        assert_eq!(
            without_timestamps(&user)?,
            serde_json::json!({
                "user_id": "new",
                "user_alias": "new hire",
                "team_id": "team-default",
                "sso_user_id": "sso-new",
                "organization_id": "org-default",
                "object_permission_id": "permission-default",
                "password_reset_required": true,
                "last_breach_check_at": null,
                "teams": [],
                "user_role": "internal_user_viewer",
                "max_budget": 12.5,
                "budget_limits": [{"budget_duration": "1d", "max_budget": 1.0}],
                "spend": 0.0,
                "user_email": "New@Example.com",
                "models": ["gpt-4o"],
                "metadata": {"source": "jwt"},
                "max_parallel_requests": 4,
                "tpm_limit": 1000,
                "rpm_limit": 60,
                "budget_duration": "30d",
                "budget_reset_at": "2026-12-01T00:00:00.250+00:00",
                "allowed_cache_controls": ["no-cache"],
                "policies": ["policy-default"],
                "model_spend": {},
                "model_max_budget": {"gpt-4o": 5.0},
                "organization_memberships": [],
            })
        );
        assert_eq!(
            Some(user),
            super::find_user(&mut *tx, Some("new"), None).await?
        );
        assert!(
            created_at_drift < 1.0 && updated_at_drift < 1.0,
            "created_at is {created_at_drift}s and updated_at {updated_at_drift}s from UTC now"
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn insert_without_defaults_leaves_their_columns_unset(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let sign_in = sign_in("bare", None, None, None, false);

        insert_once(
            &mut OneTransaction(&mut tx),
            &sign_in,
            &NewUserDefaults::default(),
        )
        .await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let unset: bool = sqlx::query_scalar(
            r#"
            SELECT num_nulls(user_email, sso_user_id, user_role, max_budget, budget_duration, budget_reset_at,
                             models, budget_limits, tpm_limit, rpm_limit, max_parallel_requests, user_alias,
                             team_id, organization_id, object_permission_id, password_reset_required) = 16
                   AND (metadata, allowed_cache_controls, policies, model_max_budget)
                       = ('{}'::jsonb, '{}'::text[], '{}'::text[], '{}'::jsonb)
            FROM "LiteLLM_UserTable"
            WHERE user_id = 'bare'
            "#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert!(unset);
        Ok(())
    }

    #[rstest]
    #[case::free("fresh@example.com", false, Some("fresh@example.com"))]
    #[case::held_by_another_user_ignoring_case("OWNER@example.com", false, None)]
    #[case::held_but_links_any_user("owner@example.com", true, Some("owner@example.com"))]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn insert_stores_an_email_only_while_no_other_user_holds_it(
        #[case] user_email: &str,
        #[case] links_any_user: bool,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_user(
            &mut tx,
            "user-repository-owner",
            Some("Owner@Example.com"),
            None,
        )
        .await?;
        let sign_in = sign_in("new", None, Some(user_email), None, links_any_user);

        insert_once(
            &mut OneTransaction(&mut tx),
            &sign_in,
            &NewUserDefaults::default(),
        )
        .await?;

        let stored = super::find_user(&mut *tx, Some("new"), None)
            .await?
            .ok_or("user row missing")?;
        assert_eq!(stored.user_email.as_deref(), expected);
        Ok(())
    }

    #[rstest]
    #[case::stored(false, Some("sso-new"))]
    #[case::left_to_the_background_link_under_links_any_user(true, None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn insert_stores_the_sso_id_only_while_links_any_user_is_off(
        #[case] links_any_user: bool,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let sign_in = sign_in("new", Some("sso-new"), None, None, links_any_user);

        insert_once(
            &mut OneTransaction(&mut tx),
            &sign_in,
            &NewUserDefaults::default(),
        )
        .await?;

        let stored = super::find_user(&mut *tx, Some("new"), None)
            .await?
            .ok_or("user row missing")?;
        assert_eq!(stored.sso_user_id.as_deref(), expected);
        Ok(())
    }

    #[rstest]
    #[case::the_id_is_taken(sign_in("own", None, Some("own@example.com"), None, false), "own")]
    #[case::the_sso_id_is_taken(sign_in("new", Some("sso-linked"), None, None, false), "linked")]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn insert_that_conflicts_returns_the_stored_user_and_writes_nothing(
        #[case] sign_in: SignIn,
        #[case] winner: &str,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_sign_in_rows(&mut tx).await?;
        let before = super::find_user(&mut *tx, Some(winner), None).await?;

        let inserted = insert_once(
            &mut OneTransaction(&mut tx),
            &sign_in,
            &configured_defaults()?,
        )
        .await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let users: i64 = sqlx::query_scalar(
            r#"SELECT count(*) FROM "LiteLLM_UserTable" WHERE user_id IN ('own', 'new')"#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(
            (inserted, user_ids(before.clone()).as_deref(), users),
            (Inserted::Conflict(before), Some(winner), 1)
        );
        Ok(())
    }
}
