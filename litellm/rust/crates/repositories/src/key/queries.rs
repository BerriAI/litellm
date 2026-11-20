//! Virtual key reads: the `combined_view` row (the key joined with its team, membership, budgets, project and
//! organization) and the rotation grace-period lookup in `LiteLLM_DeprecatedVerificationToken`.

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sqlx::PgExecutor;

use litellm_db::Error;

use crate::{RawJson, utc_timestamp};

/// The key as the auth flow sees it: the key's own columns, with its budget row filling the limits the key
/// leaves unset, and the team, membership, project and organization values it is checked against. It
/// also reads back from the JSON Python caches it as, where a field it does not name is refused.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyView {
    pub token: String,
    pub key_name: Option<String>,
    pub key_alias: Option<String>,
    pub soft_budget_cooldown: bool,
    pub spend: f64,
    pub total_spend: f64,
    #[serde(with = "utc_timestamp::option")]
    pub expires: Option<NaiveDateTime>,
    pub models: Vec<String>,
    pub aliases: RawJson,
    pub config: RawJson,
    pub router_settings: Option<RawJson>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub agent_id: Option<String>,
    pub project_id: Option<String>,
    pub permissions: RawJson,
    pub max_parallel_requests: Option<i32>,
    pub metadata: RawJson,
    pub blocked: Option<bool>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub tpd_limit: Option<i64>,
    pub max_budget: Option<f64>,
    pub soft_budget: Option<f64>,
    pub budget_duration: Option<String>,
    #[serde(with = "utc_timestamp::option")]
    pub budget_reset_at: Option<NaiveDateTime>,
    pub allowed_cache_controls: Option<Vec<String>>,
    pub allowed_routes: Option<Vec<String>>,
    pub key_type: Option<String>,
    pub access_group_ids: Option<Vec<String>>,
    pub model_spend: RawJson,
    pub model_max_budget: RawJson,
    pub budget_fallbacks: RawJson,
    pub budget_id: Option<String>,
    pub org_id: Option<String>,
    pub object_permission_id: Option<String>,
    #[serde(with = "utc_timestamp::option")]
    pub created_at: Option<NaiveDateTime>,
    pub created_by: Option<String>,
    #[serde(with = "utc_timestamp::option")]
    pub updated_at: Option<NaiveDateTime>,
    pub updated_by: Option<String>,
    #[serde(with = "utc_timestamp::option")]
    pub settings_updated_at: Option<NaiveDateTime>,
    #[serde(with = "utc_timestamp::option")]
    pub last_active: Option<NaiveDateTime>,
    pub rotation_count: Option<i32>,
    pub auto_rotate: Option<bool>,
    pub rotation_interval: Option<String>,
    #[serde(with = "utc_timestamp::option")]
    pub last_rotation_at: Option<NaiveDateTime>,
    #[serde(with = "utc_timestamp::option")]
    pub key_rotation_at: Option<NaiveDateTime>,
    pub budget_limits: Option<RawJson>,
    pub team_spend: Option<f64>,
    pub team_alias: Option<String>,
    pub team_tpm_limit: Option<i64>,
    pub team_rpm_limit: Option<i64>,
    pub team_tpd_limit: Option<i64>,
    pub team_max_budget: Option<f64>,
    pub team_soft_budget: Option<f64>,
    pub team_model_max_budget: Option<RawJson>,
    pub team_models: Vec<String>,
    pub team_blocked: bool,
    pub team_model_aliases: Option<RawJson>,
    pub team_member: Option<RawJson>,
    pub team_metadata: Option<RawJson>,
    pub team_object_permission_id: Option<String>,
    pub team_member_spend: Option<f64>,
    pub team_member_tpm_limit: Option<i64>,
    pub team_member_rpm_limit: Option<i64>,
    pub organization_alias: Option<String>,
    pub organization_max_budget: Option<f64>,
    pub organization_tpm_limit: Option<i64>,
    pub organization_rpm_limit: Option<i64>,
    pub organization_metadata: Option<RawJson>,
    pub project_alias: Option<String>,
}

/// The key a rotated-away token still stands for, and when that grace period ends.
#[macro_rules_attribute::apply(crate::db_row)]
pub(super) struct DeprecatedKey {
    pub active_token_id: String,
    pub revoke_at: NaiveDateTime,
}

/// The [`KeyView`] of the key whose hash is `token`, or `None` when no such key exists.
pub(super) async fn find_key_view(
    executor: impl PgExecutor<'_>,
    token: &str,
) -> Result<Option<KeyView>, Error> {
    Ok(
        sqlx::query_file_as!(KeyView, "src/key/queries/find_key_view.sql", token)
            .fetch_optional(executor)
            .await?,
    )
}

/// The active key that the rotated-away `token` maps to while its grace period runs past `now`.
pub(super) async fn find_deprecated_key(
    executor: impl PgExecutor<'_>,
    token: &str,
    now: NaiveDateTime,
) -> Result<Option<DeprecatedKey>, Error> {
    Ok(sqlx::query_file_as!(
        DeprecatedKey,
        "src/key/queries/find_deprecated_key.sql",
        token,
        now
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{KeyView, find_deprecated_key, find_key_view};
    use crate::test_support::database_pool;
    use chrono::{Duration, NaiveDateTime, Utc};
    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    type Limits = (
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<f64>,
        Option<f64>,
    );

    async fn execute(
        tx: &mut Transaction<'_, Postgres>,
        sql: &'static str,
        binds: &[Option<&str>],
    ) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let mut query = sqlx::query(sql);
        for bind in binds {
            query = query.bind(*bind);
        }
        query.execute(&mut **tx).await.map(|_| ())
    }

    async fn key_view(
        tx: &mut Transaction<'_, Postgres>,
        token: &str,
    ) -> Result<KeyView, Box<dyn StdError>> {
        Ok(find_key_view(&mut **tx, token)
            .await?
            .ok_or("key view row")?)
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    async fn insert_budget(
        tx: &mut Transaction<'_, Postgres>,
        model_max_budget: Option<&str>,
    ) -> Result<(), sqlx::Error> {
        execute(
            tx,
            r#"INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, tpm_limit, rpm_limit, tpd_limit, soft_budget, model_max_budget, created_by, updated_by)
               VALUES ('kv-budget', 40.5, 10, 20, 30, 5.5, $1::jsonb, 'db-query-test', 'db-query-test')"#,
            &[model_max_budget],
        )
        .await
    }

    async fn insert_key_with_team(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        execute(
            tx,
            r#"INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, tpm_limit, soft_budget, model_max_budget, created_by, updated_by)
               VALUES ('kv-budget', 9.5, 77, 4.5, '{"gpt-4o": 1}', 'db-query-test', 'db-query-test')"#,
            &[],
        )
        .await?;
        execute(
            tx,
            r#"INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, models, blocked, max_budget, members_with_roles, metadata)
               VALUES ('kv-team', 'Team KV', ARRAY['gpt-4o'], true, 12.0,
                       '[{"user_id": "kv-user", "role": "user"}, {"user_id": "other", "role": "admin"}, {"user_id": "kv-user", "role": "admin"}]',
                       '{"tier": "gold"}')"#,
            &[],
        )
        .await?;
        execute(
            tx,
            r#"INSERT INTO "LiteLLM_VerificationToken" (token, user_id, team_id, budget_id, models, rpm_limit, expires, metadata)
               VALUES ('kv-token', 'kv-user', 'kv-team', 'kv-budget', ARRAY['m1'], 5, '2030-01-02 03:04:05.678', '{"a": [1, 2]}')"#,
            &[],
        )
        .await
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn key_view_joins_team_member_and_budget(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_key_with_team(&mut tx).await?;

        let view = key_view(&mut tx, "kv-token").await?;

        assert_eq!(
            (
                view.team_alias.as_deref(),
                view.team_blocked,
                view.team_models.as_slice(),
                view.rpm_limit,
                view.tpm_limit,
                view.max_budget,
                view.soft_budget,
                view.model_max_budget.get(),
                view.team_member.as_ref().map(crate::RawJson::get),
                view.metadata.get(),
                view.team_metadata.as_ref().map(crate::RawJson::get),
                view.expires,
            ),
            (
                Some("Team KV"),
                true,
                &["gpt-4o".to_owned()][..],
                Some(5),
                Some(77),
                Some(9.5),
                Some(4.5),
                r#"{"gpt-4o": 1}"#,
                Some(r#"{"role": "admin", "user_id": "kv-user"}"#),
                r#"{"a": [1, 2]}"#,
                Some(r#"{"tier": "gold"}"#),
                NaiveDateTime::parse_from_str("2030-01-02 03:04:05.678", "%Y-%m-%d %H:%M:%S%.f")
                    .ok(),
            )
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn key_view_serializes_timestamps_and_raw_json(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_key_with_team(&mut tx).await?;

        let json = serde_json::to_value(key_view(&mut tx, "kv-token").await?)?;

        assert_eq!(
            (
                &json["expires"],
                &json["team_member"],
                &json["budget_reset_at"],
                json.get("policies"),
                json.get("organization_id"),
            ),
            (
                &serde_json::json!("2030-01-02T03:04:05.678+00:00"),
                &serde_json::json!({"role": "admin", "user_id": "kv-user"}),
                &serde_json::Value::Null,
                None,
                None,
            )
        );
        Ok(())
    }

    #[rstest]
    #[case::key_organization(Some("kv-key-org"), Some("kv-team-org"), Some("kv-key-org"))]
    #[case::team_organization(None, Some("kv-team-org"), Some("kv-team-org"))]
    #[case::no_organization(None, None, None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn key_organization_falls_back_to_the_team_organization(
        #[case] key_org: Option<&str>,
        #[case] team_org: Option<&str>,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, None).await?;
        execute(
            &mut tx,
            r#"INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, created_by, updated_by)
               VALUES ('kv-key-org', 'Key Org', 'kv-budget', 'db-query-test', 'db-query-test'),
                      ('kv-team-org', 'Team Org', 'kv-budget', 'db-query-test', 'db-query-test')"#,
            &[],
        )
        .await?;
        execute(
            &mut tx,
            r#"INSERT INTO "LiteLLM_TeamTable" (team_id, organization_id) VALUES ('kv-team', $1)"#,
            &[team_org],
        )
        .await?;
        execute(
            &mut tx,
            r#"INSERT INTO "LiteLLM_VerificationToken" (token, team_id, organization_id) VALUES ('kv-token', 'kv-team', $1)"#,
            &[key_org],
        )
        .await?;

        let view = key_view(&mut tx, "kv-token").await?;

        assert_eq!(view.org_id.as_deref(), expected);
        Ok(())
    }

    #[rstest]
    #[case::key_limits(
        Some("kv-budget"),
        [Some("1"), Some("2"), Some("3"), Some("4.5")],
        (Some(1), Some(2), Some(3), Some(4.5), Some(5.5))
    )]
    #[case::budget_limits(
        Some("kv-budget"),
        [None; 4],
        (Some(10), Some(20), Some(30), Some(40.5), Some(5.5))
    )]
    #[case::no_budget(None, [None; 4], (None, None, None, None, None))]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn unset_key_limits_fall_back_to_the_budget_table(
        #[case] budget_id: Option<&str>,
        #[case] limits: [Option<&str>; 4],
        #[case] expected: Limits,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, None).await?;
        let [tpm, rpm, tpd, max_budget] = limits;
        execute(
            &mut tx,
            r#"INSERT INTO "LiteLLM_VerificationToken" (token, budget_id, tpm_limit, rpm_limit, tpd_limit, max_budget)
               VALUES ('kv-token', $1, $2::bigint, $3::bigint, $4::bigint, $5::double precision)"#,
            &[budget_id, tpm, rpm, tpd, max_budget],
        )
        .await?;

        let view = key_view(&mut tx, "kv-token").await?;

        assert_eq!(
            (
                view.tpm_limit,
                view.rpm_limit,
                view.tpd_limit,
                view.max_budget,
                view.soft_budget,
            ),
            expected
        );
        Ok(())
    }

    #[rstest]
    #[case::empty_object_takes_the_budget(
        "{}",
        Some(r#"{"gpt-4o": 1}"#),
        Some("kv-budget"),
        r#"{"gpt-4o": 1}"#
    )]
    #[case::object_is_kept(
        r#"{"gpt-4o": 2}"#,
        Some(r#"{"gpt-4o": 1}"#),
        Some("kv-budget"),
        r#"{"gpt-4o": 2}"#
    )]
    #[case::list_is_kept("[]", Some(r#"{"gpt-4o": 1}"#), Some("kv-budget"), "[]")]
    #[case::budget_json_null("{}", Some("null"), Some("kv-budget"), "null")]
    #[case::budget_without_model_max_budget("{}", None, Some("kv-budget"), "{}")]
    #[case::no_budget("{}", Some(r#"{"gpt-4o": 1}"#), None, "{}")]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn only_an_empty_key_model_max_budget_takes_the_budget_table(
        #[case] key_value: &str,
        #[case] budget_value: Option<&str>,
        #[case] budget_id: Option<&str>,
        #[case] expected: &str,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, budget_value).await?;
        execute(
            &mut tx,
            r#"INSERT INTO "LiteLLM_VerificationToken" (token, budget_id, model_max_budget) VALUES ('kv-token', $1, $2::jsonb)"#,
            &[budget_id, Some(key_value)],
        )
        .await?;

        let view = key_view(&mut tx, "kv-token").await?;

        assert_eq!(view.model_max_budget.get(), expected);
        Ok(())
    }

    #[rstest]
    #[case::member(
        Some(r#"[{"user_id": "kv-user", "role": "user"}]"#),
        Some("kv-user"),
        Some(r#"{"role": "user", "user_id": "kv-user"}"#)
    )]
    #[case::duplicate_user_picks_the_last(
        Some(r#"[{"user_id": "kv-user", "role": "user"}, {"user_id": "other", "role": "admin"}, {"user_id": "kv-user", "role": "admin"}]"#),
        Some("kv-user"),
        Some(r#"{"role": "admin", "user_id": "kv-user"}"#)
    )]
    #[case::no_match(
        Some(r#"[{"user_id": "other", "role": "admin"}]"#),
        Some("kv-user"),
        None
    )]
    #[case::non_string_user_id(Some(r#"[{"user_id": 5, "role": "admin"}]"#), Some("5"), None)]
    #[case::not_a_list(Some("{}"), Some("kv-user"), None)]
    #[case::json_null(Some("null"), Some("kv-user"), None)]
    #[case::key_without_user(Some(r#"[{"user_id": "kv-user", "role": "user"}]"#), None, None)]
    #[case::no_team(None, Some("kv-user"), None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn team_member_is_the_last_entry_for_the_key_user(
        #[case] members: Option<&str>,
        #[case] user_id: Option<&str>,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        if members.is_some() {
            execute(
                &mut tx,
                r#"INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles) VALUES ('kv-team', $1::jsonb)"#,
                &[members],
            )
            .await?;
        }
        execute(
            &mut tx,
            r#"INSERT INTO "LiteLLM_VerificationToken" (token, user_id, team_id) VALUES ('kv-token', $1, $2)"#,
            &[user_id, members.map(|_| "kv-team")],
        )
        .await?;

        let view = key_view(&mut tx, "kv-token").await?;

        assert_eq!(view.team_member.as_ref().map(crate::RawJson::get), expected);
        Ok(())
    }

    #[rstest]
    #[case::no_team(None, None, (&[][..], &[][..], false))]
    #[case::unset_models(None, Some((None, "false")), (&[][..], &[][..], false))]
    #[case::set_models(
        Some("{m1}"),
        Some((Some("{gpt-4o}"), "true")),
        (&["m1"][..], &["gpt-4o"][..], true)
    )]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn models_and_team_flags_default_when_unset(
        #[case] key_models: Option<&str>,
        #[case] team: Option<(Option<&str>, &str)>,
        #[case] expected: (&[&str], &[&str], bool),
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        if let Some((team_models, blocked)) = team {
            execute(
                &mut tx,
                r#"INSERT INTO "LiteLLM_TeamTable" (team_id, models, blocked) VALUES ('kv-team', $1::text[], $2::boolean)"#,
                &[team_models, Some(blocked)],
            )
            .await?;
        }
        execute(
            &mut tx,
            r#"INSERT INTO "LiteLLM_VerificationToken" (token, models, team_id) VALUES ('kv-token', $1::text[], $2)"#,
            &[key_models, team.map(|_| "kv-team")],
        )
        .await?;

        let view = key_view(&mut tx, "kv-token").await?;

        assert_eq!(
            (view.models, view.team_models, view.team_blocked),
            (strings(expected.0), strings(expected.1), expected.2)
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_key_has_no_view(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;

        assert_eq!(find_key_view(&mut *tx, "no-such-token").await?, None);
        Ok(())
    }

    #[rstest]
    #[case::within_grace_period(Duration::hours(1), Some("kv-active"))]
    #[case::after_revoke(Duration::hours(-1), None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn deprecated_key_resolves_until_revoked(
        #[case] revoke_in: Duration,
        #[case] expected: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let now = Utc::now().naive_utc();
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"INSERT INTO "LiteLLM_DeprecatedVerificationToken" (id, token, active_token_id, revoke_at)
               VALUES ('kv-deprecated', 'kv-old', 'kv-active', $1)"#,
        )
        .bind(now + revoke_in)
        .execute(&mut *tx)
        .await?;

        let found = find_deprecated_key(&mut *tx, "kv-old", now).await?;

        assert_eq!(found.map(|key| key.active_token_id).as_deref(), expected);
        Ok(())
    }
}
