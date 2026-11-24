//! Reads every `LiteLLM_ProxyModelTable` row, the model deployments added through the proxy's API.

use chrono::NaiveDateTime;
use sqlx::PgExecutor;

use litellm_db::Error;

use crate::RawJson;

/// One `LiteLLM_ProxyModelTable` row. `litellm_params` and `model_info` are exactly as stored, so the
/// deployment's credentials in `litellm_params` are still encrypted.
#[macro_rules_attribute::apply(crate::db_row)]
pub struct ProxyModel {
    pub model_id: String,
    pub model_name: String,
    pub litellm_params: RawJson,
    pub model_info: Option<RawJson>,
    pub blocked: bool,
    pub created_at: NaiveDateTime,
    pub created_by: String,
    pub updated_at: NaiveDateTime,
    pub updated_by: String,
}

/// Every row, in no particular order.
pub(super) async fn find_proxy_models(
    executor: impl PgExecutor<'_>,
) -> Result<Vec<ProxyModel>, Error> {
    Ok(
        sqlx::query_file_as!(ProxyModel, "src/proxy_model/queries/find_proxy_models.sql")
            .fetch_all(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use chrono::NaiveDate;
    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::find_proxy_models;
    use crate::test_support::database_pool;

    async fn clear_models(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture cleanup is outside the compile-checked query set"
        )]
        let result = sqlx::query(r#"DELETE FROM "LiteLLM_ProxyModelTable""#)
            .execute(&mut **tx)
            .await;
        result.map(|_| ())
    }

    async fn insert_models(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_ProxyModelTable" (
                model_id, model_name, litellm_params, model_info, blocked,
                created_at, created_by, updated_at, updated_by
            )
            VALUES ('model-a', 'gpt', '{"model": "openai/gpt-5", "api_key": "encrypted-a"}',
                    '{"team_id": "team-1"}', false,
                    '2026-11-01 00:00:00', 'admin', '2026-11-02 12:30:00.250', 'editor'),
                   ('model-b', 'claude', '{"model": "anthropic/claude", "litellm_credential_name": "shared"}',
                    NULL, true,
                    '2026-11-03 00:00:00', 'admin', '2026-11-03 00:00:00', 'admin')
            "#,
        )
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_every_model_row(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        clear_models(&mut tx).await?;
        insert_models(&mut tx).await?;

        let mut found = find_proxy_models(&mut *tx).await?;
        found.sort_by(|left, right| left.model_id.cmp(&right.model_id));

        assert_eq!(
            found
                .iter()
                .map(|model| (
                    model.model_id.as_str(),
                    model.model_name.as_str(),
                    model.litellm_params.get(),
                    model.model_info.as_ref().map(|info| info.get()),
                    model.blocked,
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "model-a",
                    "gpt",
                    r#"{"model": "openai/gpt-5", "api_key": "encrypted-a"}"#,
                    Some(r#"{"team_id": "team-1"}"#),
                    false,
                ),
                (
                    "model-b",
                    "claude",
                    r#"{"model": "anthropic/claude", "litellm_credential_name": "shared"}"#,
                    None,
                    true,
                ),
            ]
        );
        let audit = found.first().map(|model| {
            (
                model.created_at,
                model.created_by.as_str(),
                model.updated_at,
                model.updated_by.as_str(),
            )
        });
        assert_eq!(
            audit,
            Some((
                NaiveDate::from_ymd_opt(2026, 11, 1)
                    .and_then(|day| day.and_hms_opt(0, 0, 0))
                    .ok_or("valid timestamp")?,
                "admin",
                NaiveDate::from_ymd_opt(2026, 11, 2)
                    .and_then(|day| day.and_hms_milli_opt(12, 30, 0, 250))
                    .ok_or("valid timestamp")?,
                "editor",
            ))
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn empty_table_reads_as_no_rows(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        clear_models(&mut tx).await?;

        assert_eq!(find_proxy_models(&mut *tx).await?, vec![]);
        Ok(())
    }
}
