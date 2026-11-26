//! Reads the `LiteLLM_UISettings` row.

use sqlx::PgExecutor;

use litellm_db::Error;

use crate::RawJson;

/// The `ui_settings` column of the row with id `ui_settings`, or `None` when there is no such row.
pub(super) async fn find_ui_settings(
    executor: impl PgExecutor<'_>,
) -> Result<Option<RawJson>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/ui_settings/queries/find_ui_settings.sql")
            .fetch_optional(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::find_ui_settings;
    use crate::test_support::database_pool;

    async fn clear_ui_settings(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture cleanup is outside the compile-checked query set"
        )]
        let result = sqlx::query(r#"DELETE FROM "LiteLLM_UISettings""#)
            .execute(&mut **tx)
            .await;
        result.map(|_| ())
    }

    async fn insert_ui_settings(
        tx: &mut Transaction<'_, Postgres>,
        id: &str,
        ui_settings: &str,
    ) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"INSERT INTO "LiteLLM_UISettings" (id, ui_settings, updated_at) VALUES ($1, $2::jsonb, NOW())"#,
        )
        .bind(id)
        .bind(ui_settings)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[case::object(r#"{"forward_client_headers_to_llm_api": true}"#)]
    #[case::json_text_saved_as_a_string(r#""{\"forward_client_headers_to_llm_api\": true}""#)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_the_ui_settings_row_as_stored(
        #[case] stored: &str,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        clear_ui_settings(&mut tx).await?;
        insert_ui_settings(&mut tx, "other_row", r#"{"disable_custom_api_keys": true}"#).await?;
        insert_ui_settings(&mut tx, "ui_settings", stored).await?;

        let found = find_ui_settings(&mut *tx).await?;

        assert_eq!(found.as_ref().map(|value| value.get()), Some(stored));
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_row_reads_as_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        clear_ui_settings(&mut tx).await?;
        insert_ui_settings(&mut tx, "other_row", r#"{"disable_custom_api_keys": true}"#).await?;

        assert_eq!(find_ui_settings(&mut *tx).await?, None);
        Ok(())
    }
}
