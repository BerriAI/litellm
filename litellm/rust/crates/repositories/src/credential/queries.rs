//! Reads every `LiteLLM_CredentialsTable` row, the named credentials models refer to by
//! `litellm_credential_name`.

use sqlx::PgExecutor;

use litellm_db::Error;

use crate::RawJson;

/// One `LiteLLM_CredentialsTable` row. `credential_values` is exactly as stored, so each value is still
/// encrypted.
#[macro_rules_attribute::apply(crate::db_row)]
pub struct Credential {
    pub credential_name: String,
    pub credential_values: RawJson,
    pub credential_info: Option<RawJson>,
}

/// Every row, in no particular order.
pub(super) async fn find_credentials(
    executor: impl PgExecutor<'_>,
) -> Result<Vec<Credential>, Error> {
    Ok(
        sqlx::query_file_as!(Credential, "src/credential/queries/find_credentials.sql")
            .fetch_all(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::find_credentials;
    use crate::test_support::database_pool;

    async fn clear_credentials(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture cleanup is outside the compile-checked query set"
        )]
        let result = sqlx::query(r#"DELETE FROM "LiteLLM_CredentialsTable""#)
            .execute(&mut **tx)
            .await;
        result.map(|_| ())
    }

    async fn insert_credentials(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_CredentialsTable" (
                credential_id, credential_name, credential_values, credential_info, created_by, updated_by
            )
            VALUES ('credential-1', 'openai-shared', '{"api_key": "encrypted-key"}',
                    '{"custom_llm_provider": "openai"}', 'admin', 'admin'),
                   ('credential-2', 'azure-shared', '{"api_key": "encrypted-azure", "api_base": "encrypted-base"}',
                    NULL, 'admin', 'admin')
            "#,
        )
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_every_credential_row(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        clear_credentials(&mut tx).await?;
        insert_credentials(&mut tx).await?;

        let mut found = find_credentials(&mut *tx).await?;
        found.sort_by(|left, right| left.credential_name.cmp(&right.credential_name));

        assert_eq!(
            found
                .iter()
                .map(|credential| (
                    credential.credential_name.as_str(),
                    credential.credential_values.get(),
                    credential.credential_info.as_ref().map(|info| info.get()),
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "azure-shared",
                    r#"{"api_key": "encrypted-azure", "api_base": "encrypted-base"}"#,
                    None,
                ),
                (
                    "openai-shared",
                    r#"{"api_key": "encrypted-key"}"#,
                    Some(r#"{"custom_llm_provider": "openai"}"#),
                ),
            ]
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
        clear_credentials(&mut tx).await?;

        assert_eq!(find_credentials(&mut *tx).await?, vec![]);
        Ok(())
    }
}
