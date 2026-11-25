//! Reads `LiteLLM_Config` rows, one per settings section the proxy stores in the database.

use sqlx::PgExecutor;

use litellm_db::Error;

use crate::RawJson;

/// One `LiteLLM_Config` row. `param_value` is the section exactly as stored, still holding encrypted values,
/// and `None` when the column is SQL `NULL`.
#[macro_rules_attribute::apply(crate::db_row)]
pub struct ConfigParam {
    pub param_name: String,
    pub param_value: Option<RawJson>,
    pub reload_revision: i64,
}

/// The rows for whichever of `param_names` exist, in no particular order.
pub(super) async fn find_config_params(
    executor: impl PgExecutor<'_>,
    param_names: &[String],
) -> Result<Vec<ConfigParam>, Error> {
    Ok(sqlx::query_file_as!(
        ConfigParam,
        "src/config/queries/find_config_params.sql",
        param_names
    )
    .fetch_all(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::find_config_params;
    use crate::test_support::database_pool;

    async fn clear_config(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture cleanup is outside the compile-checked query set"
        )]
        let result = sqlx::query(r#"DELETE FROM "LiteLLM_Config""#)
            .execute(&mut **tx)
            .await;
        result.map(|_| ())
    }

    async fn insert_config(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_Config" (param_name, param_value, reload_revision)
            VALUES ('general_settings', '{"store_model_in_db": true}', 0),
                   ('litellm_settings', NULL, 0),
                   ('router_settings', '{"routing_strategy": "simple-shuffle"}', 7),
                   ('ui_settings', '{"disable_model_add_for_internal_users": true}', 0)
            "#,
        )
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_requested_params_with_values_and_revisions(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        clear_config(&mut tx).await?;
        insert_config(&mut tx).await?;

        let mut found = find_config_params(
            &mut *tx,
            &names(&[
                "general_settings",
                "litellm_settings",
                "router_settings",
                "not_stored",
            ]),
        )
        .await?;
        found.sort_by(|left, right| left.param_name.cmp(&right.param_name));

        assert_eq!(
            found
                .iter()
                .map(|row| (
                    row.param_name.as_str(),
                    row.param_value.as_ref().map(|value| value.get()),
                    row.reload_revision,
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "general_settings",
                    Some(r#"{"store_model_in_db": true}"#),
                    0
                ),
                ("litellm_settings", None, 0),
                (
                    "router_settings",
                    Some(r#"{"routing_strategy": "simple-shuffle"}"#),
                    7
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
        clear_config(&mut tx).await?;

        assert_eq!(
            find_config_params(
                &mut *tx,
                &names(&["general_settings", "environment_variables"])
            )
            .await?,
            vec![]
        );
        Ok(())
    }
}
