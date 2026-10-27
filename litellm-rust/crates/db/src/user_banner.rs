use serde_json::Value;
use sqlx::PgExecutor;

use crate::Error;

pub async fn find_user_banner(executor: impl PgExecutor<'_>) -> Result<Option<Value>, Error> {
    Ok(
        sqlx::query_file_scalar!("queries/user_banner/find_user_banner.sql")
            .fetch_optional(executor)
            .await?,
    )
}

pub async fn save_user_banner(
    executor: impl PgExecutor<'_>,
    settings: &Value,
) -> Result<(), Error> {
    sqlx::query_file!("queries/user_banner/save_user_banner.sql", settings)
        .execute(executor)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use serde_json::{Value, json};
    use sqlx::PgPool;
    use sqlx::types::chrono::NaiveDateTime;

    use super::{find_user_banner, save_user_banner};
    use crate::test_support::database_pool;

    async fn clear(tx: &mut sqlx::PgConnection) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture cleanup is outside the compile-checked query set"
        )]
        sqlx::query(r#"DELETE FROM "LiteLLM_UISettings" WHERE id = 'user_banner'"#)
            .execute(tx)
            .await?;
        Ok(())
    }

    async fn stored_row(
        tx: &mut sqlx::PgConnection,
    ) -> Result<(Value, NaiveDateTime, NaiveDateTime), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture reads are outside the compile-checked query set"
        )]
        sqlx::query_as(
            r#"SELECT ui_settings, created_at, updated_at FROM "LiteLLM_UISettings" WHERE id = 'user_banner'"#,
        )
        .fetch_one(tx)
        .await
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_banner_reads_as_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let mut tx = database_pool?.begin().await?;
        clear(&mut tx).await?;

        assert_eq!(find_user_banner(&mut *tx).await?, None);
        Ok(())
    }

    #[rstest]
    #[case::first_publish(None)]
    #[case::republish(Some(json!({"enabled": false, "message": "old", "severity": "info", "revision": "a"})))]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn save_replaces_the_stored_banner_and_stamps_updated_at(
        #[case] previous: Option<Value>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let mut tx = database_pool?.begin().await?;
        clear(&mut tx).await?;
        if let Some(previous) = &previous {
            save_user_banner(&mut *tx, previous).await?;
        }
        let banner = json!({"enabled": true, "message": "maintenance at 5pm", "severity": "warning", "revision": "b"});

        save_user_banner(&mut *tx, &banner).await?;

        assert_eq!(find_user_banner(&mut *tx).await?, Some(banner.clone()));
        let (stored, created_at, updated_at) = stored_row(&mut tx).await?;
        assert_eq!(stored, banner);
        assert!(updated_at >= created_at);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn banner_text_is_stored_as_data_not_sql(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let mut tx = database_pool?.begin().await?;
        clear(&mut tx).await?;
        let banner = json!({"message": r#"'); DROP TABLE "LiteLLM_UISettings"; --"#});

        save_user_banner(&mut *tx, &banner).await?;

        assert_eq!(find_user_banner(&mut *tx).await?, Some(banner));
        Ok(())
    }
}
