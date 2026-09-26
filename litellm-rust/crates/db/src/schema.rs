use sqlx::PgExecutor;

use crate::Error;

pub const REQUIRED_MIGRATION: &str = env!("LITELLM_REQUIRED_MIGRATION");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaCompatibility {
    Compatible,
    Behind { required: String },
}

pub async fn schema_compatibility(
    executor: impl PgExecutor<'_>,
    required: &str,
) -> Result<SchemaCompatibility, Error> {
    let applied = sqlx::query_scalar!(
        r#"SELECT EXISTS (
            SELECT 1 FROM _prisma_migrations
            WHERE migration_name = $1 AND finished_at IS NOT NULL AND rolled_back_at IS NULL
        ) AS "applied!""#,
        required,
    )
    .fetch_one(executor)
    .await?;
    Ok(if applied {
        SchemaCompatibility::Compatible
    } else {
        SchemaCompatibility::Behind {
            required: required.to_owned(),
        }
    })
}
