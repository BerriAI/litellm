//! Reads `LiteLLM_TagTable` rows with their budgets, and the bounded list of tag names the registry caches.

use chrono::NaiveDateTime;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::{RawJson, budget::Budget, utc_timestamp};

/// One `LiteLLM_TagTable` row and the `LiteLLM_BudgetTable` row its `budget_id` names, if it exists.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct Tag {
    tag_name: String,
    description: Option<String>,
    models: Vec<String>,
    model_info: Option<RawJson>,
    spend: f64,
    budget_id: Option<String>,
    #[serde(with = "utc_timestamp")]
    created_at: NaiveDateTime,
    created_by: Option<String>,
    #[serde(with = "utc_timestamp")]
    updated_at: NaiveDateTime,
    litellm_budget_table: Option<Json<Budget>>,
}

/// The rows for `tag_names` that exist, in no particular order.
pub(super) async fn find_tags(
    executor: impl PgExecutor<'_>,
    tag_names: &[String],
) -> Result<Vec<Tag>, Error> {
    Ok(
        sqlx::query_file_as!(Tag, "src/tag/queries/find_tags.sql", tag_names)
            .fetch_all(executor)
            .await?,
    )
}

/// Up to `limit` tag names, in no particular order. The registry asks for one more than it can hold to
/// tell a full table from an overflowing one.
pub(super) async fn find_tag_names(
    executor: impl PgExecutor<'_>,
    limit: i64,
) -> Result<Vec<String>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/tag/queries/find_tag_names.sql", limit)
            .fetch_all(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::{find_tag_names, find_tags};
    use crate::test_support::{database_pool, insert_budget};

    async fn insert_tags(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
        insert_budget(tx, "tag-budget").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_TagTable" (tag_name, description, models, model_info, spend, budget_id)
            VALUES ('tag-a', 'first', '{m1}', '{"id1": "m1"}', 1.5, 'tag-budget'),
                   ('tag-b', NULL, NULL, NULL, 0, NULL)
            "#,
        )
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_requested_tags_with_budgets(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_tags(&mut tx).await?;

        let mut found = find_tags(
            &mut *tx,
            &[
                "tag-a".to_owned(),
                "tag-b".to_owned(),
                "tag-missing".to_owned(),
            ],
        )
        .await?;
        found.sort_by(|left, right| left.tag_name.cmp(&right.tag_name));

        assert_eq!(
            found
                .iter()
                .map(|tag| (
                    tag.tag_name.as_str(),
                    tag.description.as_deref(),
                    tag.models.clone(),
                    tag.model_info.as_ref().map(|info| info.get()),
                    tag.spend,
                    tag.litellm_budget_table
                        .as_ref()
                        .map(|budget| budget.budget_id.as_str()),
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "tag-a",
                    Some("first"),
                    vec!["m1".to_owned()],
                    Some(r#"{"id1": "m1"}"#),
                    1.5,
                    Some("tag-budget"),
                ),
                ("tag-b", None, vec![], None, 0.0, None),
            ]
        );
        Ok(())
    }

    #[rstest]
    #[case::under_limit(10, 2)]
    #[case::at_limit(1, 1)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn tag_names_stop_at_limit(
        #[case] limit: i64,
        #[case] expected: usize,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture cleanup is outside the compile-checked query set"
        )]
        sqlx::query(r#"DELETE FROM "LiteLLM_TagTable""#)
            .execute(&mut *tx)
            .await?;
        insert_tags(&mut tx).await?;

        assert_eq!(find_tag_names(&mut *tx, limit).await?.len(), expected);
        Ok(())
    }
}
