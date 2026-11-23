//! Model access group budgets in `LiteLLM_ModelAccessGroupBudgetTable`.

use sqlx::PgExecutor;

use litellm_db::Error;

use crate::RawJson;

/// A group's spend with the budget row it points at, if any.
#[macro_rules_attribute::apply(crate::db_row)]
pub(super) struct ModelAccessGroupBudgetRow {
    pub access_group_name: String,
    pub spend: f64,
    pub max_budget: Option<f64>,
    pub budget_limits: Option<RawJson>,
    pub budget_id: Option<String>,
}

/// Up to `limit` names of groups that have a budget row.
pub(super) async fn model_access_group_names(
    executor: impl PgExecutor<'_>,
    limit: i64,
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/model_access_group/queries/model_access_group_names.sql",
        limit
    )
    .fetch_all(executor)
    .await?)
}

/// The budget rows of whichever of `names` have one.
pub(super) async fn find_model_access_group_budgets(
    executor: impl PgExecutor<'_>,
    names: &[String],
) -> Result<Vec<ModelAccessGroupBudgetRow>, Error> {
    Ok(sqlx::query_file_as!(
        ModelAccessGroupBudgetRow,
        "src/model_access_group/queries/find_model_access_group_budgets.sql",
        names
    )
    .fetch_all(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{find_model_access_group_budgets, model_access_group_names};
    use crate::test_support::database_pool;
    use rstest::rstest;
    use sqlx::PgPool;

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn budgets_join_the_budget_row(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, budget_limits, created_by, updated_by)
            VALUES ('mag-budget', 3.5, '[{"budget_duration": "1d", "max_budget": 1}]', 'db-query-test', 'db-query-test')
            "#,
        )
        .execute(&mut *tx)
        .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_ModelAccessGroupBudgetTable" (access_group_name, spend, budget_id)
            VALUES ('mag-with-budget', 1.25, 'mag-budget'), ('mag-without-budget', 0, NULL)
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let mut rows = find_model_access_group_budgets(
            &mut *tx,
            &[
                "mag-with-budget".to_owned(),
                "mag-without-budget".to_owned(),
                "mag-absent".to_owned(),
            ],
        )
        .await?;
        rows.sort_by(|a, b| a.access_group_name.cmp(&b.access_group_name));
        let names = model_access_group_names(&mut *tx, 1000).await?;

        assert_eq!(
            rows.iter()
                .map(|row| (
                    row.access_group_name.as_str(),
                    row.spend,
                    row.max_budget,
                    row.budget_limits.as_ref().map(crate::RawJson::get),
                    row.budget_id.as_deref(),
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "mag-with-budget",
                    1.25,
                    Some(3.5),
                    Some(r#"[{"max_budget": 1, "budget_duration": "1d"}]"#),
                    Some("mag-budget"),
                ),
                ("mag-without-budget", 0.0, None, None, None),
            ]
        );
        assert!(
            names.contains(&"mag-with-budget".to_owned())
                && names.contains(&"mag-without-budget".to_owned())
        );
        Ok(())
    }
}
