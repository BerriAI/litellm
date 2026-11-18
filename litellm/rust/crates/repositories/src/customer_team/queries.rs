//! Reads one `LiteLLM_CustomerTeamTable` row with its budget and budget override, as the spend counter reseed
//! sees it.

use chrono::NaiveDateTime;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::{budget::Budget, utc_timestamp};

/// One `LiteLLM_CustomerTeamTable` row and the `LiteLLM_BudgetTable` rows its `budget_id` and
/// `budget_override_id` name, under the `budget` and `budget_override` relation names the host reads
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct CustomerTeam {
    customer_team_id: String,
    alias: Option<String>,
    parent_org_id: String,
    spend: f64,
    spend_reset_window: Option<String>,
    budget_id: Option<String>,
    budget_override_id: Option<String>,
    #[serde(with = "utc_timestamp::option")]
    budget_override_expires_at: Option<NaiveDateTime>,
    #[serde(with = "utc_timestamp")]
    created_at: NaiveDateTime,
    created_by: String,
    #[serde(with = "utc_timestamp")]
    updated_at: NaiveDateTime,
    updated_by: String,
    budget: Option<Json<Budget>>,
    budget_override: Option<Json<Budget>>,
}

pub(super) async fn find_customer_team(
    executor: impl PgExecutor<'_>,
    customer_team_id: &str,
) -> Result<Option<CustomerTeam>, Error> {
    Ok(sqlx::query_file_as!(
        CustomerTeam,
        "src/customer_team/queries/find_customer_team.sql",
        customer_team_id
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::find_customer_team;
    use crate::test_support::{
        database_pool, insert_budget, insert_customer_organization, inserted_budget_json,
    };

    async fn insert_customer_team(
        tx: &mut Transaction<'_, Postgres>,
        budget_id: Option<&str>,
        budget_override_id: Option<&str>,
    ) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_CustomerTeamTable"
                (customer_team_id, alias, parent_org_id, spend, spend_reset_window, budget_id, budget_override_id,
                 budget_override_expires_at, created_at, created_by, updated_at, updated_by)
            VALUES ('customer-team-1', 'platform', 'customer-org-1', 4.5, '1mo', $1, $2, '2026-12-24 00:00:00',
                    '2026-01-02 03:04:05.678', 'creator', '2026-02-03 04:05:06.789', 'updater')
            "#,
        )
        .bind(budget_id)
        .bind(budget_override_id)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[case::with_budgets(Some("team-budget"), Some("team-override"))]
    #[case::override_only(None, Some("team-override"))]
    #[case::without_budgets(None, None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_customer_team_with_its_budgets(
        #[case] budget_id: Option<&str>,
        #[case] budget_override_id: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        if let Some(budget_id) = budget_id {
            insert_budget(&mut tx, budget_id).await?;
        }
        if let Some(budget_override_id) = budget_override_id {
            insert_budget(&mut tx, budget_override_id).await?;
        }
        insert_customer_organization(&mut tx, "customer-org-1", None).await?;
        insert_customer_team(&mut tx, budget_id, budget_override_id).await?;

        let found = find_customer_team(&mut *tx, "customer-team-1")
            .await?
            .ok_or("row missing")?;

        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "customer_team_id": "customer-team-1",
                "alias": "platform",
                "parent_org_id": "customer-org-1",
                "spend": 4.5,
                "spend_reset_window": "1mo",
                "budget_id": budget_id,
                "budget_override_id": budget_override_id,
                "budget_override_expires_at": "2026-12-24T00:00:00+00:00",
                "created_at": "2026-01-02T03:04:05.678+00:00",
                "created_by": "creator",
                "updated_at": "2026-02-03T04:05:06.789+00:00",
                "updated_by": "updater",
                "budget": budget_id.map(inserted_budget_json),
                "budget_override": budget_override_id.map(inserted_budget_json),
            })
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_row_is_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        assert!(
            find_customer_team(&pool, "customer-team-missing")
                .await?
                .is_none()
        );
        Ok(())
    }
}
