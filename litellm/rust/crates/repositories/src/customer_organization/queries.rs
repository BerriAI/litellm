//! Reads one `LiteLLM_CustomerOrganizationTable` row with its budget, as the spend counter reseed sees it.

use chrono::NaiveDateTime;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::{budget::Budget, utc_timestamp};

/// One `LiteLLM_CustomerOrganizationTable` row and the `LiteLLM_BudgetTable` row its `budget_id` names, if it
/// exists, under the `budget` relation name the host reads
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct CustomerOrganization {
    customer_org_id: String,
    alias: Option<String>,
    spend: f64,
    budget_template_ids: Vec<String>,
    budget_id: Option<String>,
    default_budget_template_id: Option<String>,
    #[serde(with = "utc_timestamp")]
    created_at: NaiveDateTime,
    created_by: String,
    #[serde(with = "utc_timestamp")]
    updated_at: NaiveDateTime,
    updated_by: String,
    budget: Option<Json<Budget>>,
}

pub(super) async fn find_customer_organization(
    executor: impl PgExecutor<'_>,
    customer_org_id: &str,
) -> Result<Option<CustomerOrganization>, Error> {
    Ok(sqlx::query_file_as!(
        CustomerOrganization,
        "src/customer_organization/queries/find_customer_organization.sql",
        customer_org_id
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::find_customer_organization;
    use crate::test_support::{
        database_pool, insert_budget, insert_customer_organization, inserted_budget_json,
    };

    #[rstest]
    #[case::with_budget(Some("customer-org-budget"))]
    #[case::without_budget(None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_customer_organization_with_its_budget(
        #[case] budget_id: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        if let Some(budget_id) = budget_id {
            insert_budget(&mut tx, budget_id).await?;
        }
        insert_customer_organization(&mut tx, "customer-org-1", budget_id).await?;

        let found = find_customer_organization(&mut *tx, "customer-org-1")
            .await?
            .ok_or("row missing")?;

        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "customer_org_id": "customer-org-1",
                "alias": "acme",
                "spend": 2.5,
                "budget_template_ids": ["template-1"],
                "budget_id": budget_id,
                "default_budget_template_id": null,
                "created_at": "2026-01-02T03:04:05.678+00:00",
                "created_by": "creator",
                "updated_at": "2026-02-03T04:05:06.789+00:00",
                "updated_by": "updater",
                "budget": budget_id.map(inserted_budget_json),
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
            find_customer_organization(&pool, "customer-org-missing")
                .await?
                .is_none()
        );
        Ok(())
    }
}
