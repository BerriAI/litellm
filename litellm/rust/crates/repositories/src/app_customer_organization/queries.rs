//! Reads one `LiteLLM_AppCustomerOrgTable` row, a customer organization's standing within an app, as the spend
//! counter reseed sees it.

use sqlx::PgExecutor;

use litellm_db::Error;

/// Every scalar column of one `LiteLLM_AppCustomerOrgTable` row
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct AppCustomerOrganization {
    app_id: String,
    customer_org_id: String,
    spend: f64,
    spend_reset_window: Option<String>,
    tier_budget_id: Option<String>,
    budget_override_id: Option<String>,
    customer_member_budget_id: Option<String>,
}

pub(super) async fn find_app_customer_organization(
    executor: impl PgExecutor<'_>,
    app_id: &str,
    customer_org_id: &str,
) -> Result<Option<AppCustomerOrganization>, Error> {
    Ok(sqlx::query_file_as!(
        AppCustomerOrganization,
        "src/app_customer_organization/queries/find_app_customer_organization.sql",
        app_id,
        customer_org_id,
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::find_app_customer_organization;
    use crate::test_support::{
        database_pool, insert_app, insert_budget, insert_customer_organization,
    };

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_only_the_named_pair(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_app(&mut tx, "app-1").await?;
        insert_app(&mut tx, "app-2").await?;
        insert_customer_organization(&mut tx, "customer-org-1", None).await?;
        insert_budget(&mut tx, "app-org-tier").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_AppCustomerOrgTable" (app_id, customer_org_id, spend, spend_reset_window, tier_budget_id)
            VALUES ('app-1', 'customer-org-1', 3.75, '1mo', 'app-org-tier'), ('app-2', 'customer-org-1', 9.0, NULL, NULL)
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let found = find_app_customer_organization(&mut *tx, "app-1", "customer-org-1")
            .await?
            .ok_or("row missing")?;

        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "app_id": "app-1",
                "customer_org_id": "customer-org-1",
                "spend": 3.75,
                "spend_reset_window": "1mo",
                "tier_budget_id": "app-org-tier",
                "budget_override_id": null,
                "customer_member_budget_id": null,
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
            find_app_customer_organization(&pool, "app-missing", "customer-org-missing")
                .await?
                .is_none()
        );
        Ok(())
    }
}
