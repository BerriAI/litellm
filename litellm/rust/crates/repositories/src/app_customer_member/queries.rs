//! Reads one `LiteLLM_AppCustomerMemberTable` row, a customer member's standing within an app, as the spend
//! counter reseed sees it.

use chrono::NaiveDateTime;
use sqlx::PgExecutor;

use litellm_db::Error;

use crate::utc_timestamp;

/// Every scalar column of one `LiteLLM_AppCustomerMemberTable` row
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct AppCustomerMember {
    app_id: String,
    customer_member_id: String,
    spend: f64,
    spend_reset_window: Option<String>,
    budget_override_id: Option<String>,
    budget_id: Option<String>,
    #[serde(with = "utc_timestamp::option")]
    budget_override_expires_at: Option<NaiveDateTime>,
}

pub(super) async fn find_app_customer_member(
    executor: impl PgExecutor<'_>,
    app_id: &str,
    customer_member_id: &str,
) -> Result<Option<AppCustomerMember>, Error> {
    Ok(sqlx::query_file_as!(
        AppCustomerMember,
        "src/app_customer_member/queries/find_app_customer_member.sql",
        app_id,
        customer_member_id,
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::find_app_customer_member;
    use crate::test_support::{
        database_pool, insert_app, insert_budget, insert_customer_member,
        insert_customer_organization,
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
        insert_customer_member(&mut tx, "customer-member-1", "customer-org-1", None, None).await?;
        insert_budget(&mut tx, "app-member-override").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_AppCustomerMemberTable" (app_id, customer_member_id, spend, spend_reset_window, budget_override_id, budget_id, budget_override_expires_at)
            VALUES ('app-1', 'customer-member-1', 0.5, '1mo', 'app-member-override', 'app-member-override', '2026-12-24 00:00:00'), ('app-2', 'customer-member-1', 9.0, NULL, NULL, NULL, NULL)
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let found = find_app_customer_member(&mut *tx, "app-1", "customer-member-1")
            .await?
            .ok_or("row missing")?;

        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "app_id": "app-1",
                "customer_member_id": "customer-member-1",
                "spend": 0.5,
                "spend_reset_window": "1mo",
                "budget_override_id": "app-member-override",
                "budget_id": "app-member-override",
                "budget_override_expires_at": "2026-12-24T00:00:00+00:00",
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
            find_app_customer_member(&pool, "app-missing", "customer-member-missing")
                .await?
                .is_none()
        );
        Ok(())
    }
}
