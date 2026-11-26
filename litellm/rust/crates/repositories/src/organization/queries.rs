//! `LiteLLM_OrganizationTable` reads for auth: one by id, with its `LiteLLM_BudgetTable` row nested in on request, or
//! the rows sharing an alias.

use serde::Serialize;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::{RawJson, budget::Budget};

/// One `LiteLLM_OrganizationTable` row, with its budget only when the caller asked for it, as the host's
/// `LiteLLM_OrganizationTable` model reads it
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Serialize)]
pub struct Organization {
    pub organization_id: String,
    pub organization_alias: String,
    pub budget_id: String,
    pub metadata: RawJson,
    pub models: Vec<String>,
    pub spend: f64,
    pub model_spend: RawJson,
    pub object_permission_id: Option<String>,
    pub created_by: String,
    pub updated_by: String,
    pub litellm_budget_table: Option<Json<Budget>>,
}

/// The organization `organization_id`, joined to its budget when `include_budget_table` is set, or `None` when no row has
/// that id.
pub(super) async fn find_organization(
    executor: impl PgExecutor<'_>,
    organization_id: &str,
    include_budget_table: bool,
) -> Result<Option<Organization>, Error> {
    Ok(sqlx::query_file_as!(
        Organization,
        "src/organization/queries/find_organization.sql",
        organization_id,
        include_budget_table,
    )
    .fetch_optional(executor)
    .await?)
}

/// Up to two organizations whose alias is `organization_alias`, without their budgets, so a caller can tell a
/// unique alias from a shared one.
pub(super) async fn find_organizations_by_alias(
    executor: impl PgExecutor<'_>,
    organization_alias: &str,
) -> Result<Vec<Organization>, Error> {
    Ok(sqlx::query_file_as!(
        Organization,
        "src/organization/queries/find_organizations_by_alias.sql",
        organization_alias,
    )
    .fetch_all(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{find_organization, find_organizations_by_alias};
    use crate::test_support::{database_pool, insert_budget};
    use rstest::rstest;
    use sqlx::PgPool;

    #[rstest]
    #[case::with_budget(true, Some(Some(10.5)))]
    #[case::without_budget(false, None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_organization_and_budget_on_request(
        #[case] include_budget_table: bool,
        #[case] expected_max_budget: Option<Option<f64>>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, "org-budget").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _inserted = sqlx::query(
            r#"INSERT INTO "LiteLLM_OrganizationTable"
                   (organization_id, organization_alias, budget_id, models, metadata, created_by, updated_by)
               VALUES ('find-org', 'Org', 'org-budget', NULL, '{"k": [1]}', 'a', 'b')"#,
        )
        .execute(&mut *tx)
        .await?;

        let organization = find_organization(&mut *tx, "find-org", include_budget_table)
            .await?
            .ok_or("organization missing")?;

        assert_eq!(organization.organization_alias, "Org");
        assert!(organization.models.is_empty());
        assert_eq!(organization.metadata.get(), r#"{"k": [1]}"#);
        assert_eq!(
            organization
                .litellm_budget_table
                .map(|budget| budget.max_budget),
            expected_max_budget
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_organization_is_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        assert!(
            find_organization(&pool, "no-such-org", true)
                .await?
                .is_none()
        );
        Ok(())
    }

    #[rstest]
    #[case::unique("Unique", &["org-alias-a"])]
    #[case::shared("Shared", &["org-alias-b", "org-alias-c"])]
    #[case::missing("No such org", &[])]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_organizations_by_alias_without_budgets(
        #[case] alias: &str,
        #[case] expected: &[&str],
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, "org-alias-budget").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _inserted = sqlx::query(
            r#"INSERT INTO "LiteLLM_OrganizationTable"
                   (organization_id, organization_alias, budget_id, created_by, updated_by)
               VALUES ('org-alias-a', 'Unique', 'org-alias-budget', 'a', 'b'),
                      ('org-alias-b', 'Shared', 'org-alias-budget', 'a', 'b'),
                      ('org-alias-c', 'Shared', 'org-alias-budget', 'a', 'b'),
                      ('org-alias-d', 'Shared', 'org-alias-budget', 'a', 'b')"#,
        )
        .execute(&mut *tx)
        .await?;

        let organizations = find_organizations_by_alias(&mut *tx, alias).await?;

        assert_eq!(
            organizations
                .iter()
                .map(|organization| (
                    organization.organization_id.as_str(),
                    organization.litellm_budget_table.is_some()
                ))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|organization_id| (*organization_id, false))
                .collect::<Vec<_>>()
        );
        Ok(())
    }
}
