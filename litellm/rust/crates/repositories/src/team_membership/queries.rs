//! `LiteLLM_TeamMembership` reads for auth, with the member's `LiteLLM_BudgetTable` row nested in by the SQL.

use serde::Serialize;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::budget::Budget;

/// A user's membership of a team, with the per-member budget when one is assigned, as the host's
/// `LiteLLM_TeamMembership` model reads it
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Serialize)]
pub struct TeamMembership {
    user_id: String,
    team_id: String,
    budget_id: Option<String>,
    spend: f64,
    total_spend: f64,
    litellm_budget_table: Option<Json<Budget>>,
}

/// `user_id`'s membership of `team_id`, or `None` when the user is not a member.
pub(super) async fn find_team_membership(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    team_id: &str,
) -> Result<Option<TeamMembership>, Error> {
    Ok(sqlx::query_file_as!(
        TeamMembership,
        "src/team_membership/queries/find_team_membership.sql",
        user_id,
        team_id,
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::find_team_membership;
    use crate::test_support::{database_pool, insert_budget};
    use rstest::rstest;
    use sqlx::PgPool;

    #[rstest]
    #[case::with_member_budget(Some("member-budget"))]
    #[case::without_member_budget(None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_membership_with_member_budget(
        #[case] budget_id: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, "member-budget").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _inserted = sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, total_spend, budget_id)
               VALUES ('member', 'member-team', 1.5, 2.5, $1)"#,
        )
        .bind(budget_id)
        .execute(&mut *tx)
        .await?;

        let membership = find_team_membership(&mut *tx, "member", "member-team")
            .await?
            .ok_or("membership missing")?;

        assert_eq!(membership.budget_id.as_deref(), budget_id);
        assert_eq!(membership.spend, 1.5);
        assert_eq!(membership.total_spend, 2.5);
        assert_eq!(
            membership
                .litellm_budget_table
                .map(|budget| (budget.rpm_limit, budget.allowed_models.clone())),
            budget_id.map(|_| (Some(10), Some(vec!["gpt-4o".to_owned()])))
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_membership_is_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        assert!(
            find_team_membership(&pool, "no-such-user", "no-such-team")
                .await?
                .is_none()
        );
        Ok(())
    }
}
