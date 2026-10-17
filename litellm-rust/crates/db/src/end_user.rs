use sqlx::PgPool;

use crate::Error;

/// An end user's spend and budget, as the proxy's end-user budget checks read them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EndUser {
    /// `LiteLLM_EndUserTable.spend`, when the end user has a row.
    pub spend: Option<f64>,
    pub blocked: bool,
    /// The row's budget, else the default budget's.
    pub max_budget: Option<f64>,
}

/// Reads `end_user_id`, with `default_budget_id` as the budget of an end user without its own.
pub async fn find_end_user(
    pool: &PgPool,
    end_user_id: &str,
    default_budget_id: Option<&str>,
) -> Result<EndUser, Error> {
    let row = sqlx::query!(
        r#"SELECT e.spend AS "spend?",
                  e.blocked AS "blocked?",
                  b.max_budget AS "max_budget?",
                  (SELECT d.max_budget FROM "LiteLLM_BudgetTable" AS d
                    WHERE d.budget_id = $2) AS "default_max_budget?"
           FROM (SELECT 1) AS one
           LEFT JOIN "LiteLLM_EndUserTable" AS e ON e.user_id = $1
           LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = e.budget_id"#,
        end_user_id,
        default_budget_id,
    )
    .fetch_one(pool)
    .await?;
    Ok(EndUser {
        spend: row.spend,
        blocked: row.blocked.unwrap_or(false),
        max_budget: row.max_budget.or(row.default_max_budget),
    })
}
