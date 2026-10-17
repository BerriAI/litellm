use sqlx::PgPool;

use crate::{Error, RateLimits};

/// An end user's spend and budget, as the proxy's end-user budget checks read them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EndUser {
    /// `LiteLLM_EndUserTable.spend`, when the end user has a row.
    pub spend: Option<f64>,
    pub blocked: bool,
    /// The budget's `max_budget`: the override's, else the row's, else the default's.
    pub max_budget: Option<f64>,
    /// That budget's `rpm_limit` and `tpm_limit`.
    pub limits: RateLimits,
}

/// Reads `end_user_id`, with `default_budget_id` as the budget of an end user without its own
/// and `override_budget_id`, when given, as its budget whatever the row says.
pub async fn find_end_user(
    pool: &PgPool,
    end_user_id: &str,
    default_budget_id: Option<&str>,
    override_budget_id: Option<&str>,
) -> Result<EndUser, Error> {
    let row = sqlx::query!(
        r#"SELECT e.spend AS "spend?",
                  e.blocked AS "blocked?",
                  b.max_budget AS "max_budget?",
                  b.rpm_limit AS "rpm_limit?",
                  b.tpm_limit AS "tpm_limit?"
           FROM (SELECT 1) AS one
           LEFT JOIN "LiteLLM_EndUserTable" AS e ON e.user_id = $1
           LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = COALESCE($3, e.budget_id, $2)"#,
        end_user_id,
        default_budget_id,
        override_budget_id,
    )
    .fetch_one(pool)
    .await?;
    Ok(EndUser {
        spend: row.spend,
        blocked: row.blocked.unwrap_or(false),
        max_budget: row.max_budget,
        limits: RateLimits {
            rpm: row.rpm_limit,
            tpm: row.tpm_limit,
        },
    })
}
