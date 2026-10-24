use sqlx::PgExecutor;

use crate::Error;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RateLimits {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EndUser {
    pub spend: Option<f64>,
    pub blocked: bool,
    pub max_budget: Option<f64>,
    pub limits: RateLimits,
}

#[macro_rules_attribute::apply(crate::db_row)]
struct EndUserRow {
    spend: Option<f64>,
    blocked: Option<bool>,
    max_budget: Option<f64>,
    rpm_limit: Option<i64>,
    tpm_limit: Option<i64>,
}

pub async fn find_end_user(
    executor: impl PgExecutor<'_>,
    end_user_id: &str,
    default_budget_id: Option<&str>,
    key_budget_id: Option<&str>,
) -> Result<EndUser, Error> {
    let row = sqlx::query_file_as!(
        EndUserRow,
        "queries/end_user/find_end_user.sql",
        end_user_id,
        default_budget_id,
        key_budget_id,
    )
    .fetch_one(executor)
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
