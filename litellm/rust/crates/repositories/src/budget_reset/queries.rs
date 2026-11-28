//! The reset scheduler's paged reads, guarded advances, and maintained window writes.

use chrono::NaiveDateTime;
use serde_json::Value;
use sqlx::PgExecutor;

use super::WindowOwner;
use crate::RawJson;

#[macro_rules_attribute::apply(crate::db_row)]
pub struct BudgetWindows {
    pub id: String,
    pub limits: RawJson,
}

#[macro_rules_attribute::apply(crate::db_row)]
pub struct LifetimeBudget {
    pub id: String,
    pub duration: String,
    pub reset_at: Option<NaiveDateTime>,
    pub spend: f64,
    pub max_budget: Option<f64>,
    pub rollover_max_budget: Option<f64>,
    pub rollover_credit: Option<f64>,
}

#[macro_rules_attribute::apply(crate::db_row)]
pub struct ResetEntity {
    pub kind: String,
    pub id: String,
    pub team_id: Option<String>,
}

pub(super) async fn advance_tier(
    executor: impl PgExecutor<'_>,
    budget: &LifetimeBudget,
    end: NaiveDateTime,
    rollover: bool,
    default_end_user_tier: Option<&str>,
) -> Result<Vec<ResetEntity>, sqlx::Error> {
    sqlx::query_file_as!(
        ResetEntity,
        "src/budget_reset/queries/advance_budget_lifetime.sql",
        budget.id,
        end,
        budget.reset_at,
        budget.duration,
        rollover,
        default_end_user_tier,
    )
    .fetch_all(executor)
    .await
}

pub(super) async fn lifetime_page(
    executor: impl PgExecutor<'_>,
    owner: WindowOwner,
    after: &str,
    now: NaiveDateTime,
    limit: i64,
) -> Result<Vec<LifetimeBudget>, sqlx::Error> {
    macro_rules! page {
        ($file:literal) => {
            sqlx::query_file_as!(LifetimeBudget, $file, after, now, limit)
                .fetch_all(executor)
                .await
        };
    }
    match owner {
        WindowOwner::Key => page!("src/budget_reset/queries/page_key_lifetime.sql"),
        WindowOwner::Team => page!("src/budget_reset/queries/page_team_lifetime.sql"),
        WindowOwner::User => page!("src/budget_reset/queries/page_user_lifetime.sql"),
        WindowOwner::Budget => page!("src/budget_reset/queries/page_budget_lifetime.sql"),
    }
}

pub(super) async fn advance_lifetime(
    executor: impl PgExecutor<'_>,
    owner: WindowOwner,
    budget: &LifetimeBudget,
    end: NaiveDateTime,
    decrement: f64,
    credit: Option<f64>,
) -> Result<bool, sqlx::Error> {
    macro_rules! advance {
        ($file:literal $(, $arg:expr)*) => {
            sqlx::query_file!($file, budget.id, end, decrement, budget.reset_at, budget.duration $(, $arg)*)
                .execute(executor)
                .await?
        };
    }
    let result = match owner {
        WindowOwner::Key => advance!("src/budget_reset/queries/advance_key_lifetime.sql"),
        WindowOwner::Team => advance!("src/budget_reset/queries/advance_team_lifetime.sql"),
        WindowOwner::User => advance!("src/budget_reset/queries/advance_user_lifetime.sql", credit),
        WindowOwner::Budget => return Ok(false),
    };
    Ok(result.rows_affected() == 1)
}

pub(super) async fn page(
    executor: impl PgExecutor<'_>,
    owner: WindowOwner,
    after: &str,
    limit: i64,
) -> Result<Vec<BudgetWindows>, sqlx::Error> {
    macro_rules! page {
        ($file:literal) => {
            sqlx::query_file_as!(BudgetWindows, $file, after, limit)
                .fetch_all(executor)
                .await
        };
    }
    match owner {
        WindowOwner::Key => page!("src/budget_reset/queries/page_key_windows.sql"),
        WindowOwner::Team => page!("src/budget_reset/queries/page_team_windows.sql"),
        WindowOwner::User => page!("src/budget_reset/queries/page_user_windows.sql"),
        WindowOwner::Budget => page!("src/budget_reset/queries/page_budget_windows.sql"),
    }
}

pub(super) async fn advance(
    executor: impl PgExecutor<'_>,
    owner: WindowOwner,
    id: &str,
    expected: &str,
    updated: &Value,
) -> Result<bool, sqlx::Error> {
    macro_rules! advance {
        ($file:literal) => {
            sqlx::query_file!($file, id, expected, updated)
                .execute(executor)
                .await?
        };
    }
    let result = match owner {
        WindowOwner::Key => advance!("src/budget_reset/queries/advance_key_windows.sql"),
        WindowOwner::Team => advance!("src/budget_reset/queries/advance_team_windows.sql"),
        WindowOwner::User => advance!("src/budget_reset/queries/advance_user_windows.sql"),
        WindowOwner::Budget => advance!("src/budget_reset/queries/advance_budget_windows.sql"),
    };
    Ok(result.rows_affected() == 1)
}

pub(super) async fn roll(
    executor: impl PgExecutor<'_>,
    entity_type: &str,
    id: &str,
    durations: &[String],
    starts: &[NaiveDateTime],
) -> Result<(), sqlx::Error> {
    sqlx::query_file!(
        "src/budget_reset/queries/roll_window_spend.sql",
        entity_type,
        id,
        durations,
        starts,
    )
    .execute(executor)
    .await?;
    Ok(())
}
