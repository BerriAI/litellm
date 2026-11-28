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
