use sqlx::{
    PgConnection,
    types::{
        JsonValue,
        chrono::{DateTime, Utc},
    },
};

use crate::Error;

declare_queries! {
    apply_baseline_session_corrections(changes: &JsonValue) => "../../../litellm/proxy/db/queries/autorouter/apply_baseline_session_corrections.sql";
    apply_baseline_user_session_corrections(changes: &JsonValue) => "../../../litellm/proxy/db/queries/autorouter/apply_baseline_user_session_corrections.sql";
    claim_dirty_baseline_comparisons() -> BaselineComparisonScopeRow => "../../../litellm/proxy/db/queries/autorouter/claim_dirty_baseline_comparisons.sql";
    delete_retired_baseline_observations(cutoff: DateTime<Utc>, batch_size: i32) => "../../../litellm/proxy/db/queries/autorouter/delete_retired_baseline_observations.sql";
    find_baseline_observation_changed_before(scope: &str, published_revision: i64, last_at: Option<f64>) -> FoundRow => "../../../litellm/proxy/db/queries/autorouter/find_baseline_observation_changed_before.sql";
    mark_baseline_observation_conflicted(request_id: &str, scope: &str, data: &str, revision: i64) => "../../../litellm/proxy/db/queries/autorouter/mark_baseline_observation_conflicted.sql";
    publish_baseline_to_spend_logs(changes: &JsonValue) => "../../../litellm/proxy/db/queries/autorouter/publish_baseline_to_spend_logs.sql";
    read_baseline_observation(request_id: &str, scope: &str) -> BaselineObservationRow => "../../../litellm/proxy/db/queries/autorouter/read_baseline_observation.sql";
    read_baseline_observation_page(scope: &str, after_revision: i64, cursor: Option<f64>, page_timestamps: i32, withdraw_from: Option<f64>) -> BaselineObservationRow => "../../../litellm/proxy/db/queries/autorouter/read_baseline_observation_page.sql";
    retire_expired_baseline_comparisons(cutoff: DateTime<Utc>, batch_size: i32) => "../../../litellm/proxy/db/queries/autorouter/retire_expired_baseline_comparisons.sql";
    store_baseline_publications(changes: &JsonValue) => "../../../litellm/proxy/db/queries/autorouter/store_baseline_publications.sql";
}

declare_rows! {
    struct BaselineComparisonScopeRow {
        scope: String,
    }

    struct FoundRow {
        found: Option<i32>,
    }

    struct BaselineObservationRow {
        data: String,
        publication: Option<String>,
        conflicted: bool,
        started_at: f64,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LockedComparison {
    pub revision: i64,
    pub published_revision: i64,
    pub retired: bool,
    pub checkpoint: HistoryCheckpoint,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HistoryCheckpoint {
    Initial { equivalent: bool },
    Published { history: String },
}

struct LockedComparisonRow {
    revision: i64,
    published_revision: i64,
    initial_equivalent: bool,
    retired: bool,
    history: Option<String>,
}

impl From<LockedComparisonRow> for LockedComparison {
    fn from(row: LockedComparisonRow) -> Self {
        let checkpoint = match row.history {
            Some(history) => HistoryCheckpoint::Published { history },
            None => HistoryCheckpoint::Initial {
                equivalent: row.initial_equivalent,
            },
        };
        Self {
            revision: row.revision,
            published_revision: row.published_revision,
            retired: row.retired,
            checkpoint,
        }
    }
}

pub async fn create_baseline_comparison(
    conn: &mut PgConnection,
    scope: &str,
    api_key: &str,
    session_id: &str,
    router_name: &str,
) -> Result<bool, Error> {
    let result = sqlx::query_file!(
        "../../../litellm/proxy/db/queries/autorouter/create_baseline_comparison.sql",
        scope,
        api_key,
        session_id,
        router_name,
    )
    .execute(conn)
    .await?;
    Ok(result.rows_affected() == 1)
}

pub async fn lock_baseline_comparison(
    conn: &mut PgConnection,
    scope: &str,
) -> Result<Option<LockedComparison>, Error> {
    let row = sqlx::query_file_as!(
        LockedComparisonRow,
        "../../../litellm/proxy/db/queries/autorouter/lock_baseline_comparison.sql",
        scope,
    )
    .fetch_optional(conn)
    .await?;
    Ok(row.map(LockedComparison::from))
}

pub async fn advance_baseline_comparison_revision(
    conn: &mut PgConnection,
    scope: &str,
    revision: i64,
) -> Result<bool, Error> {
    let result = sqlx::query_file!(
        "../../../litellm/proxy/db/queries/autorouter/advance_baseline_comparison_revision.sql",
        scope,
        revision,
    )
    .execute(conn)
    .await?;
    Ok(result.rows_affected() == 1)
}

pub async fn publish_baseline_comparison_history(
    conn: &mut PgConnection,
    scope: &str,
    history: &str,
) -> Result<bool, Error> {
    let result = sqlx::query_file!(
        "../../../litellm/proxy/db/queries/autorouter/publish_baseline_comparison_history.sql",
        scope,
        history,
    )
    .execute(conn)
    .await?;
    Ok(result.rows_affected() == 1)
}

pub async fn insert_baseline_observation(
    conn: &mut PgConnection,
    request_id: &str,
    scope: &str,
    started_at: f64,
    revision: i64,
    data: &str,
) -> Result<bool, Error> {
    let result = sqlx::query_file!(
        "../../../litellm/proxy/db/queries/autorouter/insert_baseline_observation.sql",
        request_id,
        scope,
        started_at,
        revision,
        data,
    )
    .execute(conn)
    .await?;
    Ok(result.rows_affected() == 1)
}

pub async fn find_baseline_observation_without_spend_log(
    conn: &mut PgConnection,
    scope: &str,
) -> Result<bool, Error> {
    let found = sqlx::query_file_scalar!(
        "../../../litellm/proxy/db/queries/autorouter/find_baseline_observation_without_spend_log.sql",
        scope,
    )
    .fetch_optional(conn)
    .await?;
    Ok(found.is_some())
}
