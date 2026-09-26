use sqlx::types::{
    JsonValue,
    chrono::{DateTime, Utc},
};

declare_queries! {
    advance_baseline_comparison_revision(scope: &str, revision: i64) => "../../../litellm/proxy/db/queries/autorouter/advance_baseline_comparison_revision.sql";
    apply_baseline_session_corrections(changes: &JsonValue) => "../../../litellm/proxy/db/queries/autorouter/apply_baseline_session_corrections.sql";
    apply_baseline_user_session_corrections(changes: &JsonValue) => "../../../litellm/proxy/db/queries/autorouter/apply_baseline_user_session_corrections.sql";
    claim_dirty_baseline_comparisons() -> BaselineComparisonScopeRow => "../../../litellm/proxy/db/queries/autorouter/claim_dirty_baseline_comparisons.sql";
    create_baseline_comparison(scope: &str, api_key: &str, session_id: &str, router_name: &str) => "../../../litellm/proxy/db/queries/autorouter/create_baseline_comparison.sql";
    delete_retired_baseline_observations(cutoff: DateTime<Utc>, batch_size: i32) => "../../../litellm/proxy/db/queries/autorouter/delete_retired_baseline_observations.sql";
    find_baseline_observation_changed_before(scope: &str, published_revision: i64, last_at: Option<f64>) -> FoundRow => "../../../litellm/proxy/db/queries/autorouter/find_baseline_observation_changed_before.sql";
    find_baseline_observation_without_spend_log(scope: &str) -> FoundRow => "../../../litellm/proxy/db/queries/autorouter/find_baseline_observation_without_spend_log.sql";
    insert_baseline_observation(request_id: &str, scope: &str, started_at: f64, revision: i64, data: &str) => "../../../litellm/proxy/db/queries/autorouter/insert_baseline_observation.sql";
    lock_baseline_comparison(scope: &str) -> BaselineComparisonLockRow => "../../../litellm/proxy/db/queries/autorouter/lock_baseline_comparison.sql";
    mark_baseline_observation_conflicted(request_id: &str, scope: &str, data: &str, revision: i64) => "../../../litellm/proxy/db/queries/autorouter/mark_baseline_observation_conflicted.sql";
    publish_baseline_comparison_history(scope: &str, history: &str) => "../../../litellm/proxy/db/queries/autorouter/publish_baseline_comparison_history.sql";
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

    struct BaselineComparisonLockRow {
        revision: i64,
        published_revision: i64,
        initial_equivalent: bool,
        retired: bool,
        history: Option<String>,
    }

    struct BaselineObservationRow {
        data: String,
        publication: Option<String>,
        conflicted: bool,
        started_at: f64,
    }
}
