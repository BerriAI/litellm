#[expect(
    dead_code,
    reason = "definition only: Python executes this query until Rust takes it over"
)]
async fn key_auth_combined_view(executor: impl sqlx::PgExecutor<'_>, hashed_token: &str) {
    let _ = sqlx::query_file!(
        "../../../litellm/proxy/db/queries/key_auth_combined_view.sql",
        hashed_token
    )
    .fetch_optional(executor)
    .await;
}
