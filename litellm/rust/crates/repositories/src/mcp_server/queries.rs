//! Reads what the proxy's MCP registry takes from the stored MCP servers.

use sqlx::PgExecutor;

use litellm_db::Error;

pub(super) async fn any_server_forwards_headers(
    executor: impl PgExecutor<'_>,
) -> Result<bool, Error> {
    Ok(
        sqlx::query_file_scalar!("src/mcp_server/queries/any_server_forwards_headers.sql")
            .fetch_one(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::any_server_forwards_headers;
    use crate::test_support::database_pool;

    #[rstest]
    #[case::no_server(None, false)]
    #[case::a_server_forwarding_no_header(Some("{}"), false)]
    #[case::a_server_forwarding_only_the_authorization(Some("{Authorization}"), false)]
    #[case::a_server_forwarding_a_header(Some("{authorization,x-github-pat}"), true)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_stored_server_forwards_headers_only_through_a_name_besides_the_authorization(
        #[case] extra_headers: Option<&'static str>,
        #[case] forwards: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture statements are outside the compile-checked query set"
        )]
        sqlx::query(r#"DELETE FROM "LiteLLM_MCPServerTable""#)
            .execute(&mut *tx)
            .await?;
        if let Some(extra_headers) = extra_headers {
            #[expect(
                clippy::disallowed_methods,
                reason = "Fixture statements are outside the compile-checked query set"
            )]
            sqlx::query(
                r#"INSERT INTO "LiteLLM_MCPServerTable" (server_id, transport, mcp_access_groups, extra_headers) VALUES ('s', 'http', '{}', $1::text[])"#,
            )
            .bind(extra_headers)
            .execute(&mut *tx)
            .await?;
        }

        assert_eq!(any_server_forwards_headers(&mut *tx).await?, forwards);
        Ok(())
    }
}
