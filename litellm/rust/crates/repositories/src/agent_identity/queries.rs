//! The write that records a managed agent's authentication on `LiteLLM_AgentIdentity`.

use sqlx::PgExecutor;

use litellm_db::Error;

/// Stamps `last_authenticated_at` on the identity of `agent_id` while it is active at `revision` and its agent is
/// enabled and identity managed; whether a row was stamped.
pub(super) async fn record_authentication(
    executor: impl PgExecutor<'_>,
    agent_id: &str,
    revision: &str,
) -> Result<bool, Error> {
    let done = sqlx::query_file!(
        "src/agent_identity/queries/record_authentication.sql",
        agent_id,
        revision
    )
    .execute(executor)
    .await?;
    Ok(done.rows_affected() == 1)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use crate::test_support::{database_pool, leave_utc};

    /// `agent-identity-a` as the case sets it up, beside `agent-identity-b`, which is always current
    async fn insert_agents(
        tx: &mut Transaction<'_, Postgres>,
        enabled: bool,
        identity_managed: bool,
        active: bool,
    ) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_AgentsTable"
                (agent_id, agent_name, agent_card_params, enabled, identity_managed, created_by, updated_by)
            VALUES ('agent-identity-a', 'agent-identity-a', '{}', $1, $2, 'db-query-test', 'db-query-test'),
                   ('agent-identity-b', 'agent-identity-b', '{}', true, true, 'db-query-test', 'db-query-test')
            "#,
        )
        .bind(enabled)
        .bind(identity_managed)
        .execute(&mut **tx)
        .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_AgentIdentity"
                (agent_id, active, provider, issuer, tenant_id, client_id, revision)
            VALUES ('agent-identity-a', $1, 'microsoft', 'https://issuer', 'tenant', 'client-a', 'rev-1'),
                   ('agent-identity-b', true, 'microsoft', 'https://issuer', 'tenant', 'client-b', 'rev-1')
            "#,
        )
        .bind(active)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[case::current(true, true, true, "rev-1", true)]
    #[case::stale_revision(true, true, true, "rev-0", false)]
    #[case::inactive_identity(true, true, false, "rev-1", false)]
    #[case::disabled_agent(false, true, true, "rev-1", false)]
    #[case::agent_no_longer_identity_managed(true, false, true, "rev-1", false)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn authentication_is_recorded_in_utc_only_on_the_current_identity_of_a_managed_agent(
        #[case] enabled: bool,
        #[case] identity_managed: bool,
        #[case] active: bool,
        #[case] revision: &str,
        #[case] expected: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_agents(&mut tx, enabled, identity_managed, active).await?;
        leave_utc(&mut tx).await?;

        let recorded = super::record_authentication(&mut *tx, "agent-identity-a", revision).await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let stamped_within_a_second: Vec<(String, Option<bool>)> = sqlx::query_as(
            r#"
            SELECT agent_id, abs(extract(epoch FROM last_authenticated_at - (now() AT TIME ZONE 'UTC'))) < 1
            FROM "LiteLLM_AgentIdentity"
            WHERE agent_id IN ('agent-identity-a', 'agent-identity-b')
            ORDER BY agent_id
            "#,
        )
        .fetch_all(&mut *tx)
        .await?;
        assert_eq!(
            (recorded, stamped_within_a_second),
            (
                expected,
                vec![
                    ("agent-identity-a".to_owned(), expected.then_some(true)),
                    ("agent-identity-b".to_owned(), None),
                ]
            )
        );
        Ok(())
    }
}
