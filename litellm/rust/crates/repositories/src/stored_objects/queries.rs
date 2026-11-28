//! Reads whether the stored object tables hold a row that changes how the proxy answers.

use sqlx::PgExecutor;

use litellm_db::Error;

/// Whether the database holds each kind of stored object, as Python reads it into what a request
/// reaches.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Copy, Default, Eq)]
pub struct StoredObjects {
    /// An active `LiteLLM_GuardrailsTable` row with `default_on` set, which runs on every request.
    pub default_on_guardrails: bool,
    /// A `LiteLLM_PolicyAttachmentTable` row, which attaches a policy to some requests.
    pub policy_attachments: bool,
    /// A `LiteLLM_ToolTable` row whose input or output policy is not the default `untrusted`.
    pub tool_policies: bool,
    /// A non-empty `cache_config` row in `LiteLLM_CacheConfig`, which turns response caching on.
    pub cache_settings: bool,
    /// A `hashicorp_vault` or `cyberark` row in `LiteLLM_ConfigOverrides`, which replaces the
    /// secret manager.
    pub config_overrides: bool,
    /// The `anthropic_beta_headers_reload_config` row of `LiteLLM_Config`.
    pub anthropic_beta_headers_reload: bool,
    /// The `model_cost_map_reload_config` row of `LiteLLM_Config`.
    pub model_cost_map_reload: bool,
}

pub(super) async fn find_stored_objects(
    executor: impl PgExecutor<'_>,
) -> Result<StoredObjects, Error> {
    Ok(sqlx::query_file_as!(
        StoredObjects,
        "src/stored_objects/queries/find_stored_objects.sql"
    )
    .fetch_one(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::{StoredObjects, find_stored_objects};
    use crate::test_support::database_pool;

    const TABLES: [&str; 6] = [
        r#"DELETE FROM "LiteLLM_GuardrailsTable""#,
        r#"DELETE FROM "LiteLLM_PolicyAttachmentTable""#,
        r#"DELETE FROM "LiteLLM_ToolTable""#,
        r#"DELETE FROM "LiteLLM_CacheConfig""#,
        r#"DELETE FROM "LiteLLM_ConfigOverrides""#,
        r#"DELETE FROM "LiteLLM_Config""#,
    ];

    async fn run(
        tx: &mut Transaction<'_, Postgres>,
        statements: &[&'static str],
    ) -> Result<(), sqlx::Error> {
        for statement in statements {
            #[expect(
                clippy::disallowed_methods,
                reason = "Fixture statements are outside the compile-checked query set"
            )]
            let result = sqlx::query(*statement).execute(&mut **tx).await;
            result?;
        }
        Ok(())
    }

    const NONE: StoredObjects = StoredObjects {
        default_on_guardrails: false,
        policy_attachments: false,
        tool_policies: false,
        cache_settings: false,
        config_overrides: false,
        anthropic_beta_headers_reload: false,
        model_cost_map_reload: false,
    };

    #[rstest]
    #[case::nothing(None, NONE)]
    #[case::a_default_on_guardrail(
        Some(r#"INSERT INTO "LiteLLM_GuardrailsTable" (guardrail_id, guardrail_name, litellm_params, status, updated_at) VALUES ('g', 'g', '{"guardrail": "presidio", "default_on": true}', 'active', NOW())"#),
        StoredObjects { default_on_guardrails: true, ..NONE }
    )]
    #[case::default_on_written_as_a_word(
        Some(r#"INSERT INTO "LiteLLM_GuardrailsTable" (guardrail_id, guardrail_name, litellm_params, status, updated_at) VALUES ('g', 'g', '{"guardrail": "presidio", "default_on": "yes"}', 'active', NOW())"#),
        StoredObjects { default_on_guardrails: true, ..NONE }
    )]
    #[case::a_guardrail_a_request_asks_for(Some(r#"INSERT INTO "LiteLLM_GuardrailsTable" (guardrail_id, guardrail_name, litellm_params, status, updated_at) VALUES ('g', 'g', '{"guardrail": "presidio", "default_on": false}', 'active', NOW())"#), NONE)]
    #[case::a_guardrail_awaiting_review(Some(r#"INSERT INTO "LiteLLM_GuardrailsTable" (guardrail_id, guardrail_name, litellm_params, status, updated_at) VALUES ('g', 'g', '{"guardrail": "presidio", "default_on": true}', 'pending_review', NOW())"#), NONE)]
    #[case::a_policy_attachment(
        Some(r#"INSERT INTO "LiteLLM_PolicyAttachmentTable" (attachment_id, policy_name, scope) VALUES ('a', 'p', '*')"#),
        StoredObjects { policy_attachments: true, ..NONE }
    )]
    #[case::a_discovered_tool(
        Some(r#"INSERT INTO "LiteLLM_ToolTable" (tool_id, tool_name) VALUES ('t', 'search')"#),
        NONE
    )]
    #[case::a_blocked_tool(
        Some(r#"INSERT INTO "LiteLLM_ToolTable" (tool_id, tool_name, input_policy) VALUES ('t', 'search', 'blocked')"#),
        StoredObjects { tool_policies: true, ..NONE }
    )]
    #[case::empty_cache_settings(
        Some(r#"INSERT INTO "LiteLLM_CacheConfig" (id, cache_settings, updated_at) VALUES ('cache_config', '{}', NOW())"#),
        NONE
    )]
    #[case::cache_settings(
        Some(r#"INSERT INTO "LiteLLM_CacheConfig" (id, cache_settings, updated_at) VALUES ('cache_config', '{"type": "redis"}', NOW())"#),
        StoredObjects { cache_settings: true, ..NONE }
    )]
    #[case::a_vault_override(
        Some(r#"INSERT INTO "LiteLLM_ConfigOverrides" (config_type, config_value, updated_at) VALUES ('hashicorp_vault', '{}', NOW())"#),
        StoredObjects { config_overrides: true, ..NONE }
    )]
    #[case::an_anthropic_beta_headers_reload(
        Some(r#"INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES ('anthropic_beta_headers_reload_config', '{"interval_hours": 6}')"#),
        StoredObjects { anthropic_beta_headers_reload: true, ..NONE }
    )]
    #[case::a_model_cost_map_reload(
        Some(r#"INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES ('model_cost_map_reload_config', '{"interval_hours": 6}')"#),
        StoredObjects { model_cost_map_reload: true, ..NONE }
    )]
    #[case::a_settings_row(
        Some(r#"INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES ('general_settings', '{}')"#),
        NONE
    )]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_which_stored_objects_the_database_holds(
        #[case] row: Option<&'static str>,
        #[case] expected: StoredObjects,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        run(&mut tx, &TABLES).await?;
        run(&mut tx, row.as_slice()).await?;

        assert_eq!(find_stored_objects(&mut *tx).await?, expected);
        Ok(())
    }
}
