//! The rows the proxy loads on top of its config file when `store_model_in_db` is on: the settings
//! sections in `LiteLLM_Config`, the models in `LiteLLM_ProxyModelTable`, the guardrails in
//! `LiteLLM_GuardrailsTable` and the policy attachments in `LiteLLM_PolicyAttachmentTable`.

use sqlx::PgExecutor;

use litellm_db::Error;

/// A settings section the database holds as an object: the keys it sets, and whether it turns
/// `store_model_in_db` on the way the proxy reads it, `true` or any spelling of `"true"`.
#[macro_rules_attribute::apply(crate::db_row)]
pub(super) struct ConfigSectionRow {
    pub param_name: String,
    pub keys: Vec<String>,
    pub store_model_in_db: bool,
}

/// Whichever of `sections` the database holds as an object.
pub(super) async fn config_sections(
    executor: impl PgExecutor<'_>,
    sections: &[String],
) -> Result<Vec<ConfigSectionRow>, Error> {
    Ok(sqlx::query_file_as!(
        ConfigSectionRow,
        "src/stored_config/queries/config_sections.sql",
        sections
    )
    .fetch_all(executor)
    .await?)
}

/// The name of every model group a database model joins, each once.
pub(super) async fn model_names(executor: impl PgExecutor<'_>) -> Result<Vec<String>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/stored_config/queries/model_names.sql")
            .fetch_all(executor)
            .await?,
    )
}

/// The active guardrails whose `default_on` the proxy reads as true, so they run on every request.
/// A value it would not read as false counts as on.
pub(super) async fn default_on_guardrails(
    executor: impl PgExecutor<'_>,
) -> Result<Vec<String>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/stored_config/queries/default_on_guardrails.sql")
            .fetch_all(executor)
            .await?,
    )
}

/// The name of every policy an attachment attaches, each once.
pub(super) async fn attached_policies(executor: impl PgExecutor<'_>) -> Result<Vec<String>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/stored_config/queries/attached_policies.sql")
            .fetch_all(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{attached_policies, config_sections, default_on_guardrails, model_names};
    use crate::test_support::database_pool;
    use rstest::rstest;
    use sqlx::PgPool;

    const SECTIONS: [&str; 2] = ["general_settings", "router_settings"];

    const FIXTURES: [&str; 8] = [
        r#"DELETE FROM "LiteLLM_Config""#,
        r#"DELETE FROM "LiteLLM_ProxyModelTable""#,
        r#"DELETE FROM "LiteLLM_GuardrailsTable""#,
        r#"DELETE FROM "LiteLLM_PolicyAttachmentTable""#,
        r#"INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES
            ('general_settings', '{"store_model_in_db": "True", "alerting": ["slack"]}'),
            ('router_settings', '{"fallbacks": []}'),
            ('litellm_settings', '{"drop_params": true}'),
            ('environment_variables', '"not an object"')"#,
        r#"INSERT INTO "LiteLLM_ProxyModelTable" (model_id, model_name, litellm_params, created_by, updated_by) VALUES
            ('a', 'claude', '{}', 'test', 'test'),
            ('b', 'claude', '{}', 'test', 'test'),
            ('c', 'gpt-*', '{}', 'test', 'test')"#,
        r#"INSERT INTO "LiteLLM_GuardrailsTable" (guardrail_id, guardrail_name, litellm_params, status, updated_at) VALUES
            ('a', 'on', '{"default_on": true}', 'active', NOW()),
            ('b', 'spelled-on', '{"default_on": "yes"}', 'active', NOW()),
            ('c', 'off', '{"default_on": false}', 'active', NOW()),
            ('d', 'unset', '{}', 'active', NOW()),
            ('e', 'pending', '{"default_on": true}', 'pending_review', NOW())"#,
        r#"INSERT INTO "LiteLLM_PolicyAttachmentTable" (attachment_id, policy_name, scope) VALUES
            ('a', 'pii', '*'), ('b', 'pii', NULL), ('c', 'audit', NULL)"#,
    ];

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn each_read_returns_what_the_proxy_loads(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        for fixture in FIXTURES {
            #[expect(
                clippy::disallowed_methods,
                reason = "Fixture inserts are outside the compile-checked query set"
            )]
            sqlx::query(fixture).execute(&mut *tx).await?;
        }

        let mut sections = config_sections(&mut *tx, &SECTIONS.map(str::to_owned)).await?;
        sections.sort_by(|a, b| a.param_name.cmp(&b.param_name));

        assert_eq!(
            (
                sections
                    .iter()
                    .map(|row| (
                        row.param_name.as_str(),
                        row.keys.clone(),
                        row.store_model_in_db
                    ))
                    .collect::<Vec<_>>(),
                model_names(&mut *tx).await?,
                default_on_guardrails(&mut *tx).await?,
                attached_policies(&mut *tx).await?,
            ),
            (
                vec![
                    (
                        "general_settings",
                        vec!["alerting".to_owned(), "store_model_in_db".to_owned()],
                        true
                    ),
                    ("router_settings", vec!["fallbacks".to_owned()], false),
                ],
                vec!["claude".to_owned(), "gpt-*".to_owned()],
                vec!["on".to_owned(), "spelled-on".to_owned()],
                vec!["audit".to_owned(), "pii".to_owned()],
            )
        );
        Ok(())
    }
}
