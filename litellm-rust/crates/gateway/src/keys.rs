use std::{sync::Arc, time::Duration};

use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_db::{PgPool, PoolSettings};
use litellm_gateway_auth::{
    KeyError,
    keys::{EntryFuture, KeyEntry, KeyHash, KeyLookup, LookupFuture},
};
use litellm_secrets::source::SecretSource;

use crate::{Error, key_status};

/// Virtual keys read from the proxy's `LiteLLM_VerificationToken` table.
pub struct PostgresKeys {
    pool: PgPool,
    personal_keys_use_teams: bool,
}

impl KeyLookup for PostgresKeys {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move { self.lookup_entry(hash).await.map(|entry| entry.status) })
    }

    fn lookup_entry<'a>(&'a self, hash: &'a KeyHash) -> EntryFuture<'a> {
        Box::pin(async move {
            let row = litellm_db::find_verification_token(&self.pool, hash.as_str())
                .await
                .map_err(|error| KeyError::Lookup(Box::new(error)))?;
            Ok(match row {
                Some(row) => KeyEntry {
                    team_id: row.team_id.clone(),
                    user_id: row.user_id.clone(),
                    status: Some(key_status::status(row, self.personal_keys_use_teams)),
                },
                None => KeyEntry::default(),
            })
        })
    }
}

/// The proxy's database, when one is configured: `general_settings.database_url` (which may be
/// an `os.environ/` reference) or else `DATABASE_URL`, as the Python proxy resolves it. The pool
/// connects on first use, so the gateway starts, and the master key works, with the database down.
pub async fn database(
    config: &Config,
    secrets: &dyn SecretSource,
) -> Result<Option<PgPool>, Error> {
    let Some(url) = database_url(config, secrets).await? else {
        return Ok(None);
    };
    let general = &config.general_settings;
    let defaults = PoolSettings::default();
    let settings = PoolSettings {
        max_connections: general
            .database_connection_pool_limit
            .and_then(|limit| u32::try_from(limit).ok())
            .unwrap_or(defaults.max_connections),
        acquire_timeout: general
            .database_connection_timeout
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            .unwrap_or(defaults.acquire_timeout),
        idle_timeout: general
            .database_max_idle_connection_lifetime
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            .or(defaults.idle_timeout),
    };
    Ok(Some(litellm_db::connect_lazy(url.expose(), &settings)?))
}

/// The key store over the proxy's database.
pub fn store(config: &Config, pool: PgPool) -> Result<Arc<dyn KeyLookup>, Error> {
    // Inference does not run global guardrails or policies, so a virtual key would get past
    // them; refuse to start rather than serve keys the operator expects to be guarded.
    if !config.guardrails.is_empty()
        || !config.policies.is_empty()
        || !config.policy_attachments.is_empty()
    {
        return Err(Error::GuardedVirtualKeys);
    }
    Ok(Arc::new(PostgresKeys {
        pool,
        personal_keys_use_teams: personal_keys_use_teams(config),
    }))
}

/// `litellm_settings.personal_key_model_access_from_teams`.
fn personal_keys_use_teams(config: &Config) -> bool {
    matches!(
        config
            .litellm_settings
            .additional_fields
            .get("personal_key_model_access_from_teams"),
        Some(litellm_config::Value::Bool(true))
    )
}

async fn database_url(
    config: &Config,
    secrets: &dyn SecretSource,
) -> Result<Option<SecretValue>, Error> {
    let configured = config
        .general_settings
        .database_url
        .as_ref()
        .filter(|url| !url.expose().trim().is_empty());
    let url = match configured {
        Some(url) => match url.expose().strip_prefix("os.environ/") {
            Some(name) => Some(
                secrets
                    .get_secret_str(name)
                    .await?
                    .ok_or_else(|| Error::DatabaseUrl(name.to_owned()))?,
            ),
            None => Some(url.clone()),
        },
        None => secrets.get_secret_str("DATABASE_URL").await?,
    };
    Ok(url.filter(|url| !url.expose().trim().is_empty()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use futures_util::future::BoxFuture;
    use rstest::rstest;

    use super::*;

    struct Secrets(HashMap<&'static str, &'static str>);

    impl SecretSource for Secrets {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
            Box::pin(async move { Ok(self.0.get(name).map(|value| SecretValue::new(*value))) })
        }
    }

    fn config(database_url: Option<&str>) -> Config {
        let settings = database_url
            .map(|url| format!("general_settings:\n  database_url: '{url}'\n"))
            .unwrap_or_default();
        Config::from_yaml(&format!("model_list: []\n{settings}")).unwrap()
    }

    #[rstest]
    #[case::none(None, &[], None)]
    #[case::environment(None, &[("DATABASE_URL", "postgres://env/db")], Some("postgres://env/db"))]
    #[case::config_wins(
        Some("postgres://config/db"),
        &[("DATABASE_URL", "postgres://env/db")],
        Some("postgres://config/db")
    )]
    #[case::reference(
        Some("os.environ/KEYS_DB"),
        &[("KEYS_DB", "postgres://ref/db")],
        Some("postgres://ref/db")
    )]
    #[case::blank_config_falls_back(
        Some(" "),
        &[("DATABASE_URL", "postgres://env/db")],
        Some("postgres://env/db")
    )]
    #[case::blank_environment(None, &[("DATABASE_URL", "")], None)]
    #[tokio::test]
    async fn resolves_the_database_url_like_the_proxy(
        #[case] configured: Option<&str>,
        #[case] environment: &[(&'static str, &'static str)],
        #[case] expected: Option<&str>,
    ) {
        let secrets = Secrets(environment.iter().copied().collect());
        let url = database_url(&config(configured), &secrets).await.unwrap();
        assert_eq!(url.as_ref().map(SecretValue::expose), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn unset_reference_fails_startup() {
        let secrets = Secrets(HashMap::new());
        assert!(matches!(
            database_url(&config(Some("os.environ/MISSING")), &secrets).await,
            Err(Error::DatabaseUrl(name)) if name == "MISSING"
        ));
    }

    #[rstest]
    #[case::unset("", false)]
    #[case::enabled(
        "litellm_settings:\n  personal_key_model_access_from_teams: true\n",
        true
    )]
    #[case::disabled(
        "litellm_settings:\n  personal_key_model_access_from_teams: false\n",
        false
    )]
    fn reads_the_personal_key_team_access_setting(#[case] settings: &str, #[case] expected: bool) {
        let config = Config::from_yaml(&format!("model_list: []\n{settings}")).unwrap();
        assert_eq!(personal_keys_use_teams(&config), expected);
    }
}
