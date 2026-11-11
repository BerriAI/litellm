mod host;
mod mcp;

use std::sync::Arc;

use litellm_config::Config;
use litellm_gateway_auth::{
    Auth, Clock, CoreAuthenticator, LocalAdministrator, NoAdditionalPolicy, SystemClock,
};
use litellm_gateway_auth_core::{HostFacts, Settings};
use litellm_secret_value::SecretValue;
use litellm_secrets::source::SecretSource;

pub(crate) use mcp::bind_session_owner;

use crate::Error;
use host::Host;

/// Authentication through `gateway-auth-core`, with the gateway as its host.
pub async fn auth(
    config: &Config,
    secrets: &dyn SecretSource,
    env: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Auth, Error> {
    let master_key =
        litellm_gateway_auth::master_key(config.general_settings.master_key.as_ref(), secrets)
            .await?;
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    Ok(Auth::new(
        Arc::new(CoreAuthenticator::new(
            Host,
            settings(config, master_key, env),
            clock.clone(),
        )),
        Arc::new(LocalAdministrator),
        Arc::new(NoAdditionalPolicy),
        clock,
    ))
}

pub(crate) fn settings(
    config: &Config,
    master_key: Option<SecretValue>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Settings {
    let facts = HostFacts {
        master_key,
        // TODO: `premium` needs the `LITELLM_LICENSE` check, which only Python runs
        premium: false,
        has_db: false,
        ui_login: false,
        proxy_budget: false,
        metrics_require_auth: metrics_require_auth(config),
    };
    Settings::new(&config.general_settings, facts, env)
}

fn metrics_require_auth(config: &Config) -> bool {
    config
        .litellm_settings
        .additional_fields
        .get("require_auth_for_metrics_endpoint")
        .and_then(|value| value.as_bool())
        != Some(false)
}
