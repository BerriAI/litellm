mod host;
mod mcp;

use std::sync::Arc;

use litellm_config::Config;
use litellm_gateway_auth::{
    Auth, Clock, CoreAuthenticator, Error, LocalAdministrator, NoAdditionalPolicy, PythonBearer,
    SystemClock, master_key,
};
use litellm_gateway_auth_core::{HostFacts, RouteGroups, Settings};
use litellm_secrets::source::SecretSource;

pub(crate) use mcp::bind_session_owner;

use host::Host;

/// Authentication through `gateway-auth-core`, with the gateway as its host. It reads the key the
/// way the Python proxy does, so a request Python would authenticate differently is refused here.
pub async fn auth(
    config: &Config,
    secrets: &dyn SecretSource,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Auth, Error> {
    let general = &config.general_settings;
    let facts = HostFacts {
        master_key: master_key(general.master_key.as_ref(), secrets).await?,
        // TODO: `premium` needs the `LITELLM_LICENSE` check, which only Python runs, so the JWT
        // path is planned as `PremiumRequired` and never runs here
        premium: false,
        has_db: false,
        ui_login: false,
        proxy_budget: false,
        metrics_require_auth: config
            .litellm_settings
            .additional_fields
            .get("require_auth_for_metrics_endpoint")
            .and_then(|value| value.as_bool())
            != Some(false),
        // TODO: `LiteLLMRoutes` is Python's table and nothing hands it to this process yet, so no
        // built-in public or master-key-only route is known here
        route_groups: RouteGroups::default(),
    };
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let authenticator =
        CoreAuthenticator::new(Host, Settings::new(general, facts, env), clock.clone());
    Ok(Auth::new(
        Arc::new(authenticator),
        Arc::new(LocalAdministrator),
        Arc::new(NoAdditionalPolicy),
        clock,
    )
    .with_extractor(Arc::new(PythonBearer)))
}
