//! What several inline test modules build the same way.

use litellm_config::Config;
use litellm_gateway_auth::{Auth, AuthenticatedRequest};
use litellm_secret_value::SecretValue;

pub(crate) const MASTER_KEY: &str = "sk-master";

/// The caller the master key authenticates.
pub(crate) async fn master() -> AuthenticatedRequest {
    let config =
        Config::from_yaml(&format!("general_settings:\n  master_key: {MASTER_KEY}\n")).unwrap();
    Auth::from_config(&config)
        .authenticate(&SecretValue::new(MASTER_KEY))
        .await
        .unwrap()
}
