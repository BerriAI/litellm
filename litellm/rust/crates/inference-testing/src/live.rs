use std::{future::Future, sync::Arc, time::Duration};

use litellm_http::{ClientVariant, HttpSettings, HttpSettingsLayer, Resolution};
use litellm_inference::resources::CoreResources;
use litellm_secrets::source::{EnvironmentSecrets, SecretSource};

pub const DEADLINE: Duration = Duration::from_secs(120);

pub struct LiveResources {
    pub http: litellm_http::Client,
    pub auth: Arc<litellm_auth::AuthServices>,
    pub secrets: Arc<dyn SecretSource>,
}

impl Default for LiveResources {
    fn default() -> Self {
        let resources = CoreResources::new(Arc::new(super::http_pool()));
        let settings =
            HttpSettings::from_layers([HttpSettingsLayer::from_environment(&|key: &str| {
                std::env::var(key).ok()
            })]);
        let resolution = Resolution::from(&settings);
        let http = resources
            .pool
            .client(&resolution.config, ClientVariant::Provider)
            .expect("live provider HTTP settings must be valid");
        let secrets_http = resources
            .pool
            .client(&resolution.config, ClientVariant::NoRedirect)
            .expect("live secret HTTP settings must be valid");
        Self {
            http,
            auth: resources.auth,
            secrets: Arc::new(EnvironmentSecrets::python_compatible(secrets_http)),
        }
    }
}

pub fn model(route: &str, provider: &str) -> String {
    let name = format!("LITELLM_LIVE_{route}_{provider}_MODEL").to_ascii_uppercase();
    std::env::var(&name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            panic!("set {name} to an available current model before running this live case")
        })
}

pub async fn within_deadline<T>(operation: impl Future<Output = T>) -> T {
    tokio::time::timeout(DEADLINE, operation)
        .await
        .expect("live case exceeded its 120 second deadline")
}
