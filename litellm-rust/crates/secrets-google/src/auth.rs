use std::sync::Arc;

use litellm_auth_gcp::{GoogleCredentials, VertexConfig};
use litellm_auth_types::{InputSource, Sourced};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::SecretValue;

pub(crate) fn credentials(
    project: Option<String>,
    credentials: Option<SecretValue>,
    environment: Arc<dyn Lookup + Send + Sync>,
) -> GoogleCredentials {
    GoogleCredentials::new(
        VertexConfig::new(
            credentials.map(|value| Sourced::new(value, InputSource::Environment)),
            project,
            None,
        ),
        Arc::new(move |name| environment.get(name)),
    )
}
