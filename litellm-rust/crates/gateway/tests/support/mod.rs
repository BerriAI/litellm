use std::sync::Arc;

use litellm_gateway_model_catalog::{ModelCatalog, Settings};
use tokio_util::sync::CancellationToken;

/// A catalog pinned to the bundled map, so no test fetches the remote one
pub fn bundled_catalog() -> Arc<ModelCatalog> {
    let env = |name: &str| (name == "LITELLM_LOCAL_MODEL_COST_MAP").then(|| "true".to_owned());
    let catalog = ModelCatalog::boot(
        Settings::from_env(env),
        litellm_http::Client::for_test(),
        &env,
        CancellationToken::new(),
    )
    .unwrap();
    Arc::new(catalog)
}
