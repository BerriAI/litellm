use litellm_model_catalog::IntegrityLimits;
use serde_json::{Map, Value, json};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use crate::{ModelCatalog, Settings};

/// A map with one chat row per `(name, input price)`
pub(crate) fn document(prices: &[(&str, f64)]) -> Vec<u8> {
    let rows: Map<String, Value> = prices
        .iter()
        .map(|(name, price)| {
            (
                (*name).to_owned(),
                json!({"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": price}),
            )
        })
        .collect();
    serde_json::to_vec(&rows).unwrap()
}

/// `body` written to a file, with its `file://` URL. The file lives as long as the directory
pub(crate) fn file(body: &[u8]) -> (TempDir, String) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model_prices.json");
    std::fs::write(&path, body).unwrap();
    let url = format!("file://{}", path.display());
    (directory, url)
}

/// A catalog whose size check passes any non-empty map, so tests can serve small documents
pub(crate) fn catalog(remote_url: Option<String>, shutdown: CancellationToken) -> ModelCatalog {
    let settings = Settings {
        remote_url,
        limits: IntegrityLimits::new(1, 0.0).unwrap(),
    };
    ModelCatalog::boot(
        settings,
        litellm_http::Client::for_test(),
        &|_| None,
        shutdown,
    )
    .unwrap()
}

pub(crate) fn input_price(catalog: &litellm_model_catalog::Catalog, model: &str) -> Option<Value> {
    catalog
        .lookup(model)?
        .entry
        .field("input_cost_per_token")
        .cloned()
}
