//! Builds the hooks and observers inference requests run through, and the per-deployment prices
//! observers that read cost need.

use std::sync::Arc;

use litellm_config::Config;
use litellm_gateway_hooks::{
    DEFAULT_QUEUE, Facts, Hook, Observer, Observers, Pipeline, Prices, deployment_fields,
};
use litellm_gateway_inference::{Gateway, Hooks};
use litellm_model_catalog::{Catalog, Provenance};
use litellm_secrets::source::SecretSource;
use serde_json::{Map, Value};

use crate::Error;

/// A local copy of the model cost map, read instead of fetching it.
const COST_MAP_PATH: &str = "LITELLM_MODEL_COST_MAP_PATH";
/// Where to fetch the model cost map from, as the proxy's `LITELLM_MODEL_COST_MAP_URL`.
const COST_MAP_URL: &str = "LITELLM_MODEL_COST_MAP_URL";
const DEFAULT_COST_MAP_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

/// Installs `hooks` and `observers` on `inference`, in the order given. Nothing is installed when
/// both are empty, so requests skip the pipeline altogether.
pub async fn install(
    inference: &Gateway,
    config: &Config,
    hooks: Vec<Arc<dyn Hook>>,
    observers: Vec<Arc<dyn Observer>>,
) -> Result<(), Error> {
    if hooks.is_empty() && observers.is_empty() {
        return Ok(());
    }
    let needs_cost = observers
        .iter()
        .any(|observer| observer.reads().contains(Facts::COST))
        || hooks.iter().any(|hook| hook.reads().contains(Facts::COST));
    let prices = match needs_cost {
        true => {
            let catalog = load_catalog(inference).await;
            prices(config, inference, catalog.as_ref())
        }
        false => Prices::default(),
    };
    let names: Vec<_> = hooks
        .iter()
        .map(|hook| hook.name())
        .chain(observers.iter().map(|observer| observer.name()))
        .collect();
    let pipeline = Pipeline::new(
        hooks,
        Observers::start(observers, DEFAULT_QUEUE),
        Arc::new(prices),
    );
    let master_key_hash = master_key_hash(config, inference.secrets.as_ref()).await?;
    tracing::info!(
        ?names,
        priced_deployments = pipeline.prices().len(),
        "hooks installed"
    );
    inference.install_hooks(Hooks::new(Arc::new(pipeline), master_key_hash));
    Ok(())
}

/// Lets observers write out what they hold; call after the server stops taking requests.
pub async fn close(inference: &Gateway) {
    if let Some(hooks) = inference.hooks() {
        hooks.pipeline().close().await;
    }
}

/// The sha256 of the master key, which the proxy records as the key of master key requests.
async fn master_key_hash(
    config: &Config,
    secrets: &dyn SecretSource,
) -> Result<Option<String>, Error> {
    let Some(configured) = &config.general_settings.master_key else {
        return Ok(None);
    };
    let key = match configured.expose().strip_prefix("os.environ/") {
        Some(name) => secrets.get_secret_str(name).await?,
        None => Some(configured.clone()),
    };
    Ok(key.map(|key| litellm_gateway_auth::hash_token(key.expose())))
}

/// The model cost map, from a local copy or the proxy's URL. Without it only deployments that
/// set their own prices are priced.
async fn load_catalog(inference: &Gateway) -> Option<Catalog> {
    let (body, source) = match std::env::var(COST_MAP_PATH) {
        Ok(path) => match std::fs::read(&path) {
            Ok(body) => (body, path),
            Err(error) => {
                tracing::warn!(%path, %error, "cannot read the model cost map");
                return None;
            }
        },
        Err(_) => {
            let url = std::env::var(COST_MAP_URL).unwrap_or_else(|_| DEFAULT_COST_MAP_URL.into());
            match fetch(inference, &url).await {
                Ok(body) => (body, url),
                Err(error) => {
                    tracing::warn!(%url, %error, "cannot fetch the model cost map");
                    return None;
                }
            }
        }
    };
    let provenance = Provenance {
        source: Some(source),
        ..Provenance::default()
    };
    match Catalog::parse(&body, provenance) {
        Ok(catalog) => Some(catalog),
        Err(error) => {
            tracing::warn!(%error, "cannot parse the model cost map");
            None
        }
    }
}

async fn fetch(inference: &Gateway, url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let client = inference
        .resources
        .pool
        .client(&inference.http, litellm_http::ClientVariant::Provider)?;
    let response = client
        .get(url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await?
        .error_for_status()?;
    Ok(response.bytes().await?.to_vec())
}

/// Every deployment's prices: the cost map entry for its model, overlaid by the prices its
/// `model_info` and `litellm_params` set, keyed by the router's deployment ids.
fn prices(config: &Config, inference: &Gateway, catalog: Option<&Catalog>) -> Prices {
    let mut prices = Prices::default();
    for (model, id) in config
        .model_list
        .iter()
        .zip(inference.models.deployment_ids())
    {
        let model_info = json_object(model.model_info.iter());
        let params = json_object(
            model
                .litellm_params
                .extra
                .iter()
                .map(|(name, value)| (name, value.clone())),
        );
        let base_model = model_info.get("base_model").and_then(Value::as_str);
        let fields = deployment_fields(
            catalog,
            &model.litellm_params.model,
            base_model,
            &[&model_info, &params],
        );
        prices.insert(id.into(), &fields);
    }
    prices
}

fn json_object<'a, V: serde::Serialize + 'a>(
    entries: impl Iterator<Item = (&'a String, V)>,
) -> Map<String, Value> {
    entries
        .filter_map(|(name, value)| Some((name.clone(), serde_json::to_value(value).ok()?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[tokio::test]
    async fn deployment_prices_come_from_config_and_the_cost_map() {
        let config = Config::from_yaml(
            r#"
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-x
      api_key: sk
    model_info:
      id: priced-by-map
  - model_name: custom
    litellm_params:
      model: openai/my-model
      api_key: sk
      input_cost_per_token: 0.5
      output_cost_per_token: 1
"#,
        )
        .unwrap();
        let inference = crate::build_inference(&config).unwrap();
        let catalog = Catalog::parse(
            br#"{"claude-x": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}}"#,
            Provenance::default(),
        )
        .unwrap();
        let prices = prices(&config, &inference, Some(&catalog));
        let usage = litellm_gateway_hooks::Usage {
            input_tokens: 10,
            output_tokens: 1,
            ..Default::default()
        };
        assert_eq!(prices.cost("priced-by-map", &usage), Some(0.012));
        let custom = inference.models.deployment_ids().nth(1).unwrap();
        assert_eq!(prices.cost(custom, &usage), Some(6.0));
    }
}
