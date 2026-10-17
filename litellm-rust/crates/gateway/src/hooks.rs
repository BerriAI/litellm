//! Builds the hooks and observers inference requests run through, and the per-deployment prices
//! observers that read cost need.

use std::{collections::HashSet, sync::Arc};

use litellm_config::{Config, Model};
use litellm_gateway_hooks::{
    DEFAULT_QUEUE, Facts, Hook, Observer, Observers, Pipeline, Prices, deployment_fields,
};
use litellm_gateway_inference::{Gateway, Hooks};
use litellm_llms::base_llm::model_info::ModelInfo;
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
    let reads_model_info = config.model_list.iter().any(reads_model_info);
    if hooks.is_empty() && observers.is_empty() && !reads_model_info {
        return Ok(());
    }
    let needs_cost = observers
        .iter()
        .any(|observer| observer.reads().contains(Facts::COST))
        || hooks.iter().any(|hook| hook.reads().contains(Facts::COST));
    let catalog = match needs_cost || reads_model_info {
        true => load_catalog(inference).await.map(Arc::new),
        false => None,
    };
    if let Some(catalog) = catalog.as_ref().filter(|_| reads_model_info) {
        inference.install_model_info(Arc::new(CostMap::new(catalog.clone())));
    }
    if hooks.is_empty() && observers.is_empty() {
        return Ok(());
    }
    let prices = match needs_cost {
        true => prices(config, inference, catalog.as_deref()),
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
/// `model_info` and `litellm_params` set, keyed by the router's deployment ids. An Azure
/// deployment that names neither a `base_model` nor prices of its own is priced by the model
/// its response names, as the proxy prices it.
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
        let base_model = params
            .get("base_model")
            .or_else(|| model_info.get("base_model"))
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty());
        let name = match is_azure(model) && !model.litellm_params.model.starts_with(AZURE_ROUTE) {
            true => format!("{AZURE_ROUTE}{}", model.litellm_params.model),
            false => model.litellm_params.model.clone(),
        };
        let overrides = [&model_info, &params];
        if is_azure(model)
            && base_model.is_none()
            && deployment_fields(None, &name, None, &overrides).is_empty()
        {
            prices.price_by_response_model(id.into(), AZURE_PROVIDER);
        }
        let fields = deployment_fields(catalog, &name, base_model, &overrides);
        prices.insert(id.into(), &fields);
    }
    if let Some(catalog) = catalog.filter(|_| prices.prices_by_response_model()) {
        for name in catalog
            .model_names()
            .filter(|name| name.starts_with(AZURE_ROUTE))
        {
            if let Some(found) = catalog.lookup(name) {
                prices.insert_model(name, found.entry.fields());
            }
        }
    }
    prices
}

const AZURE_PROVIDER: &str = "azure";
const AZURE_ROUTE: &str = "azure/";

fn is_azure(model: &Model) -> bool {
    match model.litellm_params.custom_llm_provider.as_deref() {
        Some(provider) => provider == AZURE_PROVIDER,
        None => model.litellm_params.model.starts_with(AZURE_ROUTE),
    }
}

/// Whether the deployment's chat param rules read model capabilities from the cost map:
/// Azure, and the Anthropic and Bedrock chat translations.
fn reads_model_info(model: &Model) -> bool {
    is_azure(model)
        || litellm_inference::provider::resolve_llm_provider(
            &model.litellm_params.model,
            model.litellm_params.custom_llm_provider.as_deref(),
            "chat completions",
        )
        .is_ok_and(|resolved| matches!(<&str>::from(resolved.provider), "anthropic" | "bedrock"))
}

/// The model cost map as the capabilities the chat param rules read.
struct CostMap {
    catalog: Arc<Catalog>,
    keys: HashSet<String>,
}

impl CostMap {
    fn new(catalog: Arc<Catalog>) -> Self {
        let keys = catalog
            .model_names()
            .filter_map(|name| catalog.lookup(name))
            .flat_map(|found| found.entry.fields().keys().cloned().collect::<Vec<_>>())
            .collect();
        Self { catalog, keys }
    }
}

impl ModelInfo for CostMap {
    fn entry(&self, model: &str) -> Option<&Map<String, Value>> {
        self.catalog.lookup(model).map(|found| found.entry.fields())
    }

    fn declares(&self, key: &str) -> bool {
        self.keys.contains(key)
    }
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
