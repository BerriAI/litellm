use litellm_config::{Config, LiteLlmSettings};
use litellm_core_utils::settings::merge;
use litellm_gateway_inference::Gateway;
use litellm_gateway_model_catalog::{
    CatalogWriter, CostMapKeysPartial, CostMapSettings, revalidate,
};
use litellm_http::ClientVariant;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::{Error, secrets};

pub fn settings(config: &Config) -> Result<CostMapSettings, Error> {
    let environment = secrets::environment_values(&config.environment_variables)?;
    let lookup = |name: &str| secrets::environment_lookup(&environment, name);
    Ok(CostMapSettings::from_partial(merge([
        CostMapKeysPartial::from_env(&lookup),
        written(&config.litellm_settings),
    ])))
}

pub fn spawn(
    settings: &CostMapSettings,
    writer: CatalogWriter,
    inference: &Gateway,
    shutdown: CancellationToken,
) -> Result<Option<JoinHandle<()>>, Error> {
    let Some(url) = settings.remote_url() else {
        return Ok(None);
    };
    let client = inference
        .resources
        .pool
        .client(&inference.http, ClientVariant::Provider)?;
    Ok(Some(tokio::spawn(revalidate(
        writer,
        client,
        url.to_owned(),
        None,
        shutdown,
    ))))
}

fn written(settings: &LiteLlmSettings) -> CostMapKeysPartial {
    CostMapKeysPartial {
        local_only: None,
        offline: None,
        url: settings.model_cost_map_url.clone(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::environment_wins(
        "LITELLM_MODEL_COST_MAP_URL: https://environment.example.test/prices.json",
        "https://environment.example.test/prices.json"
    )]
    #[case::empty_environment_falls_through(
        "LITELLM_MODEL_COST_MAP_URL: ''",
        "https://config.example.test/prices.json"
    )]
    fn environment_overrides_typed_config(#[case] environment: &str, #[case] expected_url: &str) {
        let config = Config::from_yaml(&format!(
            "environment_variables: {{{environment}}}\nlitellm_settings: {{model_cost_map_url: https://config.example.test/prices.json}}"
        ))
        .unwrap();
        assert_eq!(settings(&config).unwrap().remote_url(), Some(expected_url));
    }

    #[rstest]
    fn missing_url_uses_shared_default_registry() {
        let config = Config::from_yaml(
            "model_list: []\nenvironment_variables: {LITELLM_MODEL_COST_MAP_URL: '', LITELLM_LOCAL_MODEL_COST_MAP: false, LITELLM_OFFLINE_MODE: false}",
        )
        .unwrap();
        let settings = settings(&config).unwrap();
        assert!(settings.remote_url().is_some());
        assert!(!settings.is_env_forced());
    }
}
