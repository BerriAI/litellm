//! The gateway's router: `model_list` built through router-core's deployment build, and
//! `router_settings` read by the core. The config load resolved every `os.environ/` reference.

use std::collections::HashMap;

use crate::host::NativeBootHost;
use litellm_config::{Config, Spelled};
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_llms::anthropic::model_capabilities::messages_route_capabilities;
use litellm_model_catalog::Catalog;
use litellm_python_compat::flags::lax_bool;
use litellm_router_core::{
    BuildPath, Deployment, Router, RouterConfig, build::Install, registry::DeploymentFilter,
};
use serde_json::{Map, Value};

use crate::Error;

fn written(value: &impl serde::Serialize) -> Map<String, Value> {
    match serde_json::to_value(value) {
        Ok(Value::Object(fields)) => fields,
        _ => Map::new(),
    }
}

fn entries(config: &Config) -> Vec<Map<String, Value>> {
    config
        .model_list
        .iter()
        .map(|model| {
            let params = written(&model.written_params);
            written(&*model.additional_fields)
                .into_iter()
                .chain([
                    (
                        "model_name".to_owned(),
                        Value::from(model.model_name.as_str()),
                    ),
                    ("litellm_params".to_owned(), Value::Object(params)),
                    (
                        "model_info".to_owned(),
                        Value::Object(written(&*model.model_info)),
                    ),
                ])
                .collect()
        })
        .collect()
}

fn is_on(flag: Option<&Spelled<bool>>) -> bool {
    match flag {
        Some(Spelled::Value(on)) => *on,
        Some(Spelled::Text(text)) => lax_bool(&text.as_str().into()) == Some(true),
        None => false,
    }
}

fn additional_drop_params(deployment: &Deployment) -> Vec<String> {
    deployment
        .params
        .extra
        .get("additional_drop_params")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// How the messages route shapes a call to `deployment`: its `drop_params`, and the
/// capabilities the catalog and its `model_info` declare.
fn shaping(
    deployment: &Deployment,
    catalog: &Catalog,
    drops_params_globally: bool,
) -> MessagesShaping {
    let params = &deployment.params;
    let drop_params = drops_params_globally || is_on(params.drop_params.as_ref());
    let base_model = params.extra.get("base_model").and_then(Value::as_str);
    MessagesShaping {
        settings: MessagesSettings {
            drop_params,
            drops_unsupported_params: drop_params,
            additional_drop_params: additional_drop_params(deployment),
            base_model: base_model.map(str::to_owned),
            ..MessagesSettings::default()
        },
        capabilities: messages_route_capabilities(
            catalog,
            &params.model,
            params.custom_llm_provider.as_deref(),
            base_model,
            &|capability| {
                deployment
                    .model_info
                    .raw
                    .get(capability)
                    .and_then(Value::as_bool)
            },
        ),
        ..MessagesShaping::default()
    }
}

/// The router `config` describes, and how the messages route shapes a call to each of its
/// deployments, by id.
pub(crate) async fn router(
    config: &Config,
    catalog: &Catalog,
) -> Result<(Router, HashMap<String, MessagesShaping>), Error> {
    let (settings, dynamic) = RouterConfig::from_settings(&config.router_settings)?;
    let router = Router::new(settings, dynamic)?;
    router
        .build(
            &NativeBootHost,
            BuildPath::Create,
            entries(config),
            true,
            Install::Replace,
        )
        .await?;
    let drops_params_globally = config.litellm_settings.drop_params == Some(true);
    let shaping = router
        .registry()
        .deployments(&DeploymentFilter::default())
        .iter()
        .map(|deployment| {
            (
                deployment.id.clone(),
                shaping(deployment, catalog, drops_params_globally),
            )
        })
        .collect();
    Ok((router, shaping))
}

/// The router `config` describes with no host behind its build, for a test that asks it only what
/// its settings and model groups say.
#[cfg(test)]
pub(crate) fn unhosted(config: &Config) -> Router {
    let (settings, dynamic) = RouterConfig::from_settings(&config.router_settings).unwrap();
    let model_list = entries(config)
        .into_iter()
        .map(|entry| serde_json::from_value(Value::Object(entry)).unwrap())
        .collect();
    let settings = RouterConfig {
        model_list,
        ..settings
    };
    Router::new(settings, dynamic).unwrap()
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use litellm_model_catalog::Catalog;
    use rstest::rstest;

    fn catalog() -> Catalog {
        Catalog::parse(br#"{"provider-model": {"litellm_provider": "provider"}}"#)
            .unwrap()
            .value
    }

    async fn built(
        yaml: &str,
    ) -> (
        litellm_router_core::Router,
        super::HashMap<String, super::MessagesShaping>,
    ) {
        let config = Config::from_yaml(yaml).unwrap();
        super::router(&config, &catalog()).await.unwrap()
    }

    #[rstest]
    #[case::unset("", "", false)]
    #[case::deployment_flag("", "\n      drop_params: true", true)]
    #[case::deployment_word("", "\n      drop_params: \" True \"", true)]
    #[case::deployment_off("", "\n      drop_params: false", false)]
    #[case::deployment_unknown_word("", "\n      drop_params: maybe", false)]
    #[case::deployment_yes("", "\n      drop_params: yes", true)]
    #[case::deployment_no("", "\n      drop_params: \"no\"", false)]
    #[case::global_flag("litellm_settings:\n  drop_params: true\n", "", true)]
    #[case::global_word("litellm_settings:\n  drop_params: \"true\"\n", "", true)]
    #[case::global_off("litellm_settings:\n  drop_params: false\n", "", false)]
    #[case::global_over_deployment_off(
        "litellm_settings:\n  drop_params: true\n",
        "\n      drop_params: false",
        true
    )]
    #[tokio::test]
    async fn configured_drop_params_reach_the_deployment(
        #[case] settings: &str,
        #[case] deployment: &str,
        #[case] expected: bool,
    ) {
        let (_, shaping) = built(&format!(
            "{settings}model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model{deployment}"
        ))
        .await;
        let settings = &shaping.values().next().unwrap().settings;

        assert_eq!(
            (settings.drop_params, settings.drops_unsupported_params),
            (expected, expected)
        );
    }

    #[rstest]
    #[case::declared("\n    model_info:\n      supports_reasoning: true", true)]
    #[case::declared_off("\n    model_info:\n      supports_reasoning: false", false)]
    #[case::undeclared("", false)]
    #[tokio::test]
    async fn model_info_flags_reach_the_deployment(
        #[case] model_info: &str,
        #[case] expected: bool,
    ) {
        let (_, shaping) = built(&format!(
            "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/claude-unlisted-9{model_info}"
        ))
        .await;

        assert_eq!(
            shaping
                .values()
                .next()
                .unwrap()
                .capabilities
                .supports_reasoning,
            expected
        );
    }
}
