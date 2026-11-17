//! The gateway's `model_list`, as router-core builds it, and how the messages route shapes a call
//! to each deployment.

use std::collections::HashMap;

use litellm_config::{Config, Spelled};
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_llms::anthropic::model_capabilities::anthropic_model_capabilities;
use litellm_model_catalog::Catalog;
use litellm_python_compat::flags::lax_bool;
use litellm_router_core::{Deployment, DeploymentSpec, Router, registry::DeploymentFilter};
use serde_json::{Map, Value};

fn written(value: &impl serde::Serialize) -> Map<String, Value> {
    match serde_json::to_value(value) {
        Ok(Value::Object(fields)) => fields,
        _ => Map::new(),
    }
}

/// Each `model_list` entry as router-core reads it, its params as written.
pub(crate) fn deployments(config: &Config) -> Vec<DeploymentSpec> {
    config
        .model_list
        .iter()
        .map(|model| DeploymentSpec {
            model_name: model.model_name.clone(),
            litellm_params: written(&model.written_params),
            model_info: written(&*model.model_info),
            extra: written(&*model.additional_fields),
            provider: None,
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

fn shaping(
    deployment: &Deployment,
    catalog: &Catalog,
    drops_params_globally: bool,
) -> MessagesShaping {
    let params = &deployment.params;
    let drop_params = drops_params_globally || is_on(params.drop_params.as_ref());
    MessagesShaping {
        settings: MessagesSettings {
            drop_params,
            drops_unsupported_params: drop_params,
            additional_drop_params: additional_drop_params(deployment),
            ..MessagesSettings::default()
        },
        capabilities: anthropic_model_capabilities(
            catalog,
            &params.model,
            params.custom_llm_provider.as_deref(),
            params.extra.get("base_model").and_then(Value::as_str),
            &|capability| {
                deployment
                    .model_info
                    .raw
                    .get(capability)
                    .and_then(Value::as_bool)
            },
        ),
    }
}

/// How the messages route shapes a call to each of `router`'s deployments, by deployment id.
pub(crate) fn shapings(
    router: &Router,
    config: &Config,
    catalog: &Catalog,
) -> HashMap<String, MessagesShaping> {
    let drops_params_globally = is_on(config.litellm_settings.drop_params.as_ref());
    router
        .registry()
        .deployments(&DeploymentFilter::default())
        .iter()
        .map(|deployment| {
            (
                deployment.id.clone(),
                shaping(deployment, catalog, drops_params_globally),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use litellm_model_catalog::Catalog;
    use rstest::rstest;

    use super::*;

    fn catalog() -> Catalog {
        Catalog::parse(br#"{"provider-model": {"litellm_provider": "provider"}}"#)
            .unwrap()
            .value
    }

    fn shaped(yaml: &str) -> MessagesShaping {
        let config = Config::from_yaml(yaml).unwrap();
        let router =
            litellm_gateway_inference::model_router(&config.router_settings, deployments(&config))
                .unwrap();
        shapings(&router, &config, &catalog())
            .into_values()
            .next()
            .unwrap()
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
    fn configured_drop_params_reach_the_deployment(
        #[case] settings: &str,
        #[case] deployment: &str,
        #[case] expected: bool,
    ) {
        let settings = shaped(&format!(
            "{settings}model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model{deployment}"
        ))
        .settings;

        assert_eq!(
            (settings.drop_params, settings.drops_unsupported_params),
            (expected, expected)
        );
    }

    #[rstest]
    #[case::declared("\n    model_info:\n      supports_reasoning: true", true)]
    #[case::declared_off("\n    model_info:\n      supports_reasoning: false", false)]
    #[case::undeclared("", false)]
    fn model_info_flags_reach_the_deployment(#[case] model_info: &str, #[case] expected: bool) {
        let shaping = shaped(&format!(
            "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/claude-unlisted-9{model_info}"
        ));

        assert_eq!(shaping.capabilities.supports_reasoning, expected);
    }
}
