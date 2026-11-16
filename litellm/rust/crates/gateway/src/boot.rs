//! The gateway's router: `model_list` built through router-core's deployment build, with the
//! environment and the model catalog as the registry host, and `router_settings` read by the core.

use std::sync::Arc;

use crate::host::{FACTS, NativeBootHost, environment_value};
use litellm_config::{Config, ENV_REFERENCE_PREFIX, Spelled};
use litellm_core_utils::{serde_compat::parse_pydantic_bool, settings::Lookup};
use litellm_gateway_inference::ModelTable;
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_llms::anthropic::model_capabilities::anthropic_model_capabilities;
use litellm_model_catalog::CostMap;
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

/// `_check_for_os_environ_vars`, which the proxy runs over the config before any router sees it:
/// a reference nested in an object, or in an object inside a list, resolves too. Top-level
/// references are left to the router's build, which resolves them as `set_model_list` does.
fn nested_resolved(value: &Value, env: &impl Lookup) -> Value {
    let resolved = |fields: &Map<String, Value>| {
        Value::Object(
            fields
                .iter()
                .map(|(name, value)| (name.clone(), resolved_value(value, env)))
                .collect(),
        )
    };
    match value {
        Value::Object(fields) => resolved(fields),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| match item {
                    Value::Object(fields) => resolved(fields),
                    _ => item.clone(),
                })
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn resolved_value(value: &Value, env: &impl Lookup) -> Value {
    match value {
        Value::String(text) if text.starts_with(ENV_REFERENCE_PREFIX) => {
            environment_value(text, env)
        }
        _ => nested_resolved(value, env),
    }
}

fn entries(config: &Config, env: &impl Lookup) -> Vec<Map<String, Value>> {
    config
        .model_list
        .iter()
        .map(|model| {
            let params = written(&model.written_params)
                .into_iter()
                .map(|(name, value)| (name, nested_resolved(&value, env)))
                .collect();
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

fn is_on(flag: Option<&Spelled<bool>>, env: &impl Lookup) -> bool {
    match flag {
        Some(Spelled::Value(on)) => *on,
        Some(Spelled::Text(text)) => match text.strip_prefix(ENV_REFERENCE_PREFIX) {
            Some(name) => env
                .get(name)
                .is_some_and(|value| parse_pydantic_bool(&value) == Some(true)),
            None => parse_pydantic_bool(text) == Some(true),
        },
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

/// What the messages route needs of each deployment that the cost map does not say: its
/// `drop_params` settings, read once at boot.
pub(crate) struct Shaper {
    deployments: Vec<(Arc<Deployment>, MessagesSettings)>,
}

impl Shaper {
    fn new(router: &Router, config: &Config, env: &impl Lookup) -> Self {
        let drops_params_globally = is_on(config.litellm_settings.drop_params.as_ref(), env);
        let deployments = router
            .registry()
            .deployments(&DeploymentFilter::default())
            .into_iter()
            .map(|deployment| {
                let drop_params =
                    drops_params_globally || is_on(deployment.params.drop_params.as_ref(), env);
                let settings = MessagesSettings {
                    drop_params,
                    drops_unsupported_params: drop_params,
                    additional_drop_params: additional_drop_params(&deployment),
                    ..MessagesSettings::default()
                };
                (deployment, settings)
            })
            .collect();
        Self { deployments }
    }

    /// The model table `cost_map` gives: how the messages route shapes a call to each deployment,
    /// with the capabilities the catalog and its `model_info` declare.
    pub(crate) fn table(&self, cost_map: CostMap) -> ModelTable {
        let shaping = self
            .deployments
            .iter()
            .map(|(deployment, settings)| {
                let params = &deployment.params;
                let capabilities = anthropic_model_capabilities(
                    cost_map.catalog(),
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
                );
                let shaping = MessagesShaping {
                    settings: settings.clone(),
                    capabilities,
                };
                (deployment.id.clone(), shaping)
            })
            .collect();
        ModelTable { cost_map, shaping }
    }
}

/// The router `config` describes, and what the messages route needs of each of its deployments.
pub(crate) async fn router(
    config: &Config,
    env: &(impl Lookup + Sync),
) -> Result<(Router, Shaper), Error> {
    let (settings, dynamic) = RouterConfig::from_settings(&config.router_settings)?;
    let router = Router::new(settings, dynamic, FACTS)?;
    router
        .build(
            &NativeBootHost { env },
            BuildPath::Create,
            entries(config, env),
            true,
            Install::Replace,
        )
        .await?;
    let shaper = Shaper::new(&router, config, env);
    Ok((router, shaper))
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use litellm_model_catalog::{CostMap, IntegrityLimits};
    use rstest::rstest;
    use serde_json::{Value, json};

    fn environment(name: &str) -> Option<String> {
        match name {
            "PROMPT" => Some("Answer in French.".to_owned()),
            "ON_WORD" => Some(" True ".to_owned()),
            "OFF_WORD" => Some("false".to_owned()),
            "DROP_PARAMS_FLAG" => Some(" Yes ".to_owned()),
            _ => None,
        }
    }

    async fn built(
        yaml: &str,
    ) -> (
        litellm_router_core::Router,
        std::collections::HashMap<String, super::MessagesShaping>,
    ) {
        let config = Config::from_yaml(yaml).unwrap();
        let (router, shaper) = super::router(&config, &environment).await.unwrap();
        let cost_map = CostMap::read(
            br#"{"provider-model": {"litellm_provider": "provider"}}"#,
            None,
            IntegrityLimits::UNCHECKED,
        )
        .unwrap();
        (router, shaper.table(cost_map).shaping)
    }

    #[rstest]
    #[case::unset("", "", false)]
    #[case::deployment_flag("", "\n      drop_params: true", true)]
    #[case::deployment_word("", "\n      drop_params: \" True \"", true)]
    #[case::deployment_off("", "\n      drop_params: false", false)]
    #[case::deployment_unknown_word("", "\n      drop_params: maybe", false)]
    #[case::deployment_yes("", "\n      drop_params: yes", true)]
    #[case::deployment_no("", "\n      drop_params: \"no\"", false)]
    #[case::deployment_env_reference("", "\n      drop_params: os.environ/DROP_PARAMS_FLAG", true)]
    #[case::deployment_unset_env_reference(
        "",
        "\n      drop_params: os.environ/MISSING_FLAG",
        false
    )]
    #[case::global_env_reference(
        "litellm_settings:\n  drop_params: os.environ/DROP_PARAMS_FLAG\n",
        "",
        true
    )]
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
    #[case::text(json!("os.environ/PROMPT"), json!("Answer in French."))]
    #[case::true_word(json!("os.environ/ON_WORD"), json!(true))]
    #[case::false_word(json!("os.environ/OFF_WORD"), json!(false))]
    #[case::unset(json!("os.environ/UNSET"), Value::Null)]
    #[case::plain(json!("Answer in French."), json!("Answer in French."))]
    #[case::nested_object(json!({"inner": "os.environ/PROMPT"}), json!({"inner": "Answer in French."}))]
    #[case::object_in_a_list(
        json!([{"authorization_token": "os.environ/PROMPT"}]),
        json!([{"authorization_token": "Answer in French."}])
    )]
    #[case::text_in_a_list(json!(["os.environ/PROMPT"]), json!(["os.environ/PROMPT"]))]
    #[tokio::test]
    async fn deployment_params_resolve_os_environ_references_as_the_proxy_does(
        #[case] configured: Value,
        #[case] expected: Value,
    ) {
        let (router, _) = built(&format!(
            "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model\n      system: {configured}"
        ))
        .await;

        assert_eq!(
            router.registry().group("public-model").unwrap()[0]
                .params
                .extra["system"],
            expected
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
