//! The gateway's router: `model_list` built through router-core's deployment build, and
//! `router_settings` read by the core. The config load resolved every `os.environ/` reference.

use std::{collections::HashMap, time::Duration};

use crate::host::NativeBootHost;
use litellm_config::{Config, LiteLlmSettings, Spelled};
use litellm_gateway_inference::messages::drop_paths;
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_llms::anthropic::model_capabilities::messages_route_capabilities;
use litellm_model_catalog::Catalog;
use litellm_python_compat::{flags::lax_bool, number::float};
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

/// `litellm_settings.request_timeout` as Python's `float()` reads it, when the native call keeps
/// it as a timeout: zero and a value no `Duration` holds are none.
pub(crate) fn request_timeout(settings: &LiteLlmSettings) -> Option<f64> {
    let seconds = match settings.request_timeout.as_ref()? {
        Spelled::Value(seconds) => *seconds,
        Spelled::Text(text) => float(text)?,
    };
    Duration::try_from_secs_f64(seconds)
        .is_ok_and(|timeout| !timeout.is_zero())
        .then_some(seconds)
}

/// What `litellm_settings` sets for a messages call to any deployment.
fn configured(settings: &LiteLlmSettings) -> MessagesSettings {
    MessagesSettings {
        global_drop_params: settings.drop_params == Some(true),
        reasoning_auto_summary: settings.reasoning_auto_summary,
        request_timeout: request_timeout(settings),
        ..MessagesSettings::default()
    }
}

/// How the messages route shapes a call to `deployment`: the `configured` settings under its own
/// `drop_params` and `additional_drop_params`, and the capabilities the catalog and its
/// `model_info` declare.
fn shaping(
    deployment: &Deployment,
    catalog: &Catalog,
    configured: &MessagesSettings,
) -> MessagesShaping {
    let params = &deployment.params;
    let drop_params = configured.global_drop_params || is_on(params.drop_params.as_ref());
    let base_model = params.extra.get("base_model").and_then(Value::as_str);
    MessagesShaping {
        settings: MessagesSettings {
            drop_params,
            drops_unsupported_params: drop_params,
            additional_drop_params: params
                .extra
                .get("additional_drop_params")
                .map(drop_paths)
                .unwrap_or_default(),
            base_model: base_model.map(str::to_owned),
            ..configured.clone()
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
    let configured = configured(&config.litellm_settings);
    let shaping = router
        .registry()
        .deployments(&DeploymentFilter::default())
        .iter()
        .map(|deployment| {
            (
                deployment.id.clone(),
                shaping(deployment, catalog, &configured),
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
    #[case::unset("model_list: []\n", None)]
    #[case::seconds("litellm_settings:\n  request_timeout: 30\n", Some(30.0))]
    #[case::text_float_reads("litellm_settings:\n  request_timeout: \" 2.5 \"\n", Some(2.5))]
    #[case::text_float_refuses("litellm_settings:\n  request_timeout: soon\n", None)]
    #[case::zero("litellm_settings:\n  request_timeout: 0\n", None)]
    #[case::negative("litellm_settings:\n  request_timeout: -1\n", None)]
    fn a_request_timeout_is_read_as_float_reads_it(
        #[case] yaml: &str,
        #[case] expected: Option<f64>,
    ) {
        let config = Config::from_yaml(yaml).unwrap();

        assert_eq!(super::request_timeout(&config.litellm_settings), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn litellm_settings_reach_every_deployment() {
        let (_, shaping) = built(
            "litellm_settings:\n  drop_params: true\n  reasoning_auto_summary: true\n  request_timeout: 30\nmodel_list:\n  - model_name: a\n    litellm_params:\n      model: provider/a\n  - model_name: b\n    litellm_params:\n      model: provider/b\n      drop_params: false\n",
        )
        .await;
        let read: Vec<_> = shaping
            .values()
            .map(|shaping| {
                let settings = &shaping.settings;
                (
                    settings.global_drop_params,
                    settings.reasoning_auto_summary,
                    settings.request_timeout,
                )
            })
            .collect();

        assert_eq!(read, [(true, true, Some(30.0)); 2]);
    }

    #[rstest]
    #[case::a_list("[metadata.user_id, 1, thinking]", &["metadata.user_id", "thinking"])]
    #[case::text("metadata.user_id", &[])]
    #[case::a_map("{metadata: user_id}", &[])]
    #[tokio::test]
    async fn a_deployments_additional_drop_params_are_the_text_entries_of_its_list(
        #[case] written: &str,
        #[case] expected: &[&str],
    ) {
        let (_, shaping) = built(&format!(
            "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model\n      additional_drop_params: {written}"
        ))
        .await;

        assert_eq!(
            shaping
                .values()
                .next()
                .unwrap()
                .settings
                .additional_drop_params,
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
