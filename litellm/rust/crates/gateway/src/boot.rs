//! The gateway's router: `model_list` built through router-core's deployment build, and
//! `router_settings` read by the core. The config load resolved every `os.environ/` reference.

use std::time::Duration;

use crate::host::NativeBootHost;
use litellm_config::{Config, LiteLlmSettings, Spelled};
use litellm_inference_messages::MessagesSettings;
use litellm_python_compat::number::float;
use litellm_router_core::{BuildPath, Router, RouterConfig, build::Install};
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
pub(crate) fn configured(settings: &LiteLlmSettings) -> MessagesSettings {
    MessagesSettings {
        global_drop_params: settings.drop_params == Some(true),
        reasoning_auto_summary: settings.reasoning_auto_summary,
        request_timeout: request_timeout(settings),
        ..MessagesSettings::default()
    }
}

/// The router `config` describes.
pub(crate) async fn router(config: &Config) -> Result<Router, Error> {
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
    Ok(router)
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
    use rstest::rstest;

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
    #[case::unset("model_list: []\n", (false, false, None))]
    #[case::set(
        "litellm_settings:\n  drop_params: true\n  reasoning_auto_summary: true\n  request_timeout: 30\n",
        (true, true, Some(30.0))
    )]
    #[case::drop_params_as_text("litellm_settings:\n  drop_params: \"true\"\n", (true, false, None))]
    fn litellm_settings_reach_every_messages_call(
        #[case] yaml: &str,
        #[case] expected: (bool, bool, Option<f64>),
    ) {
        let config = Config::from_yaml(yaml).unwrap();

        let settings = super::configured(&config.litellm_settings);

        assert_eq!(
            (
                settings.global_drop_params,
                settings.reasoning_auto_summary,
                settings.request_timeout,
            ),
            expected
        );
    }
}
