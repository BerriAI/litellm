//! The config's `s3_v2` callback, `S3Logger` of litellm/integrations/s3_v2.py: the
//! `litellm-callbacks-s3-log` sink Python runs behind it, built here from `litellm.s3_callback_params`
//! and the constants `S3Logger()` reads from litellm/constants.py, and flushed every
//! `DEFAULT_S3_FLUSH_INTERVAL_SECONDS` as its `_periodic_flush` does, and at shutdown.
//!
//! What only the Python host answers is named by [`python_hosted`], so admit sends a config that
//! sets it to Python: a web identity token, which Python reads through `get_secret`, the base64
//! stripping `S3StripBase64Files` does to the payload before the sink sees it, and `s3_verify`,
//! which Python's `call_config` builds the sink's client with. Cold storage, the payload's
//! excluded fields and the `no-log` opt-outs are settings, body keys and key metadata admit
//! already sends to Python, so the sink here is built with no cold storage.

use std::{sync::Arc, time::Duration};

use litellm_callbacks::ErasedCallback;
use litellm_callbacks_s3_log::{
    Constants, Host, NAME, S3Log, S3LogParts, Settings, python_backoff, system_clock,
};
use litellm_config::{Config, Environment};
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_python_compat::{json::from_json, truthy::truthy};
use litellm_secrets::SecretResolver;
use litellm_tracing::Logger;
use serde_json::{Map, Value};

use super::Registered;
use crate::{Error, Resources};

/// The `litellm_settings` key `S3Logger()` reads its params from.
pub const PARAMS: &str = "s3_callback_params";
/// The variable the sink's credentials fall back to for a web identity token, which it asks its
/// host to resolve.
pub const WEB_IDENTITY_TOKEN_VARIABLE: &str = "AWS_WEB_IDENTITY_TOKEN";

/// How the gateway answers what the sink asks its host. Python counts a failed upload with
/// `handle_callback_failure`, which only a Prometheus callback reads, and admit sends a config
/// with one to Python. A web identity token is never asked for, since admit sends a config that
/// names one to Python.
struct GatewayHost;

impl Host for GatewayHost {
    fn callback_failed(&self) {}

    fn web_identity_token(&self, _reference: &str) -> Result<String, String> {
        Err("the gateway does not resolve a web identity token for s3_v2".to_owned())
    }
}

/// Whether the config's `callbacks` names `s3_v2`, and how often.
fn named(config: &Config) -> usize {
    config
        .litellm_settings
        .callbacks
        .iter()
        .flatten()
        .filter(|callback| callback.as_str() == Some(NAME))
        .count()
}

/// `litellm.s3_callback_params` as `_resolved_params` reads it, an empty mapping when it is unset.
/// `None` when it is set to something that is not a mapping, which `S3Logger()` raises on.
fn params(config: &Config) -> Option<Map<String, Value>> {
    match config.litellm_settings.additional_fields.get(PARAMS) {
        None => Some(Map::new()),
        Some(written) => match serde_json::to_value(written).ok()? {
            Value::Object(params) => Some(params),
            Value::Null => Some(Map::new()),
            _ => None,
        },
    }
}

/// Whether `value` of the param `key` asks for something only the Python host answers.
fn hosted_by_python(key: &str, value: &Value) -> bool {
    match key {
        "s3_verify" => !value.is_null(),
        _ => truthy(&from_json(value.clone())),
    }
}

/// The params only the Python host answers. See the module doc.
const PYTHON_HOSTED: [&str; 3] = [
    "s3_aws_web_identity_token",
    "s3_strip_base64_files",
    "s3_verify",
];

/// What of the config's `s3_v2` settings only the Python host answers, named as the config
/// spells it: `s3_v2` named more than once, params that are not a mapping, a param of
/// [`PYTHON_HOSTED`], or, for a config that names `s3_v2`, a web identity token the environment
/// holds.
pub(crate) fn python_hosted(config: &Config, environment: &Environment) -> Option<Box<str>> {
    if named(config) > 1 {
        return Some("litellm_settings.callbacks".into());
    }
    let Some(params) = params(config) else {
        return Some(format!("litellm_settings.{PARAMS}").into());
    };
    if let Some(key) = PYTHON_HOSTED.iter().find(|key| {
        params
            .get(**key)
            .is_some_and(|value| hosted_by_python(key, value))
    }) {
        return Some(format!("litellm_settings.{PARAMS}.{key}").into());
    }
    (named(config) == 1 && environment.aws_web_identity_token.is_some())
        .then(|| WEB_IDENTITY_TOKEN_VARIABLE.into())
}

/// What `S3Logger()` hands the sink of the constants litellm/constants.py reads from the
/// environment, as `_constants` builds them.
fn constants(environment: &Environment) -> Constants {
    let defaults = Constants::default();
    let count = |value: Option<u64>, default: usize| {
        value
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(default)
    };
    Constants {
        batch_size: count(Some(environment.s3_batch_size()), defaults.batch_size),
        max_concurrent_uploads: count(
            environment.default_s3_max_concurrent_uploads,
            defaults.max_concurrent_uploads,
        ),
        max_adaptive_concurrency: count(
            environment.default_s3_max_adaptive_concurrency,
            defaults.max_adaptive_concurrency,
        ),
        upload_attempt_timeout_seconds: environment
            .s3_upload_attempt_timeout()
            .map_or(defaults.upload_attempt_timeout_seconds, |seconds| {
                seconds as f64
            }),
        ..defaults
    }
}

/// `_region_from_environment`: `AWS_REGION`, then `AWS_REGION_NAME`, which `S3Logger()` asks for
/// only when its params name no region.
async fn region_from_environment(
    params: &Map<String, Value>,
    secrets: &SecretResolver,
) -> Result<Option<String>, Error> {
    if params
        .get("s3_region_name")
        .is_some_and(|region| truthy(&from_json(region.clone())))
    {
        return Ok(None);
    }
    let region_name = secrets.get_secret_str("AWS_REGION_NAME", None).await?;
    let region = secrets.get_secret_str("AWS_REGION", None).await?;
    Ok(region
        .or(region_name)
        .map(|region| region.expose().to_owned()))
}

/// The `s3_v2` callback of a config that names it, flushed on its interval, and nothing for any
/// other config.
pub(crate) async fn build(
    config: &Config,
    resources: &Resources,
    environment: &Environment,
) -> Result<Option<Registered>, Error> {
    if named(config) == 0 {
        return Ok(None);
    }
    let params = params(config).unwrap_or_default();
    let constants = constants(environment);
    let client = |connect_timeout_seconds: f64| {
        resources.core.pool.client(
            &HttpClientConfig {
                connect_timeout: Duration::from_secs_f64(connect_timeout_seconds.max(0.0)),
                ..resources.http.clone()
            },
            ClientVariant::Provider,
        )
    };
    let lookup = resources.environment.clone();
    let sink = S3Log::new(S3LogParts {
        settings: Settings::from_params(&params, &constants),
        client: client(constants.upload_attempt_timeout_seconds)?,
        download_client: client(constants.download_connect_timeout_seconds)?,
        environment: Arc::new(move |name: &str| lookup.get(name)),
        clock: system_clock,
        backoff: Arc::new(python_backoff),
        host: Arc::new(GatewayHost),
        cold_storage_is_s3_v2: false,
        region_from_environment: region_from_environment(&params, &resources.secrets).await?,
        logger: Logger::current(),
    });
    let every = Duration::from_secs(environment.s3_flush_interval_seconds().max(1));
    Ok(Some(Registered::flushed_every(
        sink as Arc<dyn ErasedCallback>,
        every,
    )))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use litellm_secrets::{OidcResolver, SecretManagerState};
    use rstest::rstest;

    use super::*;

    fn config(litellm_settings: &str) -> Config {
        Config::from_yaml(&format!(
            "model_list: []\nlitellm_settings:\n{litellm_settings}"
        ))
        .unwrap()
    }

    fn secrets(variables: &[(&str, &str)]) -> SecretResolver {
        let variables: BTreeMap<String, String> = variables
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        SecretResolver::new_python_compatible(
            Arc::new(SecretManagerState::default()),
            Arc::new(move |name: &str| variables.get(name).cloned()),
            OidcResolver::new(litellm_http::Client::for_test()),
        )
    }

    #[rstest]
    fn the_constants_are_the_environment_ones_over_the_defaults() {
        let environment = Environment {
            default_s3_batch_size: Some(64),
            default_s3_max_concurrent_uploads: Some(4),
            default_s3_max_adaptive_concurrency: Some(50),
            s3_upload_attempt_timeout_seconds: Some(45),
            ..Environment::default()
        };
        assert_eq!(
            (constants(&environment), constants(&Environment::default())),
            (
                Constants {
                    batch_size: 64,
                    max_concurrent_uploads: 4,
                    max_adaptive_concurrency: 50,
                    upload_attempt_timeout_seconds: 45.0,
                    ..Constants::default()
                },
                Constants::default()
            )
        );
    }

    #[rstest]
    #[case::aws_region_wins(
        "",
        &[("AWS_REGION", "us-east-2"), ("AWS_REGION_NAME", "eu-west-1")],
        Some("us-east-2")
    )]
    #[case::aws_region_name_alone("", &[("AWS_REGION_NAME", "eu-west-1")], Some("eu-west-1"))]
    #[case::the_params_region_is_enough(
        "s3_region_name: ap-south-1",
        &[("AWS_REGION", "us-east-2")],
        None
    )]
    #[case::an_empty_params_region_does_not_count(
        "s3_region_name: ''",
        &[("AWS_REGION", "us-east-2")],
        Some("us-east-2")
    )]
    #[case::nothing_names_a_region("", &[], None)]
    #[tokio::test]
    async fn the_region_is_asked_for_only_when_the_params_name_none(
        #[case] params: &str,
        #[case] variables: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let params = super::params(&config(&format!("  {PARAMS}: {{ {params} }}\n"))).unwrap();
        let region = region_from_environment(&params, &secrets(variables))
            .await
            .unwrap();
        assert_eq!(region.as_deref(), expected);
    }

    #[rstest]
    #[case::named("  callbacks: [\"s3_v2\"]\n", true)]
    #[case::params_without_the_callback("  s3_callback_params:\n    s3_bucket_name: logs\n", false)]
    #[tokio::test]
    async fn the_callback_is_built_only_for_a_config_that_names_it(
        #[case] settings: &str,
        #[case] expected: bool,
    ) {
        let config = config(settings);
        let resources = Resources::build(&config).unwrap();
        let built = build(&config, &resources, &Environment::default())
            .await
            .unwrap();
        assert_eq!(built.is_some(), expected);
    }

    #[rstest]
    #[case::no_params("  callbacks: [s3_v2]\n", None)]
    #[case::params_the_sink_reads(
        "  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n    s3_region_name: us-east-1\n    s3_use_key_prefix: true\n    s3_log_prompts_only: true\n    s3_batch_file_upload: true\n",
        None
    )]
    #[case::stripping_written_off(
        "  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_strip_base64_files: false\n",
        None
    )]
    #[case::a_web_identity_token(
        "  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_aws_web_identity_token: oidc/google/aud\n",
        Some("litellm_settings.s3_callback_params.s3_aws_web_identity_token")
    )]
    #[case::stripping_base64_files(
        "  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_strip_base64_files: true\n",
        Some("litellm_settings.s3_callback_params.s3_strip_base64_files")
    )]
    #[case::tls_verification(
        "  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_verify: false\n",
        Some("litellm_settings.s3_callback_params.s3_verify")
    )]
    #[case::params_that_are_no_mapping(
        "  callbacks: [s3_v2]\n  s3_callback_params: [s3_bucket_name]\n",
        Some("litellm_settings.s3_callback_params")
    )]
    #[case::named_twice("  callbacks: [s3_v2, s3_v2]\n", Some("litellm_settings.callbacks"))]
    fn what_only_the_python_host_answers_is_named(
        #[case] settings: &str,
        #[case] named: Option<&str>,
    ) {
        assert_eq!(
            python_hosted(&config(settings), &Environment::default()).as_deref(),
            named
        );
    }

    #[rstest]
    #[case::named("  callbacks: [s3_v2]\n", Some(WEB_IDENTITY_TOKEN_VARIABLE))]
    #[case::not_named("  callbacks: [dynamic_rate_limiter_v3]\n", None)]
    fn a_web_identity_token_in_the_environment_is_named_only_for_s3_v2(
        #[case] settings: &str,
        #[case] named: Option<&str>,
    ) {
        let environment = Environment {
            aws_web_identity_token: Some("oidc/google/aud".to_owned()),
            ..Environment::default()
        };
        assert_eq!(
            python_hosted(&config(settings), &environment).as_deref(),
            named
        );
    }
}
