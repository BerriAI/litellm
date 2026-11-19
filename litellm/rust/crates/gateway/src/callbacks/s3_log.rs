//! The `s3_v2` callback of the gateway's config.

use std::{sync::Arc, time::Duration};

use litellm_callbacks_s3_log::{
    Constants, Host, NAME, S3Log, S3LogParts, Settings, python_backoff, system_clock,
};
use litellm_config::{Config, Environment, S3CallbackParams};
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_secrets::SecretResolver;
use litellm_tracing::Logger;

use super::{Configured, named};
use crate::{Error, Resources};

struct GatewayHost;

impl Host for GatewayHost {
    fn callback_failed(&self) {}

    fn web_identity_token(&self, _reference: &str) -> Result<String, String> {
        Err("the gateway does not read a web identity token for s3_v2 yet".to_owned())
    }
}

/// What `S3Logger()` passes the sink, from the constants the proxy reads in its environment.
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

/// `AWS_REGION`, then `AWS_REGION_NAME`, which `S3Logger()` asks for only when its params name
/// no region.
async fn region_from_environment(
    params: &S3CallbackParams,
    secrets: &SecretResolver,
) -> Result<Option<String>, Error> {
    if params.s3_region_name.is_some() {
        return Ok(None);
    }
    let region_name = secrets.get_secret_str("AWS_REGION_NAME", None).await?;
    let region = secrets.get_secret_str("AWS_REGION", None).await?;
    Ok(region
        .or(region_name)
        .map(|region| region.expose().to_owned()))
}

/// The callback of a config that names `s3_v2`, and nothing for any other.
pub(super) async fn build(
    config: &Config,
    resources: &Resources,
    environment: &Environment,
) -> Result<Option<Configured>, Error> {
    if !named(config, NAME) {
        return Ok(None);
    }
    let params = config
        .litellm_settings
        .s3_callback_params
        .clone()
        .unwrap_or_default();
    let constants = constants(environment);
    let client = |connect_timeout_seconds: f64| {
        resources.core.pool.client(
            &HttpClientConfig {
                connect_timeout: Duration::from_secs_f64(connect_timeout_seconds),
                ..resources.http.clone()
            },
            ClientVariant::Provider,
        )
    };
    let lookup = resources.environment.clone();
    let sink = S3Log::new(S3LogParts {
        settings: Settings::new(&params, &constants),
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
    Ok(Some(Configured {
        callback: sink,
        flush_every: Duration::from_secs(environment.s3_flush_interval_seconds().max(1)),
    }))
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
        let params = config(&format!("  s3_callback_params: {{ {params} }}\n"))
            .litellm_settings
            .s3_callback_params
            .unwrap();
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
}
