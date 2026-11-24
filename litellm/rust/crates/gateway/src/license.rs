use std::time::Duration;

use litellm_config::{Config, Environment};
use litellm_http::{Client, ClientVariant};
use litellm_license::{Entitlements, LICENSE_SERVER, PublicKey, Verifier};
use litellm_secret_value::SecretValue;

use crate::Resources;

const LICENSE: &str = "LITELLM_LICENSE";
const DEFAULT_NON_LLM_CONNECTION_TIMEOUT: Duration = Duration::from_secs(15);

/// Verifies the license once at boot. The timeout is the process environment's alone, since Python reads it
/// before it loads the config.
pub(crate) async fn build_license(
    config: &Config,
    environment: &Environment,
    resources: &Resources,
) -> Entitlements {
    let timeout = environment
        .non_llm_connection_timeout
        .map_or(DEFAULT_NON_LLM_CONNECTION_TIMEOUT, Duration::from_secs);
    let verifier = Verifier::new(PublicKey::litellm(), LICENSE_SERVER.to_owned(), timeout);
    verify(&verifier, config, resources).await
}

/// `LITELLM_LICENSE` is read through `resources.environment`, where the config's `environment_variables`
/// replaces the process environment's.
async fn verify(verifier: &Verifier, config: &Config, resources: &Resources) -> Entitlements {
    entitlements(
        verifier,
        resources.environment.get(LICENSE).as_deref(),
        config
            .general_settings
            .litellm_license
            .as_ref()
            .map(SecretValue::expose),
        || {
            resources
                .core
                .pool
                .client(&resources.http, ClientVariant::Provider)
        },
    )
    .await
}

/// `general_settings.litellm_license` wins unless only the environment's license is enterprise, as in
/// `proxy_server.py::_use_config_license`
async fn entitlements(
    verifier: &Verifier,
    env_license: Option<&str>,
    config_license: Option<&str>,
    client: impl Fn() -> Result<Client, litellm_http::Error>,
) -> Entitlements {
    let from_env = match env_license {
        Some(license) => verifier.entitlements(license, &client).await,
        None => Entitlements::Community,
    };
    let Some(config_license) = config_license else {
        return from_env;
    };
    let from_config = verifier.entitlements(config_license, &client).await;
    if from_env.is_enterprise() && !from_config.is_enterprise() {
        from_env
    } else {
        from_config
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    const VOUCHED: &str = "vouched-license";
    const REFUSED: &str = "refused-license";

    async fn license_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("/verify_license/{VOUCHED}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"verify": true})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/verify_license/{REFUSED}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"verify": false})))
            .mount(&server)
            .await;
        server
    }

    #[rstest]
    #[case::no_license(None, None, Entitlements::Community)]
    #[case::env_vouched(Some(VOUCHED), None, Entitlements::ServerVerified)]
    #[case::env_refused(Some(REFUSED), None, Entitlements::Community)]
    #[case::config_vouched(None, Some(VOUCHED), Entitlements::ServerVerified)]
    #[case::config_vouched_over_env_refused(
        Some(REFUSED),
        Some(VOUCHED),
        Entitlements::ServerVerified
    )]
    #[case::env_vouched_over_config_refused(
        Some(VOUCHED),
        Some(REFUSED),
        Entitlements::ServerVerified
    )]
    #[case::both_refused(Some(REFUSED), Some(REFUSED), Entitlements::Community)]
    #[tokio::test]
    async fn the_config_license_wins_unless_only_the_env_license_is_enterprise(
        #[case] env_license: Option<&str>,
        #[case] config_license: Option<&str>,
        #[case] expected: Entitlements,
    ) {
        let server = license_server().await;
        let verifier = Verifier::new(PublicKey::litellm(), server.uri(), Duration::from_secs(5));

        let entitlements = entitlements(&verifier, env_license, config_license, || {
            Ok(Client::for_test())
        })
        .await;

        assert_eq!(entitlements, expected);
    }

    #[rstest]
    #[case::process_environment("model_list: []", REFUSED, Entitlements::Community)]
    #[case::config_replaces_the_process_environment(
        "model_list: []\nenvironment_variables:\n  LITELLM_LICENSE: vouched-license\n",
        VOUCHED,
        Entitlements::ServerVerified
    )]
    #[tokio::test]
    async fn the_environment_license_is_read_through_the_shared_resources(
        #[case] config: &str,
        #[case] asked: &str,
        #[case] expected: Entitlements,
    ) {
        crate::install_crypto_provider();
        let server = license_server().await;
        let verifier = Verifier::new(PublicKey::litellm(), server.uri(), Duration::from_secs(5));
        let config = Config::from_yaml(config).unwrap();
        let resources = Resources::over(
            &config,
            Arc::new(|name: &str| (name == LICENSE).then(|| REFUSED.to_owned())),
        )
        .unwrap();

        assert_eq!(verify(&verifier, &config, &resources).await, expected);
        let asked_paths: Vec<String> = server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|request| request.url.path().to_owned())
            .collect();
        assert_eq!(asked_paths, [format!("/verify_license/{asked}")]);
    }
}
