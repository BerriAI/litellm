use std::time::Duration;

use litellm_config::{Config, Environment};
use litellm_http::{Client, ClientVariant, HttpClientConfig, HttpClientPool};
use litellm_license::{Entitlements, LICENSE_SERVER, PublicKey, Verifier};
use litellm_secret_value::SecretValue;

use crate::Error;

const LICENSE: &str = "LITELLM_LICENSE";
const DEFAULT_NON_LLM_CONNECTION_TIMEOUT: Duration = Duration::from_secs(15);

/// Verifies the license once at boot. `environment_variables.LITELLM_LICENSE` replaces the process
/// environment's, as the proxy's config load does.
pub async fn build_license(
    config: &Config,
    environment: &Environment,
    pool: &HttpClientPool,
    http: &HttpClientConfig,
) -> Result<Entitlements, Error> {
    let config_environment =
        litellm_config_load::environment_values(&config.environment_variables)?;
    let env_license = config_environment
        .get(LICENSE)
        .or(environment.litellm_license.as_ref());
    let timeout = environment
        .non_llm_connection_timeout
        .map_or(DEFAULT_NON_LLM_CONNECTION_TIMEOUT, Duration::from_secs);
    let verifier = Verifier::new(PublicKey::litellm(), LICENSE_SERVER.to_owned(), timeout);
    Ok(entitlements(
        &verifier,
        env_license.map(SecretValue::expose),
        config
            .general_settings
            .litellm_license
            .as_ref()
            .map(SecretValue::expose),
        || pool.client(http, ClientVariant::Provider),
    )
    .await)
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
}
