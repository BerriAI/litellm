use std::time::Duration;

use litellm_config::Config;
use litellm_http::{Client, ClientVariant, HttpClientConfig, HttpClientPool};
use litellm_license::{Entitlements, LICENSE_SERVER, PublicKey, Verifier};
use litellm_secret_value::SecretValue;

use crate::{Error, secrets};

const LICENSE: &str = "LITELLM_LICENSE";

pub async fn build_license(
    config: &Config,
    timeout: Duration,
    pool: &HttpClientPool,
    http: &HttpClientConfig,
) -> Result<Entitlements, Error> {
    let environment = secrets::environment_values(&config.environment_variables)?;
    let env_license = environment
        .get(LICENSE)
        .map(|value| value.expose().to_owned())
        .or_else(|| std::env::var(LICENSE).ok());
    let verifier = Verifier::new(PublicKey::litellm(), LICENSE_SERVER.to_owned(), timeout);
    Ok(entitlements(
        &verifier,
        env_license.as_deref(),
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
        for (license, verify) in [(VOUCHED, true), (REFUSED, false)] {
            Mock::given(method("GET"))
                .and(path(format!("/verify_license/{license}")))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({"verify": verify})))
                .mount(&server)
                .await;
        }
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
