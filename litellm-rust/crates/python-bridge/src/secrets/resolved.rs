use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_secrets::source::SecretSource;
use litellm_secrets::{
    Error, FailurePolicy, OidcResolver, SecretManagerState, SecretResolver, SecretValue,
};

use super::config::SecretManagerSnapshot;

pub(crate) struct ResolvedSecrets {
    resolver: SecretResolver,
}

impl ResolvedSecrets {
    pub(crate) fn new(snapshot: SecretManagerSnapshot) -> Self {
        Self::from_state(snapshot.into_state())
    }

    fn from_state(state: Arc<SecretManagerState>) -> Self {
        Self {
            resolver: SecretResolver::new_python_compatible(
                state,
                Arc::new(ProcessEnvironment),
                OidcResolver::default(),
            )
            .with_failure_policy(FailurePolicy::EnvironmentFallback),
        }
    }
}

impl SecretSource for ResolvedSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        Box::pin(self.resolver.get_secret_str(name, None))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aws_sdk_secretsmanager::Client;
    use aws_sdk_secretsmanager::config::{
        BehaviorVersion, Credentials, Region, retry::RetryConfig,
    };
    use litellm_secrets::{AccessMode, KeyManagementSettings, SecretManager, SecretManagerState};
    use litellm_secrets_aws::AwsSecretsManagerV2;
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_partial_json, header},
    };

    use super::ResolvedSecrets;
    use litellm_secrets::source::SecretSource;

    fn state(server: &MockServer, settings: KeyManagementSettings) -> Arc<SecretManagerState> {
        let client = Client::from_conf(
            aws_sdk_secretsmanager::Config::builder()
                .behavior_version(BehaviorVersion::latest())
                .region(Region::new("us-east-1"))
                .credentials_provider(Credentials::new("test", "test", None, None, "test"))
                .endpoint_url(server.uri())
                .retry_config(RetryConfig::disabled())
                .build(),
        );
        Arc::new(SecretManagerState::new(
            SecretManager::AwsSecretsManagerV2(AwsSecretsManagerV2::new(
                client,
                (&settings).into(),
            )),
            settings,
        ))
    }

    async fn resolve(state: Arc<SecretManagerState>, name: &'static str) -> Option<String> {
        ResolvedSecrets::from_state(state)
            .resolve(&[name])
            .await
            .unwrap()
            .get(name)
    }

    #[tokio::test]
    async fn hosted_key_miss_falls_back_to_environment() {
        let name = "LITELLM_RUST_BRIDGE_HOSTED_KEY_MISS";
        unsafe { std::env::set_var(name, "env-key") };
        let server = MockServer::start().await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .and(body_partial_json(json!({"SecretId": name})))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"SecretString": "manager-key"})),
            )
            .expect(0)
            .mount(&server)
            .await;
        let result = resolve(
            state(
                &server,
                KeyManagementSettings {
                    hosted_keys: Some(vec!["OTHER".into()]),
                    ..Default::default()
                },
            ),
            name,
        )
        .await;
        unsafe { std::env::remove_var(name) };
        assert_eq!(result.as_deref(), Some("env-key"));
        assert_eq!(server.received_requests().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn aws_read_failure_preserves_absence_without_environment_fallback() {
        let name = "LITELLM_RUST_BRIDGE_MANAGER_FAILURE";
        unsafe { std::env::set_var(name, "env-key") };
        let server = MockServer::start().await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .respond_with(ResponseTemplate::new(500))
            .expect(2)
            .mount(&server)
            .await;
        let result = resolve(state(&server, KeyManagementSettings::default()), name).await;
        let missing = resolve(
            state(&server, KeyManagementSettings::default()),
            "LITELLM_RUST_BRIDGE_MANAGER_FAILURE_MISSING",
        )
        .await;
        unsafe { std::env::remove_var(name) };
        assert_eq!(result, None);
        assert_eq!(missing, None);
    }

    #[rstest::rstest]
    #[case::capitalized_true("True")]
    #[case::parenthesized_false("(False)")]
    #[tokio::test]
    async fn boolean_manager_values_are_absent_like_get_secret_str(#[case] value: &str) {
        let name = "LITELLM_RUST_BRIDGE_BOOLEAN_VALUE";
        let server = MockServer::start().await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"SecretString": value})))
            .expect(1)
            .mount(&server)
            .await;
        assert_eq!(
            resolve(state(&server, KeyManagementSettings::default()), name).await,
            None
        );
    }

    #[tokio::test]
    async fn undeclared_names_are_read_from_the_manager() {
        let declared = "LITELLM_RUST_BRIDGE_DECLARED";
        let undeclared = "LITELLM_RUST_BRIDGE_UNDECLARED_MANAGED";
        unsafe { std::env::set_var(undeclared, "env-key") };
        let server = MockServer::start().await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .and(body_partial_json(json!({"SecretId": declared})))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"SecretString": "declared-key"})),
            )
            .mount(&server)
            .await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .and(body_partial_json(json!({"SecretId": undeclared})))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"SecretString": "manager-key"})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let source = ResolvedSecrets::from_state(state(&server, KeyManagementSettings::default()));
        let snapshot = source.resolve(&[declared]).await.unwrap();
        assert_eq!(snapshot.get(undeclared), None);
        let result = source
            .get_secret_str(undeclared)
            .await
            .unwrap()
            .map(|value| value.expose().to_owned());
        unsafe { std::env::remove_var(undeclared) };
        assert_eq!(result.as_deref(), Some("manager-key"));
    }

    #[tokio::test]
    async fn write_only_mode_never_consults_the_manager() {
        let name = "LITELLM_RUST_BRIDGE_WRITE_ONLY";
        unsafe { std::env::set_var(name, "env-key") };
        let server = MockServer::start().await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"SecretString": "manager-key"})),
            )
            .expect(0)
            .mount(&server)
            .await;
        let result = resolve(
            state(
                &server,
                KeyManagementSettings {
                    access_mode: AccessMode::WriteOnly,
                    ..Default::default()
                },
            ),
            name,
        )
        .await;
        unsafe { std::env::remove_var(name) };
        assert_eq!(result.as_deref(), Some("env-key"));
        assert_eq!(server.received_requests().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn read_only_mode_resolves_from_the_manager() {
        let name = "LITELLM_RUST_BRIDGE_READ_ONLY";
        let server = MockServer::start().await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .and(body_partial_json(json!({"SecretId": name})))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"SecretString": "manager-key"})),
            )
            .expect(1)
            .mount(&server)
            .await;
        assert_eq!(
            resolve(state(&server, KeyManagementSettings::default()), name)
                .await
                .as_deref(),
            Some("manager-key")
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn oidc_failures_are_not_converted_to_missing_secrets() {
        let result = ResolvedSecrets::from_state(Arc::new(SecretManagerState::default()))
            .resolve(&["oidc/"])
            .await;
        assert!(matches!(result, Err(litellm_secrets::Error::InvalidOidc)));
    }

    #[tokio::test]
    async fn names_excluded_by_hosted_keys_read_the_process_environment() {
        let name = "LITELLM_RUST_BRIDGE_UNDECLARED";
        unsafe { std::env::set_var(name, "env-key") };
        let server = MockServer::start().await;
        let result = resolve(
            state(
                &server,
                KeyManagementSettings {
                    hosted_keys: Some(vec!["OTHER".into()]),
                    ..Default::default()
                },
            ),
            name,
        )
        .await;
        unsafe { std::env::remove_var(name) };
        assert_eq!(result.as_deref(), Some("env-key"));
        assert_eq!(server.received_requests().await.unwrap().len(), 0);
    }
}
