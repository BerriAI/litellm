use std::{collections::HashMap, sync::Arc};

use futures_util::{future::BoxFuture, future::try_join_all};
use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use litellm_llms::base_llm::inference::secrets::{SecretSource, Secrets};
use litellm_secrets::{
    Error, FailurePolicy, OidcResolver, Secret, SecretManagerState, SecretResolver,
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
            resolver: SecretResolver::new(
                state,
                Arc::new(ProcessEnvironment),
                OidcResolver::default(),
            )
            .with_failure_policy(FailurePolicy::EnvironmentFallback),
        }
    }
}

impl SecretSource for ResolvedSecrets {
    fn resolve<'a>(&'a self, names: &'a [&'static str]) -> BoxFuture<'a, Result<Secrets, Error>> {
        Box::pin(async move {
            let values = try_join_all(names.iter().map(|name| async move {
                self.resolver
                    .get_secret(name, None)
                    .await
                    .map(|secret| secret.map(|secret| ((*name).to_owned(), secret_value(secret))))
            }))
            .await?
            .into_iter()
            .flatten()
            .collect::<HashMap<_, _>>();
            Ok(Arc::new(ResolvedLookup { values }) as Secrets)
        })
    }
}

struct ResolvedLookup {
    values: HashMap<String, String>,
}

impl Lookup for ResolvedLookup {
    fn get(&self, name: &str) -> Option<String> {
        self.values
            .get(name)
            .cloned()
            .or_else(|| ProcessEnvironment.get(name))
    }
}

fn secret_value(secret: Secret) -> String {
    match secret {
        Secret::String(value) => value.expose().to_owned(),
        Secret::Bool(value) => if value { "True" } else { "False" }.to_owned(),
        Secret::Json(value) => value.to_string(),
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
    use litellm_llms::base_llm::inference::secrets::SecretSource;

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
    async fn manager_failure_falls_back_to_environment() {
        let name = "LITELLM_RUST_BRIDGE_MANAGER_FAILURE";
        unsafe { std::env::set_var(name, "env-key") };
        let server = MockServer::start().await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .respond_with(ResponseTemplate::new(500))
            .expect(1)
            .mount(&server)
            .await;
        let result = resolve(state(&server, KeyManagementSettings::default()), name).await;
        unsafe { std::env::remove_var(name) };
        assert_eq!(result.as_deref(), Some("env-key"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);

        let missing_server = MockServer::start().await;
        Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
            .respond_with(ResponseTemplate::new(500))
            .expect(1)
            .mount(&missing_server)
            .await;
        let missing =
            ResolvedSecrets::from_state(state(&missing_server, KeyManagementSettings::default()))
                .resolve(&["LITELLM_RUST_BRIDGE_MANAGER_FAILURE_MISSING"])
                .await;
        assert!(matches!(missing, Err(litellm_secrets::Error::Aws(_))));
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
    async fn undeclared_names_still_read_the_process_environment() {
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
