use super::*;

pub(super) fn manager(server: &MockServer, settings: KeyManagementSettings) -> AwsSecretsManagerV2 {
    let client = Client::from_conf(client_builder(server).build());
    AwsSecretsManagerV2::new(client, (&settings).into())
}

pub(super) fn loaded_manager(server: &MockServer) -> AwsSecretsManagerV2 {
    let endpoint_url = server.uri();
    let environment: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> =
        Arc::new(move |name: &str| match name {
            "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint_url.clone()),
            "AWS_ACCESS_KEY_ID" => Some("test".into()),
            "AWS_SECRET_ACCESS_KEY" => Some("test".into()),
            _ => None,
        });
    AwsSecretsManagerV2::load_aws_secret_manager(
        Some(true),
        KeyManagementSettings {
            aws_region_name: Some("us-east-1".into()),
            ..Default::default()
        },
        environment,
    )
    .unwrap()
    .unwrap()
}

#[fixture]
pub(super) fn default_settings() -> KeyManagementSettings {
    KeyManagementSettings::default()
}

pub(super) async fn scripted_actions(server: &MockServer, actions: Vec<Action>) {
    let count = actions.len() as u64;
    let step = AtomicUsize::new(0);
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(move |request: &wiremock::Request| {
            let Action {
                operation: action,
                request: expected,
                status,
                response,
            } = &actions[step.fetch_add(1, Ordering::SeqCst)];
            assert_eq!(
                request.headers["x-amz-target"],
                format!("secretsmanager.{action}")
            );
            let body: serde_json::Value = request.body_json().unwrap();
            let actual = serde_json::Value::Object(
                body.as_object()
                    .unwrap()
                    .iter()
                    .filter(|(key, _)| key.as_str() != "ClientRequestToken")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            );
            assert_eq!(&actual, expected);
            ResponseTemplate::new(*status).set_body_json(response)
        })
        .expect(count)
        .mount(server)
        .await;
}

pub(super) struct Action {
    pub(super) operation: &'static str,
    pub(super) request: serde_json::Value,
    pub(super) status: u16,
    pub(super) response: serde_json::Value,
}

pub(super) fn client_builder(server: &MockServer) -> aws_sdk_secretsmanager::config::Builder {
    aws_sdk_secretsmanager::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(Credentials::new("test", "test", None, None, "test"))
        .endpoint_url(server.uri())
        .retry_config(RetryConfig::disabled())
}
