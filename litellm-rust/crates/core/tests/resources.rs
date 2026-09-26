mod support;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use litellm_auth::AuthServices;
use litellm_auth_gcp::{
    CredentialSource, VertexAuth, VertexAuthFuture, VertexProviderLoader, VertexTokenSource,
};
use litellm_core::{
    ocr::{
        client::perform,
        wire::{OcrWireRequest, decode_request},
    },
    resources::CoreResources,
};
use litellm_http::{HttpSettings, Resolution};
use litellm_llms::base_llm::ocr::settings::OcrSettings;
use rstest::{fixture, rstest};
use serde_json::json;
use support::{ReceivedRequest, RecordingSecrets, http_pool, json_response, upstream};

struct TokenSource(String);

impl VertexTokenSource for TokenSource {
    fn project_id(&self) -> VertexAuthFuture<'_, String> {
        Box::pin(async { Ok(self.0.clone()) })
    }

    fn token(&self) -> VertexAuthFuture<'_, String> {
        Box::pin(async { Ok(self.0.clone()) })
    }
}

#[derive(Default)]
struct Loader(AtomicUsize);

impl VertexProviderLoader for Loader {
    fn load(&self, source: CredentialSource) -> VertexAuthFuture<'_, Arc<dyn VertexTokenSource>> {
        Box::pin(async move {
            self.0.fetch_add(1, Ordering::SeqCst);
            let identity = match source {
                CredentialSource::Trusted(secret) => secret.expose().to_string(),
                other => panic!("unexpected credential source: {other:?}"),
            };
            Ok(Arc::new(TokenSource(identity)) as Arc<dyn VertexTokenSource>)
        })
    }
}

#[fixture]
fn loader() -> Arc<Loader> {
    Arc::new(Loader::default())
}

#[fixture]
fn resources(loader: Arc<Loader>) -> CoreResources {
    CoreResources {
        auth: Arc::new(AuthServices {
            gcp: VertexAuth::new(loader),
            ..AuthServices::default()
        }),
        pool: Arc::new(http_pool()),
    }
}

#[rstest]
#[case::shared_identity(false, "first-identity", 1)]
#[case::different_identity(false, "second-identity", 2)]
#[case::independent_resources(true, "first-identity", 2)]
#[tokio::test]
async fn auth_survives_per_call_clients_without_freezing_settings_or_secrets(
    loader: Arc<Loader>,
    #[with(loader.clone())] resources: CoreResources,
    #[case] independent: bool,
    #[case] second_identity: &str,
    #[case] expected_loads: usize,
) {
    let response = json_response(json!({"pages": [{"index": 0, "markdown": "hello"}]}));
    let upstream = upstream([response.clone(), response]).await;
    let second_resources = if independent {
        CoreResources {
            auth: Arc::new(AuthServices {
                gcp: VertexAuth::new(loader.clone()),
                ..AuthServices::default()
            }),
            ..resources.clone()
        }
    } else {
        resources.clone()
    };
    for (owner, identity, agent, location) in [
        (&resources, "first-identity", "first-agent", "us-central1"),
        (
            &second_resources,
            second_identity,
            "second-agent",
            "europe-west4",
        ),
    ] {
        let http = Resolution::from(&HttpSettings {
            user_agent: Some(agent.into()),
            ..HttpSettings::default()
        })
        .config;
        let client = owner
            .ocr_client(
                &http,
                Default::default(),
                OcrSettings {
                    vertex_location: Some(location.into()),
                    ..OcrSettings::default()
                },
                Arc::new(RecordingSecrets::new([("VERTEXAI_CREDENTIALS", identity)])),
            )
            .unwrap();
        let request = decode_request(OcrWireRequest {
            model: "vertex_ai/mistral-ocr-maas".into(),
            document: json!({"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}),
            api_key: None,
            api_base: Some(upstream.uri()),
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: Default::default(),
            input_sources: Default::default(),
            timeout_seconds: Some(5.0),
        }).unwrap();
        let result = perform(&client, request).await.unwrap();
        assert!(!result.pages.is_empty());
    }
    let requests = upstream.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    for (request, identity, agent, location) in [
        (&requests[0], "first-identity", "first-agent", "us-central1"),
        (
            &requests[1],
            second_identity,
            "second-agent",
            "europe-west4",
        ),
    ] {
        assert_eq!(
            request.header("authorization"),
            Some(format!("Bearer {identity}").as_str())
        );
        assert_eq!(request.header("user-agent"), Some(agent));
        assert!(
            request
                .url
                .path()
                .contains(&format!("/projects/{identity}/locations/{location}/"))
        );
    }
    assert_eq!(loader.0.load(Ordering::SeqCst), expected_loads);
}
