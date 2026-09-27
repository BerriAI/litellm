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
    CoreClient,
    ocr::wire::{OcrWireRequest, decode_request},
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
        let client = CoreClient::new(
            owner.clone(),
            http,
            Arc::new(RecordingSecrets::new([("VERTEXAI_CREDENTIALS", identity)])),
        )
        .with_ocr_settings(OcrSettings {
            vertex_location: Some(location.into()),
            ..OcrSettings::default()
        });
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
        let result = client.ocr(request).await.unwrap();
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

#[rstest]
#[tokio::test]
async fn configured_client_shares_http_and_secrets_across_routes(resources: CoreResources) {
    let key = "configured-key";
    let agent = "configured-agent";
    let secrets = Arc::new(RecordingSecrets::new([
        ("ANTHROPIC_API_KEY", key),
        ("OPENAI_API_KEY", key),
        ("MISTRAL_API_KEY", key),
    ]));
    let client = CoreClient::new(
        resources,
        Resolution::from(&HttpSettings {
            user_agent: Some(agent.into()),
            ..HttpSettings::default()
        })
        .config,
        secrets.clone(),
    );
    let upstream = upstream([
        json_response(
            json!({"id":"message", "type":"message", "role":"assistant", "model":"test-model",
            "content":[], "stop_reason":"end_turn", "usage":{"input_tokens":1,"output_tokens":1}}),
        ),
        json_response(json!({"id":"response", "model":"test-model", "output":[]})),
        json_response(json!({"pages":[]})),
    ])
    .await;
    let body = serde_json::from_value(json!({
        "model": "anthropic/test-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    client
        .messages(litellm_core::messages::MessagesCall {
            body,
            api_key: None,
            api_base: Some(upstream.uri()),
            custom_llm_provider: None,
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping: Default::default(),
        })
        .await
        .unwrap();
    client
        .responses(litellm_core::responses::types::ResponsesCall {
            model: "openai/test-model".into(),
            input: json!("hi"),
            optional_params: Default::default(),
            api_key: None,
            api_base: Some(upstream.uri()),
            custom_llm_provider: None,
            extra_headers: None,
            timeout: None,
        })
        .await
        .unwrap();
    client.ocr(decode_request(OcrWireRequest {
        model: "mistral/test-model".into(),
        document: json!({"type":"document_url", "document_url":"data:application/pdf;base64,YWJj"}),
        api_key: None,
        api_base: Some(upstream.uri()),
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: Default::default(),
        input_sources: Default::default(),
        timeout_seconds: None,
    }).unwrap()).await.unwrap();

    let requests = support::received(&upstream).await;
    assert_eq!(requests.len(), 3);
    assert!(
        requests
            .iter()
            .all(|request| request.header("user-agent") == Some(agent))
    );
    assert_eq!(requests[0].header("x-api-key"), Some(key));
    assert!(
        requests[1..].iter().all(
            |request| request.header("authorization") == Some(format!("Bearer {key}").as_str())
        )
    );
    let requested = secrets.requested();
    assert!(
        ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY"]
            .iter()
            .all(|name| requested.iter().any(|requested| requested == name))
    );
}
