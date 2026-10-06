use std::sync::{Arc, mpsc};

use axum::{
    body::{Body, to_bytes},
    extract::Request,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_config::Config;
use litellm_gateway::hooks::GatewayHooks;
use litellm_gateway_inference::{Deployment, RouterHooks};
use litellm_host::interceptors::{Interceptors, RawResponse, RequestContext, WireRequest};
use litellm_inference::RouteError;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

#[derive(Clone)]
struct Recorder {
    phases: mpsc::Sender<&'static str>,
    reject_gateway: bool,
    reject_router: bool,
}

impl GatewayHooks for Recorder {
    async fn pre_call(&self, request: Request) -> Result<Request, Response> {
        self.phases.send("gateway.pre_call").unwrap();
        if self.reject_gateway {
            return Err((StatusCode::FORBIDDEN, "rejected by gateway hook").into_response());
        }
        Ok(request)
    }

    async fn post_call(&self, response: Response) -> Response {
        self.phases.send("gateway.post_call").unwrap();
        ([("x-hook-completed", "true")], response).into_response()
    }
}

impl RouterHooks for Recorder {
    async fn filter_deployments<'a>(
        &self,
        model: &str,
        deployments: Vec<&'a Deployment>,
    ) -> Vec<&'a Deployment> {
        self.phases.send("router.filter_deployments").unwrap();
        assert_eq!(model, "public-model");
        assert_eq!(deployments.len(), 1);
        if self.reject_router {
            return Vec::new();
        }
        deployments
    }
}

impl Interceptors<RouteError> for Recorder {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, RouteError> {
        self.phases
            .send("inference.before_provider_request")
            .unwrap();
        assert_eq!(context.model, wire.body["model"]);
        Ok(WireRequest {
            headers: wire
                .headers
                .into_iter()
                .chain([("x-inference-hook".into(), "called".into())])
                .collect(),
            ..wire
        })
    }

    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), RouteError> {
        self.phases
            .send("inference.after_provider_response")
            .unwrap();
        let response: Value = serde_json::from_str(&raw.body).unwrap();
        assert_eq!(response["content"][0]["text"], "provider reply");
        Ok(())
    }
}

#[fixture]
fn phases() -> (mpsc::Sender<&'static str>, mpsc::Receiver<&'static str>) {
    mpsc::channel()
}

#[rstest]
#[case::all_layers(false, false, 200, &[
    "gateway.pre_call",
    "router.filter_deployments",
    "inference.before_provider_request",
    "inference.after_provider_response",
    "gateway.post_call",
])]
#[case::gateway_rejection(true, false, 403, &["gateway.pre_call"])]
#[case::router_filters_every_candidate(false, true, 400, &[
    "gateway.pre_call",
    "router.filter_deployments",
    "gateway.post_call",
])]
#[tokio::test]
async fn request_hooks_compose_and_rejections_stop_inner_layers(
    phases: (mpsc::Sender<&'static str>, mpsc::Receiver<&'static str>),
    #[case] reject_gateway: bool,
    #[case] reject_router: bool,
    #[case] status: u16,
    #[case] expected_phases: &[&str],
) {
    let upstream = MockServer::start().await;
    let message = json!({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "text", "text": "provider reply"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-inference-hook", "called"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&message))
        .expect(if reject_gateway || reject_router {
            0
        } else {
            1
        })
        .mount(&upstream)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/test-model\n      api_key: test-key\n      api_base: {}\ngeneral_settings:\n  master_key: gateway-key\n",
        upstream.uri(),
    )).unwrap();
    let (sender, receiver) = phases;
    let recorder = Recorder {
        phases: sender,
        reject_gateway,
        reject_router,
    };
    let inference = Arc::into_inner(litellm_gateway::build_inference(&config).unwrap())
        .unwrap()
        .with_hooks(recorder.clone(), recorder.clone());
    let app =
        litellm_gateway::router_with_hooks(Arc::new(inference), &config, None, None, recorder);
    let response = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", "Bearer gateway-key")
                .body(Body::from(
                    json!({
                        "model": "public-model", "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 16,
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), status);
    assert_eq!(
        response.headers().contains_key("x-hook-completed"),
        !reject_gateway
    );
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    if reject_gateway {
        assert_eq!(body, "rejected by gateway hook");
    } else if !reject_router {
        let response: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(response["content"], message["content"]);
        assert_eq!(response["usage"], message["usage"]);
    }
    assert_eq!(receiver.try_iter().collect::<Vec<_>>(), expected_phases);
    assert_eq!(
        upstream.received_requests().await.unwrap().len(),
        usize::from(!reject_gateway && !reject_router)
    );
}
