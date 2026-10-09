use litellm_lens::{config::http_client, control::Control, pricing::GatewayTraceCosts};
use litellm_traces::TraceCostInput;
use litellm_traces_cache::TraceCostEstimator;
use rstest::{fixture, rstest};
use serde_json::json;
use std::collections::BTreeMap;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

#[fixture]
fn call() -> TraceCostInput {
    TraceCostInput {
        start_ns: 1_800_000_000_000_000_000,
        attributes: BTreeMap::from([
            ("gen_ai.request.model".into(), "provider/test-model".into()),
            ("gen_ai.usage.input_tokens".into(), "100".into()),
            ("gen_ai.usage.output_tokens".into(), "10".into()),
        ]),
    }
}

#[rstest]
#[case::paid(Some(0.125))]
#[case::free(Some(0.0))]
#[case::unknown(None)]
#[tokio::test]
async fn quotes_use_the_configured_service_connection(
    call: TraceCostInput,
    #[case] cost: Option<f64>,
) {
    let gateway = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/gateway/lens/internal/trace-costs"))
        .and(header("authorization", "Bearer test-service-token"))
        .and(body_json(json!({"calls": [call.clone()]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"costs": [cost]})))
        .expect(1)
        .mount(&gateway)
        .await;
    let estimator = GatewayTraceCosts::new(Control::new(
        http_client().unwrap(),
        format!("{}/gateway", gateway.uri()).parse().unwrap(),
        "test-service-token".into(),
    ));

    assert_eq!(estimator.estimate(vec![call]).await.unwrap(), vec![cost]);
}

#[rstest]
#[case::old_gateway(404, "")]
#[case::unavailable(503, "")]
#[case::invalid_json(200, "not-json")]
#[case::missing_costs(200, "{}")]
#[tokio::test]
async fn quote_failure_is_unavailable(
    call: TraceCostInput,
    #[case] status: u16,
    #[case] body: &str,
) {
    let gateway = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/lens/internal/trace-costs"))
        .respond_with(ResponseTemplate::new(status).set_body_string(body))
        .expect(1)
        .mount(&gateway)
        .await;
    let estimator = GatewayTraceCosts::new(Control::new(
        http_client().unwrap(),
        gateway.uri().parse().unwrap(),
        "test-service-token".into(),
    ));

    assert!(estimator.estimate(vec![call]).await.is_err());
}
