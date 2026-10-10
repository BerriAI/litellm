use litellm_inference::outbound::outbound_request;
use litellm_llms::base_llm::auth::Authenticated;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::caller_url(None, "https://caller.example/messages")]
#[case::authenticated_url(
    Some("https://session.example/messages"),
    "https://session.example/messages"
)]
fn authenticated_endpoints_outrank_the_requested_url(
    #[case] authenticated_url: Option<&str>,
    #[case] expected: &str,
) {
    let request = outbound_request(
        Authenticated {
            url: authenticated_url.map(str::to_string),
            headers: vec![("authorization".into(), "Bearer session".into())],
            signer: None,
        },
        "https://caller.example/messages".into(),
        &json!({"model": "native-test"}),
        None,
    )
    .unwrap();
    assert_eq!(request.url(), expected);
    assert_eq!(request.header("authorization"), Some("Bearer session"));
    assert_eq!(
        serde_json::from_slice::<Value>(request.body()).unwrap(),
        json!({"model": "native-test"})
    );
}
