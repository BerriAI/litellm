use litellm_auth::{AwsParams, CredentialPlacement};
use litellm_llms::{
    base_llm::{auth::AuthScheme, messages::transformation::BaseMessagesConfig},
    bedrock_mantle::messages::transformation::BEDROCK_MANTLE_MESSAGES_CONFIG as CONFIG,
};
use litellm_llms_types::formats::messages::MessagesRequest;
use litellm_router_types::LitellmParams;
use rstest::rstest;
use serde_json::json;

fn no_env(_: &str) -> Option<String> {
    None
}

#[rstest]
#[case::root("https://proxy.test", "https://proxy.test/anthropic/v1/messages")]
#[case::version("https://proxy.test/v1/", "https://proxy.test/anthropic/v1/messages")]
#[case::openai(
    "https://proxy.test/openai/v1",
    "https://proxy.test/anthropic/v1/messages"
)]
#[case::anthropic(
    "https://proxy.test/anthropic/v1",
    "https://proxy.test/anthropic/v1/messages"
)]
#[case::messages(
    "https://proxy.test/v1/messages",
    "https://proxy.test/anthropic/v1/messages"
)]
#[case::complete(
    "https://proxy.test/anthropic/v1/messages/",
    "https://proxy.test/anthropic/v1/messages"
)]
#[case::custom_path(
    "https://proxy.test/tenant/v1",
    "https://proxy.test/tenant/anthropic/v1/messages"
)]
fn native_messages_url(#[case] base: &str, #[case] expected: &str) {
    assert_eq!(
        CONFIG
            .get_complete_url(
                Some(base),
                "claude-test",
                &Default::default(),
                true,
                &no_env
            )
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::explicit(Some("explicit"), Some("mantle"), Some("bedrock"), "explicit")]
#[case::mantle_env(None, Some("mantle"), Some("bedrock"), "mantle")]
#[case::bedrock_env(None, None, Some("bedrock"), "bedrock")]
fn bearer_precedence(
    #[case] key: Option<&str>,
    #[case] mantle: Option<&str>,
    #[case] bedrock: Option<&str>,
    #[case] expected: &str,
) {
    let env = |name: &str| match name {
        "BEDROCK_MANTLE_API_KEY" => mantle.map(str::to_string),
        "AWS_BEARER_TOKEN_BEDROCK" => bedrock.map(str::to_string),
        _ => None,
    };
    let result = CONFIG
        .validate_environment(
            vec![
                ("Authorization".into(), "caller".into()),
                ("X-Api-Key".into(), "caller".into()),
            ],
            key,
            "claude-test",
            &Default::default(),
            &env,
        )
        .unwrap();
    assert!(result.headers.is_empty());
    assert!(
        matches!(result.auth, AuthScheme::Credential { placement: CredentialPlacement::Bearer, secret } if secret.expose() == expected)
    );
}

#[rstest]
#[case::endpoint(
    None,
    "claude-test",
    Some("https://bedrock-mantle.eu-west-1.api.aws/v1"),
    "eu-west-1"
)]
#[case::explicit_region(
    Some("us-west-2"),
    "claude-test",
    Some("https://bedrock-mantle.eu-west-1.api.aws/v1"),
    "us-west-2"
)]
#[case::model_region(None, "us-west-2/claude-test", None, "us-west-2")]
#[case::fallback(None, "claude-test", None, "us-east-1")]
fn signer_scope_matches_native_url(
    #[case] region: Option<&str>,
    #[case] model: &str,
    #[case] base: Option<&str>,
    #[case] expected: &str,
) {
    let params = LitellmParams {
        api_base: base.map(str::to_string),
        aws: AwsParams {
            aws_region_name: region.map(str::to_string),
            ..Default::default()
        },
        ..Default::default()
    };
    let result = CONFIG
        .validate_environment(vec![], None, model, &params, &no_env)
        .unwrap();
    let url = CONFIG
        .get_complete_url(base, model, &params, false, &no_env)
        .unwrap();
    assert_eq!(
        url,
        format!("https://bedrock-mantle.{expected}.api.aws/anthropic/v1/messages")
    );
    assert!(
        matches!(result.auth, AuthScheme::AwsSigV4 { region, service: "bedrock", .. } if region == expected)
    );
}

#[rstest]
fn workspace_is_sent_as_a_header() {
    let params: LitellmParams = serde_json::from_value(
        json!({"model": "claude-test", "aws_bedrock_project_id": "project-test"}),
    )
    .unwrap();
    let result = CONFIG
        .validate_environment(vec![], Some("key"), "claude-test", &params, &no_env)
        .unwrap();
    assert_eq!(
        result.headers,
        [("anthropic-workspace-id".into(), "project-test".into())]
    );
}

#[rstest]
#[case::native("anthropic.claude-test")]
#[case::alias("mantle/anthropic.claude-test")]
#[case::region("us-west-2/anthropic.claude-test")]
fn model_and_stream_stay_in_the_body(#[case] model: &str) {
    let body = json!({"model": model, "stream": true, "max_tokens": 16, "messages": [], "anthropic_version": "v", "anthropic_beta": ["beta"]});
    assert_eq!(
        CONFIG.wire_body(body),
        json!({"model": "anthropic.claude-test", "stream": true, "max_tokens": 16, "messages": []})
    );
}

#[rstest]
fn unknown_caller_betas_are_preserved_with_derived_feature_betas() {
    let body: MessagesRequest = serde_json::from_value(json!({"model": "claude-test", "max_tokens": 16, "messages": [], "output_format": {"type":"json_schema", "schema":{"type":"object"}}})).unwrap();
    let headers = CONFIG.request_headers(
        vec![("Anthropic-Beta".into(), "future-feature-2099-01-01".into())],
        &body,
    );
    let beta = headers
        .iter()
        .find(|(name, _)| name == "anthropic-beta")
        .unwrap()
        .1
        .split(',')
        .collect::<Vec<_>>();
    assert!(beta.contains(&"future-feature-2099-01-01"));
    assert!(
        beta.contains(
            &litellm_llms_types::providers::anthropic::AnthropicBeta::StructuredOutputs20251113
                .as_str()
        )
    );
}
