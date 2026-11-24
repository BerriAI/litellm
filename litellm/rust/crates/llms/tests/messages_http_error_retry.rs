use litellm_host::interceptors::{RetriedRequest, RetryReason};
use litellm_llms::{
    anthropic::{
        common_utils::existing_betas, messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    },
    azure_ai::messages::transformation::AZURE_AI_MESSAGES_CONFIG,
    base_llm::messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    bedrock::{
        messages::claude_platform::transformation::BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG,
        messages::invoke::transformation::BEDROCK_INVOKE_MESSAGES_CONFIG,
    },
    bedrock_mantle::messages::transformation::{
        BEDROCK_MANTLE_MESSAGES_CONFIG, BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG,
    },
    deepseek::messages::transformation::DEEPSEEK_MESSAGES_CONFIG,
    edenai::messages::transformation::EDENAI_MESSAGES_CONFIG,
    minimax::messages::transformation::MINIMAX_MESSAGES_CONFIG,
    openrouter::messages::transformation::OPENROUTER_MESSAGES_CONFIG,
    tencent::messages::transformation::TENCENT_MESSAGES_CONFIG,
    vertex_ai::messages::transformation::VERTEX_AI_MESSAGES_CONFIG,
};
use litellm_llms_types::providers::anthropic::AnthropicBeta;
use rstest::rstest;
use serde_json::{Value, json};

#[derive(Clone, Copy, PartialEq)]
enum BindingRecovery {
    None,
    HeaderBeta,
    BodyBeta,
}

fn request_for(table: &Value, case: &Value) -> Value {
    let mut body = table["request"].clone();
    if let Some(thinking) = case.get("thinking") {
        body["thinking"] = thinking.clone();
    }
    body
}

fn expected_retry(
    table: &Value,
    case: &Value,
    recovery: BindingRecovery,
) -> Option<RetriedRequest> {
    let body = request_for(table, case);
    let (retried, outcome) = match recovery {
        BindingRecovery::None => (
            case.get("other_hosts_retried").unwrap_or(&case["retried"]),
            case.get("other_hosts_outcome").unwrap_or(&case["outcome"]),
        ),
        BindingRecovery::HeaderBeta | BindingRecovery::BodyBeta => {
            (&case["retried"], &case["outcome"])
        }
    };
    retried
        .as_bool()
        .unwrap()
        .then(|| match outcome.as_str().unwrap() {
            "strip" => RetriedRequest {
                reasons: vec![RetryReason::ThinkingSignature],
                body: table["stripped_request"].clone(),
            },
            "unchanged" => RetriedRequest {
                reasons: Vec::new(),
                body,
            },
            "drop_block" => {
                let mut body = body;
                body["thinking"] = case["retry_thinking"].clone();
                if recovery == BindingRecovery::BodyBeta {
                    body["anthropic_beta"] = json!([table["binding_beta"]]);
                }
                RetriedRequest {
                    reasons: vec![RetryReason::ThinkingBinding],
                    body,
                }
            }
            other => panic!("unknown outcome {other}"),
        })
}

#[rstest]
#[case::anthropic(&ANTHROPIC_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::azure_ai(&AZURE_AI_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::bedrock_invoke(&BEDROCK_INVOKE_MESSAGES_CONFIG, BindingRecovery::BodyBeta)]
#[case::bedrock_claude_platform(&BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::deepseek(&DEEPSEEK_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::edenai(&EDENAI_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::minimax(&MINIMAX_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::openrouter(&OPENROUTER_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::tencent(&TENCENT_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::vertex_ai(&VERTEX_AI_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
fn every_messages_provider_retries_like_the_shared_cases(
    #[case] config: &dyn BaseMessagesConfig,
    #[case] recovery: BindingRecovery,
) {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/anthropic_messages_http_error_retry.json"
    ))
    .unwrap();
    let table = &fixture["http_error_retry"];
    let cases = table["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    let mismatched: Vec<&str> = cases
        .iter()
        .filter(|case| {
            let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
            config.retry_after_http_error(
                status,
                case["error_text"].as_str().unwrap(),
                &request_for(table, case),
                &MessagesTransformContext::default(),
            ) != expected_retry(table, case, recovery)
        })
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    assert_eq!(mismatched, Vec::<&str>::new());
}

#[rstest]
#[case::anthropic(&ANTHROPIC_MESSAGES_CONFIG, true)]
#[case::azure_ai(&AZURE_AI_MESSAGES_CONFIG, true)]
#[case::bedrock_claude_platform(&BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG, true)]
#[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG, true)]
#[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, true)]
#[case::vertex_ai(&VERTEX_AI_MESSAGES_CONFIG, true)]
#[case::bedrock_invoke_reads_the_body(&BEDROCK_INVOKE_MESSAGES_CONFIG, false)]
#[case::deepseek(&DEEPSEEK_MESSAGES_CONFIG, false)]
#[case::edenai(&EDENAI_MESSAGES_CONFIG, false)]
#[case::minimax(&MINIMAX_MESSAGES_CONFIG, false)]
#[case::openrouter(&OPENROUTER_MESSAGES_CONFIG, false)]
#[case::tencent(&TENCENT_MESSAGES_CONFIG, false)]
fn only_claude_header_hosts_add_the_binding_beta_to_a_retry(
    #[case] config: &dyn BaseMessagesConfig,
    #[case] adds_the_beta: bool,
) {
    let sent = vec![(
        "Anthropic-Beta".to_owned(),
        "context-1m-2025-08-07".to_owned(),
    )];

    let binding =
        existing_betas(&config.headers_for_retry(sent.clone(), &[RetryReason::ThinkingBinding]));
    let signature = config.headers_for_retry(sent.clone(), &[RetryReason::ThinkingSignature]);

    assert!(binding.contains(&AnthropicBeta::Context1m20250807));
    assert_eq!(
        binding.contains(&AnthropicBeta::ThinkingBindingControls20260801),
        adds_the_beta
    );
    assert_eq!(signature, sent);
}

#[rstest]
#[case::anthropic(&ANTHROPIC_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::azure_ai(&AZURE_AI_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::bedrock_invoke(&BEDROCK_INVOKE_MESSAGES_CONFIG, BindingRecovery::BodyBeta)]
#[case::bedrock_claude_platform(&BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
#[case::deepseek(&DEEPSEEK_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::edenai(&EDENAI_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::minimax(&MINIMAX_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::openrouter(&OPENROUTER_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::tencent(&TENCENT_MESSAGES_CONFIG, BindingRecovery::None)]
#[case::vertex_ai(&VERTEX_AI_MESSAGES_CONFIG, BindingRecovery::HeaderBeta)]
fn every_messages_provider_drops_rejected_strict_tools_like_the_shared_cases(
    #[case] config: &dyn BaseMessagesConfig,
    #[case] recovery: BindingRecovery,
) {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/anthropic_messages_http_error_retry.json"
    ))
    .unwrap();
    let replay = &fixture["http_error_retry"];
    let table = &fixture["strict_tools_retry"];
    let cases = table["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    let mismatched: Vec<&str> = cases
        .iter()
        .filter(|case| {
            let mut body = request_for(replay, case);
            body["tools"] = table["tools"].clone();
            let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
            let expected = case["retried"].as_bool().unwrap().then(|| {
                let mut reasons = Vec::new();
                let binding_thinking = case
                    .get("retry_thinking")
                    .filter(|_| recovery != BindingRecovery::None);
                let mut expected = if let Some(thinking) = binding_thinking {
                    reasons.push(RetryReason::ThinkingBinding);
                    let mut recovered = body.clone();
                    recovered["thinking"] = thinking.clone();
                    if recovery == BindingRecovery::BodyBeta {
                        recovered["anthropic_beta"] = json!([replay["binding_beta"]]);
                    }
                    recovered
                } else if case["stripped"].as_bool().unwrap() {
                    reasons.push(RetryReason::ThinkingSignature);
                    replay["stripped_request"].clone()
                } else {
                    body.clone()
                };
                expected["tools"] = if case["strict_dropped"].as_bool().unwrap() {
                    reasons.push(RetryReason::StrictTools {
                        dropped: usize::try_from(table["dropped"].as_u64().unwrap()).unwrap(),
                    });
                    table["tools_without_strict"].clone()
                } else {
                    table["tools"].clone()
                };
                RetriedRequest {
                    reasons,
                    body: expected,
                }
            });
            config.retry_after_http_error(
                status,
                case["error_text"].as_str().unwrap(),
                &body,
                &MessagesTransformContext {
                    drops_unsupported_params: case["drops_unsupported_params"].as_bool().unwrap(),
                    ..MessagesTransformContext::default()
                },
            ) != expected
        })
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    assert_eq!(mismatched, Vec::<&str>::new());
}
