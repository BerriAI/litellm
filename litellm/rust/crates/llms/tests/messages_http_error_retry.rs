use litellm_host::interceptors::{RetriedRequest, RetryReason};
use litellm_llms::{
    anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
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
use rstest::rstest;
use serde_json::Value;

#[rstest]
#[case::anthropic(&ANTHROPIC_MESSAGES_CONFIG)]
#[case::azure_ai(&AZURE_AI_MESSAGES_CONFIG)]
#[case::bedrock_invoke(&BEDROCK_INVOKE_MESSAGES_CONFIG)]
#[case::bedrock_claude_platform(&BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG)]
#[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
#[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
#[case::deepseek(&DEEPSEEK_MESSAGES_CONFIG)]
#[case::edenai(&EDENAI_MESSAGES_CONFIG)]
#[case::minimax(&MINIMAX_MESSAGES_CONFIG)]
#[case::openrouter(&OPENROUTER_MESSAGES_CONFIG)]
#[case::tencent(&TENCENT_MESSAGES_CONFIG)]
#[case::vertex_ai(&VERTEX_AI_MESSAGES_CONFIG)]
fn every_messages_provider_retries_like_the_python_base_config(
    #[case] config: &dyn BaseMessagesConfig,
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
            let mut body = table["request"].clone();
            body["thinking"] = case["thinking"].clone();
            let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
            let expected = case["retried"].as_bool().unwrap().then(|| {
                if case["stripped"].as_bool().unwrap() {
                    RetriedRequest {
                        reasons: vec![RetryReason::ThinkingSignature],
                        body: table["stripped_request"].clone(),
                    }
                } else {
                    RetriedRequest {
                        reasons: Vec::new(),
                        body: body.clone(),
                    }
                }
            });
            config.retry_after_http_error(
                status,
                case["error_text"].as_str().unwrap(),
                &body,
                &MessagesTransformContext::default(),
            ) != expected
        })
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    assert_eq!(mismatched, Vec::<&str>::new());
}

#[rstest]
#[case::anthropic(&ANTHROPIC_MESSAGES_CONFIG)]
#[case::azure_ai(&AZURE_AI_MESSAGES_CONFIG)]
#[case::bedrock_invoke(&BEDROCK_INVOKE_MESSAGES_CONFIG)]
#[case::bedrock_claude_platform(&BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG)]
#[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
#[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
#[case::deepseek(&DEEPSEEK_MESSAGES_CONFIG)]
#[case::edenai(&EDENAI_MESSAGES_CONFIG)]
#[case::minimax(&MINIMAX_MESSAGES_CONFIG)]
#[case::openrouter(&OPENROUTER_MESSAGES_CONFIG)]
#[case::tencent(&TENCENT_MESSAGES_CONFIG)]
#[case::vertex_ai(&VERTEX_AI_MESSAGES_CONFIG)]
fn every_messages_provider_drops_rejected_strict_tools_like_the_python_base_config(
    #[case] config: &dyn BaseMessagesConfig,
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
            let mut body = replay["request"].clone();
            body["thinking"] = case["thinking"].clone();
            body["tools"] = table["tools"].clone();
            let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
            let expected = case["retried"].as_bool().unwrap().then(|| {
                let mut reasons = Vec::new();
                let mut expected = if case["stripped"].as_bool().unwrap() {
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

#[rstest]
#[case::anthropic(&ANTHROPIC_MESSAGES_CONFIG)]
#[case::azure_ai(&AZURE_AI_MESSAGES_CONFIG)]
#[case::bedrock_invoke(&BEDROCK_INVOKE_MESSAGES_CONFIG)]
#[case::bedrock_claude_platform(&BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG)]
#[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
#[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
#[case::deepseek(&DEEPSEEK_MESSAGES_CONFIG)]
#[case::edenai(&EDENAI_MESSAGES_CONFIG)]
#[case::minimax(&MINIMAX_MESSAGES_CONFIG)]
#[case::openrouter(&OPENROUTER_MESSAGES_CONFIG)]
#[case::tencent(&TENCENT_MESSAGES_CONFIG)]
#[case::vertex_ai(&VERTEX_AI_MESSAGES_CONFIG)]
fn every_messages_provider_renames_rejected_tool_use_ids_like_the_python_base_config(
    #[case] config: &dyn BaseMessagesConfig,
) {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/anthropic_messages_http_error_retry.json"
    ))
    .unwrap();
    let request = &fixture["http_error_retry"]["request"];
    let table = &fixture["tool_use_id_retry"];
    let cases = table["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    let mismatched: Vec<&str> = cases
        .iter()
        .filter(|case| {
            let mut body = request.clone();
            body["messages"] = if case["renamable"].as_bool().unwrap() {
                table["messages"].clone()
            } else {
                table["renamed_messages"].clone()
            };
            let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
            let expected = case["retried"].as_bool().unwrap().then(|| {
                let mut renamed = body.clone();
                renamed["messages"] = table["renamed_messages"].clone();
                RetriedRequest {
                    reasons: Vec::new(),
                    body: renamed,
                }
            });
            config.retry_after_http_error(
                status,
                case["error_text"].as_str().unwrap(),
                &body,
                &MessagesTransformContext::default(),
            ) != expected
        })
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    assert_eq!(mismatched, Vec::<&str>::new());
}
