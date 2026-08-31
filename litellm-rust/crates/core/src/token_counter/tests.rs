use serde_json::{Map, Value, json};

use super::encoding::resolve;
use super::formatting::format_function_definitions;
use super::token_counter;
use super::types::TokenCounterRequest;

fn msg(role: &str, content: &str) -> Map<String, Value> {
    Map::from_iter([
        ("role".to_string(), json!(role)),
        ("content".to_string(), json!(content)),
    ])
}

fn msg_with_name(role: &str, content: &str, name: &str) -> Map<String, Value> {
    Map::from_iter([
        ("role".to_string(), json!(role)),
        ("content".to_string(), json!(content)),
        ("name".to_string(), json!(name)),
    ])
}

fn req<'a>(
    model: &'a str,
    text: Option<&'a str>,
    messages: Option<&'a [Map<String, Value>]>,
) -> TokenCounterRequest<'a> {
    TokenCounterRequest {
        model,
        text,
        messages,
        tools: None,
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: None,
    }
}

fn fmt_tools(tools: &[Map<String, Value>]) -> String {
    let mut out = String::new();
    format_function_definitions(tools, &mut out);
    out
}

#[test]
fn resolves_cl100k_base_for_unknown_models() {
    let tokenizer = resolve("unknown-model-xyz").unwrap();
    let count = tokenizer.count("hello world");
    let expected = tiktoken::get_encoding("cl100k_base").unwrap();
    assert_eq!(count, expected.count("hello world"));
}

#[test]
fn resolves_o200k_base_for_gpt4o() {
    let tokenizer = resolve("gpt-4o").unwrap();
    let count = tokenizer.count("hello world");
    let expected = tiktoken::get_encoding("o200k_base").unwrap();
    assert_eq!(count, expected.count("hello world"));
}

#[test]
fn normalizes_azure_model_names() {
    let tokenizer = resolve("gpt-35-turbo").unwrap();
    let count = tokenizer.count("hello");
    let expected = tiktoken::get_encoding("cl100k_base").unwrap();
    assert_eq!(count, expected.count("hello"));
}

#[test]
fn counts_empty_text_as_zero() {
    let tokenizer = resolve("gpt-4").unwrap();
    assert_eq!(tokenizer.count(""), 0);
}

#[test]
fn counts_unicode_text() {
    let tokenizer = resolve("gpt-4").unwrap();
    let count = tokenizer.count("á");
    assert!(count > 0);
}

#[test]
fn text_counter_returns_correct_count() {
    let result = token_counter(&req("gpt-4", Some("hello world"), None)).unwrap();
    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    assert_eq!(result, enc.count("hello world"));
}

#[test]
fn text_counter_with_gpt4o_uses_o200k() {
    let result = token_counter(&req("gpt-4o", Some("hello world"), None)).unwrap();
    let enc = tiktoken::get_encoding("o200k_base").unwrap();
    assert_eq!(result, enc.count("hello world"));
}

#[test]
fn rejects_both_text_and_messages() {
    let messages = vec![msg("user", "hi")];
    let result = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: Some("hello"),
        messages: Some(&messages),
        tools: None,
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: None,
    });
    assert!(result.is_err());
}

#[test]
fn rejects_neither_text_nor_messages() {
    let result = token_counter(&req("gpt-4", None, None));
    assert!(result.is_err());
}

#[test]
fn rejects_tools_with_text() {
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({"name": "test", "description": "test"}),
    )])];
    let result = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: Some("hello"),
        messages: None,
        tools: Some(&tools),
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: None,
    });
    assert!(result.is_err());
}

#[test]
fn counts_single_user_message() {
    let messages = vec![msg("user", "Hello, how are you?")];
    assert_eq!(
        token_counter(&req("gpt-4", None, Some(&messages))).unwrap(),
        13
    );
}

#[test]
fn counts_system_message() {
    let messages = vec![msg("system", "You are a bot.")];
    assert_eq!(
        token_counter(&req("gpt-4", None, Some(&messages))).unwrap(),
        12
    );
}

#[test]
fn counts_message_with_name_field() {
    let messages = vec![msg_with_name(
        "system",
        "New synergies will help drive top-line growth.",
        "example_user",
    )];
    assert_eq!(
        token_counter(&req("gpt-4", None, Some(&messages))).unwrap(),
        20
    );
}

#[test]
fn gpt35_turbo_0301_produces_valid_count() {
    let messages = vec![msg("user", "Hello, how are you?")];
    let result = token_counter(&req("gpt-3.5-turbo-0301", None, Some(&messages))).unwrap();
    assert!(result > 0);
}

#[test]
fn counts_empty_messages_as_zero_plus_overhead() {
    let messages: Vec<Map<String, Value>> = vec![];
    assert_eq!(
        token_counter(&req("gpt-4", None, Some(&messages))).unwrap(),
        3
    );
}

#[test]
fn counts_tool_call_arguments() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("assistant")),
        ("content".to_string(), json!("")),
        (
            "tool_calls".to_string(),
            json!([
                {"function": {"arguments": "{\"location\": \"Boston\"}"}}
            ]),
        ),
    ])];
    let result = token_counter(&req("gpt-4", None, Some(&messages))).unwrap();
    assert!(result > 3);
}

#[test]
fn format_function_definitions_simple_tool() {
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "search_sources",
            "description": "Retrieve sources from the Azure AI Search index",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Query string to retrieve documents"
                    }
                },
                "required": ["search_query"]
            }
        }),
    )])];

    let formatted = fmt_tools(&tools);
    assert!(formatted.contains("namespace functions {"));
    assert!(formatted.contains("type search_sources = (_: {"));
    assert!(formatted.contains("search_query: string,"));
    assert!(formatted.contains("} // namespace functions"));
}

#[test]
fn format_function_definitions_no_parameters() {
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "search_sources",
            "description": "Retrieve sources from the Azure AI Search index"
        }),
    )])];

    let formatted = fmt_tools(&tools);
    assert!(formatted.contains("type search_sources = () => any;"));
}

#[test]
fn format_function_definitions_with_enum() {
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "summarize_order",
            "description": "Summarize the customer order request",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {
                        "type": "string",
                        "enum": ["meals", "days"]
                    }
                },
                "required": ["unit"]
            }
        }),
    )])];

    let formatted = fmt_tools(&tools);
    assert!(formatted.contains("\"meals\" | \"days\""));
}

#[test]
fn format_function_definitions_with_array() {
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "get_coordinates",
            "description": "Get addresses",
            "parameters": {
                "type": "object",
                "properties": {
                    "addresses": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["addresses"]
            }
        }),
    )])];

    assert!(fmt_tools(&tools).contains("string[]"));
}

#[test]
fn format_function_definitions_with_nested_object() {
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "data_demonstration",
            "description": "This is the main function description",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_1": {
                        "type": "object",
                        "description": "The object data type as a property",
                        "properties": {
                            "string1": {"type": "string"}
                        }
                    }
                },
                "required": ["object_1"]
            }
        }),
    )])];

    let formatted = fmt_tools(&tools);
    assert!(formatted.contains("object_1: {"));
    assert!(formatted.contains("string1?: string,"));
}

#[test]
fn tool_choice_none_adds_one_token() {
    let messages = vec![msg("system", "You are a bot.")];
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "search_sources",
            "description": "Retrieve sources",
            "parameters": {
                "type": "object",
                "properties": {"search_query": {"type": "string", "description": "Query"}},
                "required": ["search_query"]
            }
        }),
    )])];

    let result_none = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages),
        tools: Some(&tools),
        tool_choice: Some(&json!("none")),
        count_response_tokens: false,
        default_token_count: None,
    })
    .unwrap();

    let result_auto = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages),
        tools: Some(&tools),
        tool_choice: Some(&json!("auto")),
        count_response_tokens: false,
        default_token_count: None,
    })
    .unwrap();

    assert_eq!(result_none, result_auto + 1);
}

#[test]
fn tool_choice_dict_adds_function_name_tokens() {
    let messages = vec![msg("system", "You are a bot.")];
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "search_sources",
            "description": "Retrieve sources",
            "parameters": {
                "type": "object",
                "properties": {"search_query": {"type": "string", "description": "Query"}},
                "required": ["search_query"]
            }
        }),
    )])];

    let result_named = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages),
        tools: Some(&tools),
        tool_choice: Some(&json!({"type": "function", "function": {"name": "search_sources"}})),
        count_response_tokens: false,
        default_token_count: None,
    })
    .unwrap();

    let result_auto = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages),
        tools: Some(&tools),
        tool_choice: Some(&json!("auto")),
        count_response_tokens: false,
        default_token_count: None,
    })
    .unwrap();

    let name_tokens = tiktoken::get_encoding("cl100k_base")
        .unwrap()
        .count("search_sources");
    assert_eq!(result_named, result_auto + 7 + name_tokens);
}

#[test]
fn count_response_tokens_skips_tool_overhead() {
    let messages = vec![msg("system", "You are a bot.")];
    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "search_sources",
            "description": "Retrieve sources",
            "parameters": {
                "type": "object",
                "properties": {"search_query": {"type": "string"}},
                "required": ["search_query"]
            }
        }),
    )])];

    let with_tools = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages),
        tools: Some(&tools),
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: None,
    })
    .unwrap();

    let response_only = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages),
        tools: Some(&tools),
        tool_choice: None,
        count_response_tokens: true,
        default_token_count: None,
    })
    .unwrap();

    assert!(with_tools > response_only);
}

#[test]
fn image_url_content_returns_unsupported() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("user")),
        (
            "content".to_string(),
            json!([
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]),
        ),
    ])];

    let result = token_counter(&req("gpt-4", None, Some(&messages)));
    assert!(matches!(result, Err(crate::CoreError::Unsupported(_))));
}

#[test]
fn image_url_with_default_token_count_returns_fallback() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("user")),
        (
            "content".to_string(),
            json!([
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]),
        ),
    ])];

    let result = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages),
        tools: None,
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: Some(250),
    })
    .unwrap();

    assert_eq!(result, 250);
}

#[test]
fn text_content_list_counts_correctly() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("user")),
        (
            "content".to_string(),
            json!([{"type": "text", "text": "Hello, how are you?"}]),
        ),
    ])];

    assert_eq!(
        token_counter(&req("gpt-4", None, Some(&messages))).unwrap(),
        13
    );
}

#[test]
fn system_message_with_tools_subtracts_four() {
    let messages_with_system = vec![msg("system", "You are a bot."), msg("user", "Hello")];
    let messages_without_system = vec![msg("user", "Hello")];

    let tools = vec![Map::from_iter([(
        "function".to_string(),
        json!({
            "name": "test",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }),
    )])];

    let with_system = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages_with_system),
        tools: Some(&tools),
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: None,
    })
    .unwrap();

    let without_system = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages_without_system),
        tools: Some(&tools),
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: None,
    })
    .unwrap();

    let enc = tiktoken::get_encoding("cl100k_base").unwrap();
    let system_msg_tokens = 3 + enc.count("system") + enc.count("You are a bot.");
    assert_eq!(without_system + system_msg_tokens - 4, with_system);
}

#[test]
fn search_results_content_counts_text() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("user")),
        ("content".to_string(), json!("What is Rust?")),
        (
            "search_results".to_string(),
            json!([
                {"text": "Rust is a systems programming language."},
                {"text": "It focuses on safety and performance."}
            ]),
        ),
    ])];

    let with_search = token_counter(&req("gpt-4", None, Some(&messages))).unwrap();

    let messages_no_search = vec![Map::from_iter([
        ("role".to_string(), json!("user")),
        ("content".to_string(), json!("What is Rust?")),
    ])];
    let without_search = token_counter(&req("gpt-4", None, Some(&messages_no_search))).unwrap();

    assert!(with_search > without_search);
}

#[test]
fn search_results_empty_list_counts_zero() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("user")),
        ("content".to_string(), json!("hello")),
        ("search_results".to_string(), json!([])),
    ])];

    let with_empty = token_counter(&req("gpt-4", None, Some(&messages))).unwrap();

    let messages_no_search = vec![msg("user", "hello")];
    let without_search = token_counter(&req("gpt-4", None, Some(&messages_no_search))).unwrap();

    assert_eq!(with_empty, without_search);
}

#[test]
fn anthropic_tool_use_returns_unsupported_without_default() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("user")),
        (
            "content".to_string(),
            json!([{"type": "tool_use", "id": "123", "name": "test", "input": {}}]),
        ),
    ])];

    let result = token_counter(&req("gpt-4", None, Some(&messages)));
    assert!(matches!(result, Err(crate::CoreError::Unsupported(_))));
}

#[test]
fn anthropic_tool_use_returns_default_when_set() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("user")),
        (
            "content".to_string(),
            json!([{"type": "tool_use", "id": "123", "name": "test", "input": {}}]),
        ),
    ])];

    let result = token_counter(&TokenCounterRequest {
        model: "gpt-4",
        text: None,
        messages: Some(&messages),
        tools: None,
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: Some(100),
    })
    .unwrap();

    assert_eq!(result, 100);
}

#[test]
fn function_call_legacy_format_counts_arguments() {
    let messages = vec![Map::from_iter([
        ("role".to_string(), json!("assistant")),
        ("content".to_string(), json!(null)),
        (
            "function_call".to_string(),
            json!({"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}),
        ),
    ])];

    let result = token_counter(&req("gpt-4", None, Some(&messages))).unwrap();
    assert!(result > 3);
}

#[test]
fn model_name_normalization_is_zero_alloc_for_known_models() {
    use super::encoding::normalized_model_name;
    use std::borrow::Cow;

    let name = normalized_model_name("gpt-4");
    assert!(matches!(name, Cow::Borrowed(_)));

    let name = normalized_model_name("gpt-35-turbo");
    assert!(matches!(name, Cow::Owned(_)));
    assert_eq!(name.as_ref(), "gpt-3.5-turbo");
}

#[test]
fn format_function_definitions_anthropic_tool_shape() {
    let tools = vec![Map::from_iter([
        ("name".to_string(), json!("search")),
        ("description".to_string(), json!("Search the web")),
        (
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
        ),
    ])];

    let formatted = fmt_tools(&tools);
    assert!(formatted.contains("type search = (_: {"));
    assert!(formatted.contains("query: string,"));
}

#[test]
fn hf_tokenizer_attempted_for_claude_2() {
    let result = resolve("claude-2");
    if let Ok(tokenizer) = result {
        let count = tokenizer.count("hello world");
        assert!(count > 0);
    } else {
        assert!(
            result.is_err(),
            "claude-2 should error when HF tokenizer unavailable"
        );
    }
}

#[test]
fn hf_tokenizer_not_attempted_for_claude_3() {
    let tokenizer = resolve("claude-3-opus").unwrap();
    let count = tokenizer.count("hello world");
    let expected = tiktoken::get_encoding("cl100k_base").unwrap();
    assert_eq!(count, expected.count("hello world"));
}

#[test]
fn resolved_tokenizer_unified_api() {
    let tiktoken_tok = resolve("gpt-4").unwrap();
    let count = tiktoken_tok.count("hello world");
    assert!(count > 0);

    let hf_result = resolve("claude-2");
    if let Ok(hf_tok) = hf_result {
        let count = hf_tok.count("hello world");
        assert!(count > 0);
    }
}
