use rstest::rstest;

use super::*;

/// Expected counts are pinned from `litellm.token_counter(model="claude-sonnet-4-5", ...)`
/// so this test also guards Python parity.
fn counter() -> TokenCounter {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
    );
    let json = std::fs::read_to_string(path).expect("anthropic tokenizer json is in the repo");
    TokenCounter::from_json(&json).expect("anthropic tokenizer loads")
}

const SIMPLE: &str = r#"{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"Hello, how are you today?"}]}"#;

const BLOCKS_AND_SYSTEM: &str = r#"{"model":"claude-sonnet-4-5","messages":[
  {"role":"system","content":"You are a terse assistant."},
  {"role":"user","name":"alice","content":[
    {"type":"text","text":"Summarise this paragraph about ships and harbours."},
    "plain string item",
    {"type":"thinking","thinking":"pondering"},
    {"type":"tool_reference","tool_name":"get_weather"}]},
  {"role":"assistant","content":[{"type":"text","text":"Sure.","cache_control":{"type":"ephemeral"}}]}]}"#;

const TOOLS_OPENAI: &str = r#"{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"weather?"}],
  "tools":[
    {"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{
      "type":"object",
      "properties":{
        "location":{"type":"string","description":"City name"},
        "unit":{"type":"string","enum":["celsius","fahrenheit"]},
        "days":{"type":"integer"},
        "tags":{"type":"array","items":{"type":"string"}},
        "opts":{"type":"object","properties":{"verbose":{"type":"boolean"},"level":{"type":"integer","enum":[1,2]}},"required":["verbose"]},
        "anything":{}},
      "required":["location"]}}},
    {"type":"function","function":{"name":"noop"}}],
  "tool_choice":{"type":"function","function":{"name":"get_weather"}}}"#;

const TOOLS_ANTHROPIC_SYSTEM: &str = r#"{"model":"claude-sonnet-4-5",
  "messages":[{"role":"system","content":"sys"},{"role":"user","content":"weather?"}],
  "tools":[{"name":"get_weather","description":"Get weather","input_schema":{
    "type":"object","properties":{"location":{"type":["string","null"]}},"required":["location"]}}],
  "tool_choice":"none"}"#;

#[rstest]
#[case::text_only(SIMPLE, 14)]
#[case::content_blocks_name_and_system(BLOCKS_AND_SYSTEM, 45)]
#[case::openai_tools_named_choice(TOOLS_OPENAI, 123)]
#[case::anthropic_tools_system_discount_choice_none(TOOLS_ANTHROPIC_SYSTEM, 53)]
fn count_request_matches_python_token_counter(#[case] body: &str, #[case] expected: usize) {
    let request = CountableRequest::parse(body.as_bytes()).expect("fixture parses");
    let count = counter().count_request(&request).expect("fixture counts");
    assert_eq!(
        count,
        InputTokenCount {
            model: "claude-sonnet-4-5".to_string(),
            input_tokens: expected,
        }
    );
}

#[test]
fn tool_definitions_render_like_python() {
    let request = CountableRequest::parse(TOOLS_OPENAI.as_bytes()).expect("fixture parses");
    let rendered = format_function_definitions(request.tools.as_deref().unwrap_or_default())
        .expect("fixture renders");
    let expected = "namespace functions {\n\n// Get weather\ntype get_weather = (_: {\n// City name\nlocation: string,\nunit?: \"celsius\" | \"fahrenheit\",\ndays?: number,\ntags?: string[],\nopts?: {\n  verbose: boolean,\n  level?: \"1\" | \"2\",\n},\nanything?: any,\n}) => any;\n\ntype noop = () => any;\n\n} // namespace functions";
    assert_eq!(rendered, expected);
}

#[test]
fn union_types_and_anthropic_schema_render_like_python() {
    let request =
        CountableRequest::parse(TOOLS_ANTHROPIC_SYSTEM.as_bytes()).expect("fixture parses");
    let rendered = format_function_definitions(request.tools.as_deref().unwrap_or_default())
        .expect("fixture renders");
    assert_eq!(
        rendered,
        "namespace functions {\n\n// Get weather\ntype get_weather = (_: {\nlocation: any,\n}) => any;\n\n} // namespace functions"
    );
}

#[rstest]
#[case::not_json(b"not json" as &[u8])]
#[case::missing_model(br#"{"messages":[]}"#)]
#[case::messages_not_a_list(br#"{"model":"m","messages":"hi"}"#)]
#[case::message_with_tool_calls(
    br#"{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{}"}}]}]}"#
)]
#[case::dict_content(
    br#"{"model":"m","messages":[{"role":"user","content":{"type":"text","text":"x"}}]}"#
)]
#[case::float_enum(
    br#"{"model":"m","messages":[],"tools":[{"name":"f","input_schema":{"type":"object","properties":{"x":{"type":"number","enum":[1.5]}}}}]}"#
)]
#[case::anthropic_tool_choice_without_function(
    br#"{"model":"m","messages":[],"tool_choice":{"type":"auto"}}"#
)]
fn shapes_outside_the_mirror_are_declined_at_parse(#[case] body: &[u8]) {
    assert!(matches!(
        CountableRequest::parse(body),
        Err(TokenCountError::Unsupported(_))
    ));
}

#[rstest]
#[case::no_messages(br#"{"model":"m"}"# as &[u8])]
#[case::image_block(
    br#"{"model":"m","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]}"#
)]
#[case::tool_result_block(
    br#"{"model":"m","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"ok"}]}]}"#
)]
#[case::array_without_items(
    br#"{"model":"m","messages":[],"tools":[{"name":"f","input_schema":{"type":"object","properties":{"x":{"type":"array"}}}}]}"#
)]
fn shapes_outside_the_mirror_are_declined_at_count(#[case] body: &[u8]) {
    let request = CountableRequest::parse(body).expect("shape parses");
    assert!(matches!(
        counter().count_request(&request),
        Err(TokenCountError::Unsupported(_))
    ));
}

#[test]
fn tool_choice_and_system_discount_change_the_count() {
    let counter = counter();
    let count = |body: &str| {
        counter
            .count_request(&CountableRequest::parse(body.as_bytes()).expect("parses"))
            .expect("counts")
            .input_tokens
    };
    let base = count(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(
        count(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tool_choice":"none"}"#),
        base + TOOL_CHOICE_NONE_TOKENS
    );
    assert_eq!(
        count(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tool_choice":"auto"}"#),
        base
    );
    let with_tools = count(
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f"}]}"#,
    );
    let with_tools_and_system = count(
        r#"{"model":"m","messages":[{"role":"system","content":"hi"}],"tools":[{"name":"f"}]}"#,
    );
    assert_eq!(
        with_tools - with_tools_and_system,
        TOOLS_WITH_SYSTEM_MESSAGE_DISCOUNT
    );
    assert_eq!(
        count(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[]}"#),
        base
    );
}

#[test]
fn loading_a_bad_tokenizer_is_a_load_error() {
    assert!(matches!(
        TokenCounter::from_json("{}"),
        Err(TokenCountError::Load(_))
    ));
}
