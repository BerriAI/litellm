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

const COMPLETIONS_PROMPT: &str =
    r#"{"model":"claude-sonnet-4-5","prompt":"Write a haiku about ships."}"#;

const COMPLETIONS_PROMPT_LIST: &str =
    r#"{"model":"claude-sonnet-4-5","prompt":["first prompt","second prompt"]}"#;

const RESPONSES_INPUT: &str = r#"{"model":"claude-sonnet-4-5","input":[
  {"role":"user","content":[{"type":"input_text","text":"Summarise caf\u00e9 menus, na\u00efve \u2014 ok? \"quoted\"\n"}]},
  {"role":"assistant","content":"Sure."}],"instructions":"be terse"}"#;

const EMBEDDINGS_TOKEN_IDS: &str =
    r#"{"model":"claude-sonnet-4-5","input":[[101,2023,5],[7]],"encoding_format":"float"}"#;

const RERANK: &str = r#"{"model":"claude-sonnet-4-5","query":"best harbour",
  "documents":["doc one",{"text":"doc two","title":"T","n":3,"ok":true,"none":null,"tags":["a","b"]}]}"#;

/// Expected counts are pinned from
/// `litellm.proxy.spend_tracking.budget_reservation._count_input_tokens(body, "claude-sonnet-4-5")`.
#[rstest]
#[case::text_only(SIMPLE, 14)]
#[case::content_blocks_name_and_system(BLOCKS_AND_SYSTEM, 45)]
#[case::openai_tools_named_choice(TOOLS_OPENAI, 123)]
#[case::anthropic_tools_system_discount_choice_none(TOOLS_ANTHROPIC_SYSTEM, 53)]
#[case::completions_prompt(COMPLETIONS_PROMPT, 7)]
#[case::completions_prompt_list(COMPLETIONS_PROMPT_LIST, 4)]
#[case::responses_input_items(RESPONSES_INPUT, 62)]
#[case::embeddings_token_ids(EMBEDDINGS_TOKEN_IDS, 5)]
#[case::rerank_query_and_documents(RERANK, 41)]
fn count_request_matches_python_token_counter(#[case] body: &str, #[case] expected: usize) {
    let request = CountableRequest::parse(body.as_bytes()).expect("fixture parses");
    let count = counter().count_request(&request).expect("fixture counts");
    assert_eq!(
        count,
        InputTokenCount {
            model: Some("claude-sonnet-4-5".to_string()),
            input_tokens: expected,
        }
    );
}

#[rstest]
#[case::null_messages_win_over_prompt(r#"{"model":"m","messages":null,"prompt":"ignored"}"#, 3)]
#[case::model_from_route(r#"{"prompt":"hi"}"#, 1)]
#[case::bools_and_ints_use_python_str(r#"{"model":"m","prompt":[true,false,42]}"#, 3)]
#[case::null_prompt_counts_zero(r#"{"model":"m","prompt":null}"#, 0)]
fn key_presence_follows_python(#[case] body: &str, #[case] expected: usize) {
    let request = CountableRequest::parse(body.as_bytes()).expect("fixture parses");
    let count = counter().count_request(&request).expect("fixture counts");
    assert_eq!(count.input_tokens, expected);
}

#[test]
fn objects_dump_like_python_json_dumps() {
    let body = r#"{"model":"m","input":{"text":"caf\u00e9 \u2014 \ud83d\ude00 \"q\" \\ \n\t\u0001\u007f ~","n":-3,"ok":true,"no":false,"none":null,"list":[1,"a",{"z":[]}],"empty":{}}}"#;
    let request = CountableRequest::parse(body.as_bytes()).expect("fixture parses");
    let dumped = python_json::dumps(request.input.as_ref().expect("input is present"))
        .expect("fixture dumps");
    assert_eq!(
        dumped,
        r#"{"text": "caf\u00e9 \u2014 \ud83d\ude00 \"q\" \\ \n\t\u0001\u007f ~", "n": -3, "ok": true, "no": false, "none": null, "list": [1, "a", {"z": []}], "empty": {}}"#
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
#[case::no_countable_input(br#"{"model":"m","instructions":"hi"}"# as &[u8])]
#[case::float_prompt(br#"{"model":"m","prompt":1.5}"#)]
#[case::float_inside_document(br#"{"model":"m","documents":[{"score":0.5}]}"#)]
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
