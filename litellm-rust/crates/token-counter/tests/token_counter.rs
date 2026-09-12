use rstest::rstest;
use serde::Deserialize;

use litellm_token_counter::{CountableRequest, Error, InputTokenCount, TokenCounter};

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
        Err(Error::RequestParse(_))
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
        Err(Error::MissingInput | Error::FloatText | Error::ContentBlock | Error::ArrayItems)
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
        base + 1
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
    assert_eq!(with_tools - with_tools_and_system, 4);
    assert_eq!(
        count(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[]}"#),
        base
    );
}

#[test]
fn loading_a_bad_tokenizer_is_a_load_error() {
    assert!(matches!(TokenCounter::from_json("{}"), Err(Error::Load(_))));
}

/// A tiktoken encoding: its fixture directory, the vendored rank file Python
/// loads, the constructor, and the model `generate.py` counted the requests for.
#[derive(Clone, Copy)]
struct TiktokenEncoding {
    fixtures: &'static str,
    rank_file: &'static str,
    load: fn(&str) -> Result<TokenCounter, Error>,
    model: &'static str,
}

const CL100K: TiktokenEncoding = TiktokenEncoding {
    fixtures: "cl100k",
    rank_file: "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
    load: TokenCounter::from_cl100k_ranks,
    model: "gpt-4",
};

const O200K: TiktokenEncoding = TiktokenEncoding {
    fixtures: "o200k",
    rank_file: "fb374d419588a4632f3f557e76b4b70aebbca790",
    load: TokenCounter::from_o200k_ranks,
    model: "gpt-4o",
};

fn tiktoken_counter(encoding: TiktokenEncoding) -> TokenCounter {
    let path = format!(
        "{}/../../../litellm/litellm_core_utils/tokenizers/{}",
        env!("CARGO_MANIFEST_DIR"),
        encoding.rank_file
    );
    let ranks = std::fs::read_to_string(&path).expect("rank file is in the repo");
    (encoding.load)(&ranks).expect("ranks load")
}

fn tiktoken_fixture(encoding: TiktokenEncoding, name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/{}/{name}",
        env!("CARGO_MANIFEST_DIR"),
        encoding.fixtures
    );
    std::fs::read_to_string(&path).expect("fixture generated by tests/fixtures/generate.py")
}

#[derive(Deserialize)]
struct TextFixture {
    text: String,
    tokens: usize,
}

#[derive(Deserialize)]
struct RequestFixture {
    body: String,
    input_tokens: usize,
}

/// Reference counts come from `tiktoken.get_encoding(name)`; see
/// `tests/fixtures/generate.py`.
#[rstest]
#[case::cl100k(CL100K)]
#[case::o200k(O200K)]
fn tiktoken_text_counts_match_tiktoken(#[case] encoding: TiktokenEncoding) {
    let counter = tiktoken_counter(encoding);
    let fixtures: Vec<TextFixture> = tiktoken_fixture(encoding, "texts.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).expect("fixture line is json"))
        .collect();
    assert!(fixtures.len() > 3000);
    let mismatches: Vec<_> = fixtures
        .iter()
        .filter_map(|fixture| {
            let count = counter.count_text(&fixture.text).expect("text counts");
            (count != fixture.tokens).then(|| (fixture.text.clone(), fixture.tokens, count))
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "(text, tiktoken, rust): {mismatches:?}"
    );
}

/// Reference counts come from the proxy's admission counter
/// (`_count_input_tokens(body, model)`), so this pins the shared message,
/// tool and reply-priming accounting on the tiktoken paths as well.
#[rstest]
#[case::cl100k(CL100K)]
#[case::o200k(O200K)]
fn tiktoken_request_counts_match_python_admission_counter(#[case] encoding: TiktokenEncoding) {
    let counter = tiktoken_counter(encoding);
    let fixtures: Vec<RequestFixture> = tiktoken_fixture(encoding, "requests.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).expect("fixture line is json"))
        .collect();
    let counts: Vec<usize> = fixtures
        .iter()
        .map(|fixture| {
            let request = CountableRequest::parse(fixture.body.as_bytes()).expect("fixture parses");
            let count = counter.count_request(&request).expect("fixture counts");
            assert_eq!(count.model.as_deref(), Some(encoding.model));
            assert_eq!(count.input_tokens, fixture.input_tokens, "{}", fixture.body);
            count.input_tokens
        })
        .collect();
    assert!(counts.last().is_some_and(|tokens| *tokens >= 50_000));
}

#[rstest]
#[case::cl100k(CL100K)]
#[case::o200k(O200K)]
fn tiktoken_shares_the_message_accounting_with_the_anthropic_path(
    #[case] encoding: TiktokenEncoding,
) {
    let counter = tiktoken_counter(encoding);
    let count = |body: &str| {
        counter
            .count_request(&CountableRequest::parse(body.as_bytes()).expect("parses"))
            .expect("counts")
            .input_tokens
    };
    let text = |text: &str| counter.count_text(text).expect("counts");
    let base = count(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(base, 3 + text("user") + text("hi") + 3);
    assert_eq!(
        count(r#"{"model":"m","messages":[{"role":"user","name":"al","content":"hi"}]}"#),
        base + text("al") + 1
    );
    assert_eq!(
        count(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tool_choice":"none"}"#),
        base + 1
    );
}

#[rstest]
#[case::empty("")]
#[case::not_base64("!!!! 0")]
#[case::missing_rank("YQ==")]
#[case::rank_not_a_number("YQ== x")]
#[case::single_byte_tokens_missing("YWI= 0")]
fn loading_a_bad_rank_file_is_a_load_error(
    #[case] rank_file: &str,
    #[values(CL100K, O200K)] encoding: TiktokenEncoding,
) {
    assert!(matches!((encoding.load)(rank_file), Err(Error::Ranks(_))));
}
