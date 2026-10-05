use litellm_cache::{
    Error, SemanticCacheContext,
    semantic::{
        Embedder, PreparedEmbedding, prompt_from_context, prompt_from_messages, str_from_messages,
    },
};
use rstest::rstest;
use serde_json::{Value, json};

fn context(messages: Option<Value>, input: Option<Value>) -> SemanticCacheContext {
    SemanticCacheContext {
        messages,
        input,
        ..SemanticCacheContext::default()
    }
}

#[rstest]
#[case::empty(json!([]), "")]
#[case::string_content(json!([{"role": "user", "content": "hello"}]), "hello")]
#[case::concatenates_messages(
    json!([{"role": "system", "content": "be brief. "}, {"role": "user", "content": "hello"}]),
    "be brief. hello",
)]
#[case::text_parts(
    json!([{"role": "user", "content": [
        {"type": "text", "text": "What is "},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "text", "text": "this?"},
    ]}]),
    "What is this?",
)]
#[case::missing_null_and_empty_content(
    json!([{"role": "assistant"}, {"role": "assistant", "content": null}, {"role": "user", "content": ""}]),
    "",
)]
#[case::search_results_hidden_behind_small_content(
    json!([{"role": "tool", "content": "small", "search_results": [
        {"source": "s", "title": "t", "content": [{"text": "hidden payload"}]},
    ]}]),
    "smallsthidden payload",
)]
#[case::title_only_search_result(
    json!([{"role": "tool", "content": "small", "search_results": [
        {"source": "s", "title": "long title", "content": []},
    ]}]),
    "smallslong title",
)]
#[case::search_results_without_content(
    json!([{"role": "tool", "search_results": [{"source": "s", "title": "t"}]}]),
    "st",
)]
#[case::search_result_fields_in_python_order(
    json!([{"role": "tool", "content": "c", "search_results": [
        {"citations": {"enabled": true}, "content": [{"text": "body"}], "title": "t", "source": "s"},
        {"source": "s2"},
    ]}]),
    r#"cstbody{"enabled":true}s2"#,
)]
#[case::null_citations_skipped(
    json!([{"role": "tool", "content": "c", "search_results": [
        {"source": "s", "citations": null},
    ]}]),
    "cs",
)]
#[case::non_string_and_non_object_entries_skipped(
    json!([{"role": "tool", "content": "c", "search_results": [
        "junk",
        {"source": 1, "title": null, "content": ["junk", {"text": 3}, {"text": "kept"}]},
    ]}]),
    "ckept",
)]
#[case::non_list_search_results_skipped(
    json!([{"role": "tool", "content": "c", "search_results": {"source": "s"}}]),
    "c",
)]
#[case::citations_compact_in_insertion_order(
    json!([{"role": "tool", "search_results": [
        {"citations": {"z": 1, "a": [1.5, true, null], "m": {"k": "v"}}},
    ]}]),
    r#"{"z":1,"a":[1.5,true,null],"m":{"k":"v"}}"#,
)]
#[case::citations_ensure_ascii(
    json!([{"role": "tool", "search_results": [{"citations": ["caf\u{e9}", "\u{4e2d}"]}]}]),
    r#"["caf\u00e9","\u4e2d"]"#,
)]
#[case::citations_astral_chars_as_surrogate_pairs(
    json!([{"role": "tool", "search_results": [{"citations": "\u{1f600}"}]}]),
    r#""\ud83d\ude00""#,
)]
#[case::citations_escapes(
    json!([{"role": "tool", "search_results": [{"citations": "q\"\\\n\t\u{1}/"}]}]),
    r#""q\"\\\n\t\u0001/""#,
)]
#[case::citations_large_float_exponent(
    json!([{"role": "tool", "search_results": [{"citations": [1e20, 1.0]}]}]),
    "[1e+20,1.0]",
)]
#[case::citations_scalars(
    json!([{"role": "tool", "search_results": [{"citations": false}, {"citations": 3}]}]),
    "false3",
)]
fn str_from_messages_matches_python(#[case] messages: Value, #[case] expected: &str) {
    assert_eq!(str_from_messages(messages.as_array().unwrap()), expected);
}

#[rstest]
#[case::no_messages(None, None)]
#[case::empty_messages(Some(json!([])), None)]
#[case::messages_not_a_list(Some(json!("hello")), None)]
#[case::messages(Some(json!([{"content": "hello"}])), Some("hello"))]
#[case::messages_without_text(Some(json!([{"content": null}])), Some(""))]
fn prompt_from_messages_reads_messages_only(
    #[case] messages: Option<Value>,
    #[case] expected: Option<&str>,
) {
    let context = context(messages, Some(json!("responses prompt")));
    assert_eq!(prompt_from_messages(&context).as_deref(), expected);
}

#[rstest]
#[case::prefers_messages(
    Some(json!([{"content": "message prompt"}])),
    Some(json!("responses prompt")),
    Some("message prompt"),
)]
#[case::empty_messages_fall_back_to_input(
    Some(json!([])),
    Some(json!("responses prompt")),
    Some("responses prompt"),
)]
#[case::messages_without_text_keep_an_empty_prompt(
    Some(json!([{"content": null}])),
    Some(json!("x")),
    Some(""),
)]
#[case::nothing(None, None, None)]
#[case::null_input(None, Some(Value::Null), None)]
#[case::blank_string(None, Some(json!("   ")), None)]
#[case::trimmed_string(
    None,
    Some(json!("  What is the capital of France?\n")),
    Some("What is the capital of France?"),
)]
#[case::image_only(
    None,
    Some(json!([{"type": "input_image", "image_url": "https://example.com"}])),
    None,
)]
#[case::structured_input(
    None,
    Some(json!([{"role": "user", "content": [
        {"type": "input_text", "text": "What is the capital of France?"},
        {"type": "input_text", "text": "Answer briefly."},
        {"type": "input_image", "image_url": "https://example.com/paris.png"},
    ]}])),
    Some("What is the capital of France?\nAnswer briefly."),
)]
#[case::model_objects_after_dump(
    None,
    Some(json!([
        {"content": [{"text": "model dump prompt"}]},
        {"content": [{"output_text": "dict prompt"}]},
        {"content": [{"input_text": "inline prompt"}]},
        {"content": [{"type": "input_image", "image_url": "https://example.com"}]},
    ])),
    Some("model dump prompt\ndict prompt\ninline prompt"),
)]
#[case::object_content(
    None,
    Some(json!({"content": [{"text": "object content prompt"}]})),
    Some("object content prompt"),
)]
#[case::string_content(None, Some(json!({"content": "  inline  "})), Some("inline"))]
#[case::null_content_uses_text_keys(
    None,
    Some(json!({"content": null, "output": "tool output"})),
    Some("tool output"),
)]
#[case::content_wins_over_text(None, Some(json!({"content": [], "text": "ignored"})), None)]
#[case::text_key_precedence(
    None,
    Some(json!({"output_text": "d", "input_text": "c", "output": "b", "text": "a"})),
    Some("a"),
)]
#[case::input_text_key(None, Some(json!({"input_text": "only input"})), Some("only input"))]
#[case::output_text_key(None, Some(json!({"output_text": "only output"})), Some("only output"))]
#[case::non_string_text_keys_skipped(
    None,
    Some(json!({"text": 1, "output": "fallback"})),
    Some("fallback"),
)]
#[case::nested_lists(None, Some(json!([["a", [" b "]], "", "c"])), Some("a\nb\nc"))]
#[case::scalars_ignored(None, Some(json!([1, true, null, "kept"])), Some("kept"))]
fn prompt_from_context_matches_python(
    #[case] messages: Option<Value>,
    #[case] input: Option<Value>,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        prompt_from_context(&context(messages, input)).as_deref(),
        expected
    );
}

/// Python `test_redis_semantic_cache_prompt_extraction_skips_blank_dict_text_keys`: a blank
/// text key falls through to the next one.
#[rstest]
#[case::blank_text_falls_through(
    json!({"text": "   ", "input_text": "fallback prompt"}),
    "fallback prompt",
)]
fn prompt_from_context_skips_blank_text_keys(#[case] input: Value, #[case] expected: &str) {
    assert_eq!(
        prompt_from_context(&context(None, Some(input))).as_deref(),
        Some(expected)
    );
}

/// Where `json.dumps(..., separators=(",", ":"))` and Python `str.strip` differ from
/// `semantic.rs`: ensure_ascii escapes DEL, small floats keep Python's two-digit exponent, and
/// strip also removes the ASCII information separators.
#[rstest]
#[case::del_is_escaped(json!([{"search_results": [{"citations": "\u{7f}"}]}]), None, r#""\u007f""#)]
#[case::small_float_exponent(json!([{"search_results": [{"citations": 1.5e-7}]}]), None, "1.5e-07")]
#[case::float_at_positional_floor(json!([{"search_results": [{"citations": 1e-4}]}]), None, "0.0001")]
#[case::float_at_scientific_ceiling(json!([{"search_results": [{"citations": 1e16}]}]), None, "1e+16")]
#[case::large_float(json!([{"search_results": [{"citations": [1.25e20, -2.5, 3.0]}]}]), None, "[1.25e+20,-2.5,3.0]")]
#[case::strip_information_separators(json!([]), Some(json!("\u{1c}a\u{1f}")), "a")]
fn python_serialization_edge_cases(
    #[case] messages: Value,
    #[case] input: Option<Value>,
    #[case] expected: &str,
) {
    let actual = match input {
        Some(input) => prompt_from_context(&context(None, Some(input))).unwrap_or_default(),
        None => str_from_messages(messages.as_array().unwrap()),
    };
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn prepared_embedding_returns_its_vector_for_any_prompt() {
    let embedding = PreparedEmbedding(vec![0.1, 0.2, 0.3]);
    let metadata = json!({"tenant": "team"});
    assert_eq!(embedding.embed("a", None), Ok(vec![0.1, 0.2, 0.3]));
    assert_eq!(
        embedding.async_embed("b", Some(&metadata)).await,
        Ok(vec![0.1, 0.2, 0.3])
    );
}

#[rstest]
#[tokio::test]
async fn embedders_default_to_async_only() {
    struct AsyncOnly;

    impl Embedder for AsyncOnly {
        async fn async_embed(&self, prompt: &str, _: Option<&Value>) -> Result<Vec<f32>, Error> {
            Ok(vec![prompt.len() as f32])
        }
    }

    assert_eq!(
        AsyncOnly.embed("abc", None),
        Err(Error::UnsupportedOperation)
    );
    assert_eq!(AsyncOnly.async_embed("abc", None).await, Ok(vec![3.0]));
}
