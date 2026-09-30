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
#[case::empty(r#"[]"#, r#""#)]
#[case::string_content(r#"[{"role": "user", "content": "hello"}]"#, r#"hello"#)]
#[case::concatenates_messages(
    r#"[{"role": "system", "content": "be brief. "}, {"role": "user", "content": "hello"}]"#,
    r#"be brief. hello"#
)]
#[case::text_parts(
    r#"[{"role": "user", "content": [{"type": "text", "text": "What is "}, {"type": "text", "text": "this?"}]}]"#,
    r#"What is this?"#,
)]
#[case::image_part_kept_in_place(
    r#"[{"role": "user", "content": [{"type": "text", "text": "What is "}, {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}, {"type": "text", "text": "this?"}]}]"#,
    r#"What is {"type":"image_url","image_url":{"url":"https://example.com/a.png"}}this?"#,
)]
#[case::missing_null_and_empty_content(
    r#"[{"role": "assistant"}, {"role": "assistant", "content": null}, {"role": "user", "content": ""}]"#,
    r#""#,
)]
#[case::non_object_messages_skipped(
    r#"["junk", 3, {"role": "user", "content": "kept"}]"#,
    r#"kept"#
)]
#[case::cache_control_ignored(
    r#"[{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]"#,
    r#"hi"#,
)]
#[case::extra_message_field_kept(
    r#"[{"role": "user", "name": "alice", "content": "hi"}]"#,
    r#"{"role":"user","name":"alice","content":"hi"}"#
)]
#[case::unknown_block_type_kept(
    r#"[{"role": "user", "content": [{"type": "brand_new_block", "payload": {"k": "v"}}, {"type": "text", "text": "!"}]}]"#,
    r#"{"type":"brand_new_block","payload":{"k":"v"}}!"#,
)]
#[case::non_string_text_kept(
    r#"[{"role": "user", "content": [{"type": "text", "text": 3}]}]"#,
    r#"{"type":"text","text":3}"#
)]
#[case::scalar_content_kept(
    r#"[{"role": "user", "content": 7}, {"role": "user", "content": [1.5, true, "x"]}]"#,
    r#"71.5truex"#
)]
#[case::search_results_kept(
    r#"[{"role": "tool", "content": "small", "search_results": [{"source": "s", "title": "t", "content": [{"type": "text", "text": "hidden payload"}], "citations": {"enabled": true}}]}]"#,
    r#"{"role":"tool","content":"small","search_results":[{"source":"s","title":"t","content":[{"type":"text","text":"hidden payload"}],"citations":{"enabled":true}}]}"#,
)]
#[case::anthropic_tool_use(
    r#"[{"role": "user", "content": "fix the failing test"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}}]}]"#,
    r#"fix the failing test{"type":"tool_use","id":1,"name":"Bash","input":{"cmd":"ls"}}"#,
)]
#[case::anthropic_string_tool_result(
    r#"[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "calc.py"}]}]"#,
    r#"{"type":"tool_result","tool_use_id":1,"content":"calc.py"}"#,
)]
#[case::anthropic_nested_text_tool_result_then_text(
    r#"[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}, {"type": "text", "text": "next"}]}]"#,
    r#"{"type":"tool_result","tool_use_id":1,"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}next"#,
)]
#[case::openai_tool_calls_then_tool_result(
    r#"[{"role": "assistant", "content": "writing", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "write", "arguments": "{\"path\": \"a\"}"}}, {"id": "c2", "type": "function", "function": {"name": "write", "arguments": "{\"path\": \"b\"}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "ok"}]"#,
    r#"{"role":"assistant","content":"writing","tool_calls":[{"id":1,"type":"function","function":{"name":"write","arguments":"{\"path\": \"a\"}"}},{"id":2,"type":"function","function":{"name":"write","arguments":"{\"path\": \"b\"}"}}]}{"role":"tool","tool_call_id":1,"content":"ok"}"#,
)]
#[case::anthropic_parallel_tool_results_tagged_with_their_call(
    r#"[{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a"}}, {"type": "tool_use", "id": "t2", "name": "Read", "input": {"path": "b"}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "B"}, {"type": "tool_result", "tool_use_id": "t1", "content": "A"}]}]"#,
    r#"{"type":"tool_use","id":1,"name":"Read","input":{"path":"a"}}{"type":"tool_use","id":2,"name":"Read","input":{"path":"b"}}{"type":"tool_result","tool_use_id":2,"content":"B"}{"type":"tool_result","tool_use_id":1,"content":"A"}"#,
)]
#[case::reused_call_ids_keep_first_position(
    r#"[{"role": "assistant", "content": null, "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "a", "arguments": "{}"}}]}, {"role": "assistant", "content": null, "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "b", "arguments": "{}"}}, {"id": "call_1", "type": "function", "function": {"name": "c", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "C"}]"#,
    r#"{"role":"assistant","tool_calls":[{"id":1,"type":"function","function":{"name":"a","arguments":"{}"}}]}{"role":"assistant","tool_calls":[{"id":1,"type":"function","function":{"name":"b","arguments":"{}"}},{"id":2,"type":"function","function":{"name":"c","arguments":"{}"}}]}{"role":"tool","tool_call_id":2,"content":"C"}"#,
)]
#[case::tool_output_cannot_forge_a_record(
    r#"[{"role": "tool", "content": "x\"}{\"role\":\"tool\",\"content\":\"y"}]"#,
    r#"{"role":"tool","content":"x\"}{\"role\":\"tool\",\"content\":\"y"}"#
)]
#[case::malformed_tool_call_entries(
    r#"[{"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function"}, "junk"]}]"#,
    r#"{"role":"assistant","tool_calls":[{"id":1,"type":"function"},"junk"]}"#,
)]
#[case::anthropic_base64_image_digested(
    r#"[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgo"}}]}]"#,
    r#"{"type":"image","source":{"type":"base64","media_type":"image/png","data":"sha256:b1a66f8de3276946"}}"#,
)]
#[case::openai_data_url_digested(
    r#"[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgoiVBORw0KGgo"}}]}]"#,
    r#"{"type":"image_url","image_url":{"url":"sha256:19c7f028a60a33b8"}}"#,
)]
#[case::thinking_signature_digested(
    r#"[{"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "EqQBCkgIARABEqQBCkgIARABEqQBCkgIARABEqQBCkgIARABEqQBCkgIARABEqQBCkgIARABEqQBCkgIARABEqQBCkgIARABEqQBCkgIARABEqQBCkgIARAB"}, {"type": "text", "text": "done"}]}]"#,
    r#"{"type":"thinking","thinking":"hmm","signature":"sha256:b20cbbd82e4b3791"}done"#,
)]
#[case::non_ascii_escaped(
    r#"[{"role": "tool", "content": "caf\u00e9 \u4e2d \ud83d\ude00 \u007f q\"\\\n\t\u0001/"}]"#,
    r#"{"role":"tool","content":"caf\u00e9 \u4e2d \ud83d\ude00 \u007f q\"\\\n\t\u0001/"}"#
)]
#[case::non_ascii_plain_text_kept_raw(
    r#"[{"role": "user", "content": "caf\u00e9 \ud83d\ude00"}]"#,
    r#"café 😀"#
)]
#[case::floats_in_python_repr(
    r#"[{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {"a": [1e+20, 1.0, 1.5e-07, 0.0001, 1e+16, -2.5, 3]}}]}]"#,
    r#"{"type":"tool_use","id":1,"name":"n","input":{"a":[1e+20,1.0,1.5e-07,0.0001,1e+16,-2.5,3]}}"#,
)]
#[case::empty_fields_dropped(
    r#"[{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}, "extra": [], "note": ""}]}]"#,
    r#"{"type":"tool_use","id":1,"name":"n"}"#,
)]
fn str_from_messages_matches_python(#[case] messages: &str, #[case] expected: &str) {
    let messages: Vec<Value> = serde_json::from_str(messages).unwrap();
    assert_eq!(str_from_messages(&messages), expected);
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
    Some(r#"[{"content": "message prompt"}]"#),
    Some(r#""responses prompt""#),
    Some(r#"message prompt"#)
)]
#[case::empty_messages_fall_back_to_input(
    Some(r#"[]"#),
    Some(r#""responses prompt""#),
    Some(r#"responses prompt"#)
)]
#[case::messages_without_text_keep_an_empty_prompt(
    Some(r#"[{"content": null}]"#),
    Some(r#""x""#),
    Some(r#""#)
)]
#[case::null_input(None, None, None)]
#[case::blank_string(None, Some(r#""   ""#), None)]
#[case::trimmed_string(
    None,
    Some(r#""  What is the capital of France?\n""#),
    Some(r#"What is the capital of France?"#)
)]
#[case::strip_information_separators(None, Some(r#""\u001ca\u001f""#), Some(r#"a"#))]
#[case::image_only_kept(
    None,
    Some(r#"[{"type": "input_image", "image_url": "https://example.com"}]"#),
    Some(r#"{"type":"input_image","image_url":"https://example.com"}"#)
)]
#[case::structured_input(
    None,
    Some(r#"[{"role": "user", "content": [{"type": "input_text", "text": "What is the capital of France?"}, {"type": "input_text", "text": "Answer briefly."}, {"type": "input_image", "image_url": "https://example.com/paris.png"}]}]"#),
    Some(r#"What is the capital of France?
Answer briefly.
{"type":"input_image","image_url":"https://example.com/paris.png"}"#),
)]
#[case::message_items_with_ids(
    None,
    Some(r#"[{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "hi", "annotations": []}]}]"#),
    Some(r#"hi"#),
)]
#[case::object_content(
    None,
    Some(r#"{"content": [{"text": "object content prompt"}]}"#),
    Some(r#"object content prompt"#)
)]
#[case::string_content(None, Some(r#"{"content": "  inline  "}"#), Some(r#"inline"#))]
#[case::unknown_keys_kept(
    None,
    Some(r#"{"text": "   ", "input_text": "fallback prompt"}"#),
    Some(r#"{"text":"   ","input_text":"fallback prompt"}"#)
)]
#[case::nested_lists(
    None,
    Some(r#"[["a", [" b "]], "", "c"]"#),
    Some(
        r#"a
b
c"#
    )
)]
#[case::scalars_kept(
    None,
    Some(r#"[1, true, null, "kept"]"#),
    Some(
        r#"1
true
kept"#
    )
)]
#[case::blank_text_parts(None, Some(r#"[{"type": "input_text", "text": "  "}]"#), None)]
#[case::responses_function_call(
    None,
    Some(r#"[{"role": "user", "content": "update the config"}, {"type": "function_call", "call_id": "c1", "name": "write_file", "arguments": "{\"path\":\"a.yaml\"}"}, {"type": "function_call_output", "call_id": "c1", "output": "ok"}]"#),
    Some(r#"update the config
{"type":"function_call","call_id":1,"name":"write_file","arguments":"{\"path\":\"a.yaml\"}"}
{"type":"function_call_output","call_id":1,"output":"ok"}"#),
)]
#[case::responses_structured_function_call_output(
    None,
    Some(r#"[{"type": "function_call", "call_id": "c1", "name": "write_file", "arguments": "{}"}, {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": " denied "}]}]"#),
    Some(r#"{"type":"function_call","call_id":1,"name":"write_file","arguments":"{}"}
{"type":"function_call_output","call_id":1,"output":[{"type":"input_text","text":" denied "}]}"#),
)]
#[case::responses_parallel_outputs_tagged_with_their_call(
    None,
    Some(r#"[{"type": "function_call", "call_id": "c1", "name": "read", "arguments": "a"}, {"type": "function_call", "call_id": "c2", "name": "read", "arguments": "b"}, {"type": "function_call_output", "call_id": "c2", "output": "B"}, {"type": "function_call_output", "call_id": "c1", "output": "A"}]"#),
    Some(r#"{"type":"function_call","call_id":1,"name":"read","arguments":"a"}
{"type":"function_call","call_id":2,"name":"read","arguments":"b"}
{"type":"function_call_output","call_id":2,"output":"B"}
{"type":"function_call_output","call_id":1,"output":"A"}"#),
)]
#[case::responses_reasoning_digested(
    None,
    Some(r#"[{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAABgAAAAB"}]"#),
    Some(r#"{"type":"reasoning","id":1,"encrypted_content":"sha256:010b9eb723d8d1e2"}"#),
)]
fn prompt_from_context_matches_python(
    #[case] messages: Option<&str>,
    #[case] input: Option<&str>,
    #[case] expected: Option<&str>,
) {
    let parse = |value: &str| serde_json::from_str::<Value>(value).unwrap();
    let context = context(messages.map(parse), input.map(parse));
    assert_eq!(prompt_from_context(&context).as_deref(), expected);
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
