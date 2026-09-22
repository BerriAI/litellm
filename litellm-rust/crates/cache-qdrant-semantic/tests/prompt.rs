use litellm_cache_qdrant_semantic::prompt_from_messages;
use serde_json::json;

#[test]
fn prompt_matches_python_message_content_rules() {
    let messages = vec![
        json!({"role": "user", "content": "hello"}),
        json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "world"},
                {"type": "image_url", "image_url": {"url": "ignored"}},
                {"type": "text", "text": "!"},
            ],
        }),
    ];

    assert_eq!(prompt_from_messages(&messages), "helloworld!");
}

#[test]
fn prompt_includes_search_result_text_and_compact_citations() {
    let messages = vec![json!({
        "role": "tool",
        "content": null,
        "search_results": [{
            "source": "source",
            "title": "title",
            "content": [{"text": "body"}],
            "citations": {"page": 1, "section": "intro"},
        }],
    })];

    assert_eq!(
        prompt_from_messages(&messages),
        r#"sourcetitlebody{"page":1,"section":"intro"}"#
    );
}
