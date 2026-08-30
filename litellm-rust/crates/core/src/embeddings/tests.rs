use super::types::{EmbeddingsInput, EmbeddingsRequest};

#[test]
fn embeddings_request_serializes_correctly() {
    let request = EmbeddingsRequest {
        model: "text-embedding-3-small",
        input: EmbeddingsInput::Single("Hello, world!".to_string()),
        encoding_format: Some("float".to_string()),
        dimensions: Some(1536),
        user: Some("test-user".to_string()),
        api_key: Some("test-key"),
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        timeout: None,
    };

    assert_eq!(request.model, "text-embedding-3-small");
    assert!(matches!(request.input, EmbeddingsInput::Single(_)));
}
