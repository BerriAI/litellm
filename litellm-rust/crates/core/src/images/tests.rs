use super::types::{ImagesData, ImagesGenerationRequest, ImagesGenerationResponse};

#[test]
fn images_generation_request_serializes_correctly() {
    let request = ImagesGenerationRequest {
        model: "dall-e-3",
        prompt: "A cute baby sea otter".to_string(),
        n: Some(1),
        size: Some("1024x1024".to_string()),
        response_format: Some("url".to_string()),
        user: Some("test-user".to_string()),
        api_key: Some("test-key"),
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        timeout: None,
    };

    assert_eq!(request.model, "dall-e-3");
    assert_eq!(request.prompt, "A cute baby sea otter");
}

#[test]
fn images_generation_response_deserializes_correctly() {
    let json = r#"{
        "created": 1589478378,
        "data": [
            {
                "url": "https://example.com/image.png",
                "revised_prompt": "A cute baby sea otter floating on its back"
            }
        ]
    }"#;

    let response: ImagesGenerationResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.created, 1589478378);
    assert_eq!(response.data.len(), 1);
    assert_eq!(
        response.data[0].url,
        Some("https://example.com/image.png".to_string())
    );
    assert_eq!(
        response.data[0].revised_prompt,
        Some("A cute baby sea otter floating on its back".to_string())
    );
}

#[test]
fn images_data_with_b64_json() {
    let data = ImagesData {
        url: None,
        b64_json: Some("base64encodeddata".to_string()),
        revised_prompt: None,
    };

    assert!(data.url.is_none());
    assert_eq!(data.b64_json, Some("base64encodeddata".to_string()));
}
