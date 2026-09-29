use serde_json::{Map, Value};

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u64>),
    TokenSequences(Vec<Vec<u64>>),
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingEncodingFormat {
    Float,
    Base64,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct EmbeddingRequest<E = Map<String, Value>> {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub encoding_format: Option<Option<EmbeddingEncodingFormat>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub dimensions: Option<Option<u64>>,
    #[serde(flatten)]
    pub extensions: E,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Integers(Vec<i64>),
    Floats(Vec<f64>),
    Base64(String),
}

#[macro_rules_attribute::apply(wire_type)]
pub struct EmbeddingData {
    pub index: u64,
    pub object: String,
    pub embedding: EmbeddingVector,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct EmbeddingResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<EmbeddingData>,
    pub usage: EmbeddingUsage,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{EmbeddingRequest, EmbeddingResponse};
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::text(json!("text"))]
    #[case::texts(json!(["first", "second"]))]
    #[case::tokens(json!([1, 2]))]
    #[case::token_sequences(json!([[1, 2], [3]]))]
    fn request_preserves_supported_input_shapes(#[case] input: Value) {
        let wire = json!({"model":"embedding-model","input":input,"future_field":null});
        let request: EmbeddingRequest = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }

    #[rstest]
    #[case::integers(json!([-2, 3]))]
    #[case::floats(json!([0.1, -0.2]))]
    #[case::base64(json!("AA=="))]
    fn response_preserves_vectors_and_extensions(#[case] embedding: Value) {
        let wire = json!({
            "id":"response", "object":"list", "model":"embedding-model",
            "data":[{"index":0,"object":"embedding","embedding":embedding,"future_field":null}],
            "usage":{"prompt_tokens":2,"total_tokens":2,"completion_tokens":0}
        });
        let response: EmbeddingResponse = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(response).unwrap(), wire);
    }

    #[rstest]
    #[case::object(json!({"text":"input"}))]
    #[case::mixed(json!(["text", 1]))]
    #[case::fractional_tokens(json!([1.5]))]
    fn request_rejects_malformed_input(#[case] input: Value) {
        assert!(
            serde_json::from_value::<EmbeddingRequest>(json!({"model":"model","input":input}))
                .is_err()
        );
    }
}
