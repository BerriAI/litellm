use serde_json::{Map, Value};

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MistralEmbeddingDtype {
    Float,
    Int8,
    Uint8,
    Binary,
    Ubinary,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MistralEmbeddingOptions {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub output_dimension: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub output_dtype: Option<Option<MistralEmbeddingDtype>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{MistralEmbeddingDtype, MistralEmbeddingOptions};
    use crate::formats::embeddings::EmbeddingRequest;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::float(MistralEmbeddingDtype::Float)]
    #[case::int8(MistralEmbeddingDtype::Int8)]
    #[case::uint8(MistralEmbeddingDtype::Uint8)]
    #[case::binary(MistralEmbeddingDtype::Binary)]
    #[case::ubinary(MistralEmbeddingDtype::Ubinary)]
    fn typed_extensions_share_the_embedding_request_contract(#[case] dtype: MistralEmbeddingDtype) {
        let wire = json!({
            "model":"embedding-model", "input":["text"], "encoding_format":"base64",
            "output_dimension":64, "output_dtype":dtype, "metadata":{"tag":"value"}
        });
        let request: EmbeddingRequest<MistralEmbeddingOptions> =
            serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(request.extensions.output_dtype, Some(Some(dtype)));
        assert_eq!(request.extensions.output_dimension, Some(Some(64)));
        assert!(!request.extensions.extra_fields.contains_key("output_dtype"));
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }
}
