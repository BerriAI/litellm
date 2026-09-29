use serde_json::{Map, Value};

use crate::formats::ocr::{OcrDocument, OcrPage, OcrUsageInfo};

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum MistralOcrPages {
    Range(String),
    Indices(Vec<u64>),
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MistralOcrTableFormat {
    Html,
    Markdown,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MistralOcrConfidenceGranularity {
    Word,
    Page,
    Block,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MistralOcrOptions {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub pages: Option<Option<MistralOcrPages>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub include_image_base64: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub image_limit: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub image_min_size: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub bbox_annotation_format: Option<Option<Map<String, Value>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub document_annotation_format: Option<Option<Map<String, Value>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub document_annotation_prompt: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub table_format: Option<Option<crate::recognized::Recognized<MistralOcrTableFormat>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub extract_header: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub extract_footer: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub include_blocks: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub confidence_scores_granularity:
        Option<Option<crate::recognized::Recognized<MistralOcrConfidenceGranularity>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub id: Option<Option<String>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralOcrRequest<P = Map<String, Value>> {
    pub model: String,
    pub document: OcrDocument,
    #[serde(flatten)]
    pub params: P,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MistralOcrResponse {
    #[serde(default)]
    pub pages: Vec<OcrPage>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serde_with::rust::double_option::serialize",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub model: Option<Option<String>>,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<OcrUsageInfo>,

    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{MistralOcrOptions, MistralOcrRequest, MistralOcrResponse};
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::document(json!({"type":"document_url","document_url":"https://example.com/document.pdf"}))]
    #[case::image(json!({"type":"image_url","image_url":"data:image/png;base64,AA=="}))]
    fn request_preserves_document_and_unvalidated_options(#[case] document: Value) {
        let wire = json!({
            "model":"ocr-model", "document":document, "pages":null,
            "table_format":"html", "include_image_base64":false,
            "document_annotation_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}},
            "future_option":{"nested":[1,null,true]}
        });
        let request: MistralOcrRequest = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }

    #[rstest]
    #[case::indices(json!([0, 2]))]
    #[case::range(json!("0,2-4"))]
    #[case::null(json!(null))]
    fn typed_options_preserve_page_selection_and_explicit_null(#[case] pages: Value) {
        let wire = json!({
            "model":"ocr-model", "document":{"type":"document_url","document_url":"https://example.com/document.pdf"},
            "pages":pages, "extract_header":false, "include_image_base64":null,
            "table_format":"html", "future_option":{"nested":null}
        });
        let request: MistralOcrRequest<MistralOcrOptions> =
            serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(request.params.extract_header, Some(Some(false)));
        assert_eq!(request.params.include_image_base64, Some(None));
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }

    #[rstest]
    #[case::missing(json!({}), None)]
    #[case::null(json!({"model":null}), Some(None))]
    #[case::present(json!({"model":"returned-model"}), Some(Some("returned-model".into())))]
    fn response_preserves_model_presence_and_extensions(
        #[case] model_field: Value,
        #[case] expected: Option<Option<String>>,
    ) {
        let wire = Value::Object(
            json!({
                "pages":[{"index":0,"markdown":"text","blocks":[{"type":"text","bbox":{"x":1}}]}],
                "future_field":{"nested":null}
            })
            .as_object()
            .unwrap()
            .iter()
            .chain(model_field.as_object().unwrap())
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        );
        let response: MistralOcrResponse = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(response.model, expected);
        let encoded = serde_json::to_value(&response).unwrap();
        assert_eq!(encoded.get("model"), wire.get("model"));
        assert_eq!(encoded["pages"][0]["blocks"], wire["pages"][0]["blocks"]);
        assert_eq!(encoded["future_field"], wire["future_field"]);
        assert_eq!(
            serde_json::from_value::<MistralOcrResponse>(encoded).unwrap(),
            response
        );
    }

    #[rstest]
    #[case::model(json!({"model":false}))]
    #[case::page(json!({"pages":[{"index":0,"markdown":false}]}))]
    #[case::usage(json!({"usage_info":{"pages_processed":1.5}}))]
    fn response_rejects_malformed_typed_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<MistralOcrResponse>(wire).is_err());
    }
}
