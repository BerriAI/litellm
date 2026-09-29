pub mod options;

use serde_json::{Map, Value};

#[macro_rules_attribute::apply(wire_type)]
#[serde(transparent)]
pub struct ReductoFileId(pub String);

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoV3Request<P = Map<String, Value>> {
    pub input: ReductoFileId,
    #[serde(flatten)]
    pub params: P,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoLegacyRequest {
    pub document_url: ReductoFileId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ReductoLegacyOptions>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoLegacyOptions {
    pub enhance: Value,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoUploadResponse {
    pub file_id: Option<String>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoBoundingBox {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
    pub page: u64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub original_page: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
pub enum ReductoBlockType {
    Header,
    Footer,
    Title,
    #[serde(rename = "Section Header")]
    SectionHeader,
    #[serde(rename = "Page Number")]
    PageNumber,
    #[serde(rename = "List Item")]
    ListItem,
    Figure,
    Table,
    #[serde(rename = "Key Value")]
    KeyValue,
    Text,
    Comment,
    Signature,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoGranularConfidence {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub extract_confidence: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub parse_confidence: Option<Option<f64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoMergedTable {
    pub content: String,
    pub bbox: ReductoBoundingBox,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub image_url: Option<Option<String>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoParseBlock {
    #[serde(rename = "type")]
    pub kind: crate::recognized::Recognized<ReductoBlockType>,
    pub bbox: ReductoBoundingBox,
    pub content: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub image_url: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub chart_data: Option<Option<Vec<String>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub confidence: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub granular_confidence: Option<Option<ReductoGranularConfidence>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub merged_tables: Option<Option<Vec<ReductoMergedTable>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub extra: Option<Option<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoParseChunk {
    pub content: String,
    pub embed: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub enriched: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub enrichment_success: Option<Option<bool>>,
    pub blocks: Vec<ReductoParseBlock>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoOcrEntry {
    pub text: String,
    pub bbox: ReductoBoundingBox,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub confidence: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub chunk_index: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub rotation: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoOcrResult {
    pub words: Vec<ReductoOcrEntry>,
    pub lines: Vec<ReductoOcrEntry>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ReductoParseResult {
    Full {
        chunks: Vec<ReductoParseChunk>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "serde_with::rust::double_option::deserialize"
        )]
        ocr: Option<Option<ReductoOcrResult>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "serde_with::rust::double_option::deserialize"
        )]
        custom: Option<Option<Value>>,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
    Url {
        url: String,
        result_id: String,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoParseUsage {
    pub num_pages: u64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub credits: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub non_empty_cell_count: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct ReductoParseResponse {
    pub job_id: String,
    pub duration: f64,
    pub usage: ReductoParseUsage,
    pub result: crate::recognized::Recognized<ReductoParseResult>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub pdf_url: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub studio_link: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub document_properties: Option<Option<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{ReductoLegacyRequest, ReductoParseResponse, ReductoParseResult, ReductoV3Request};
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    fn parse_requests_preserve_nested_options() {
        let wire = json!({
            "input":"reducto://document", "settings":{"return_images":["table"]},
            "retrieval":{"chunking":{"chunk_mode":"page"}}, "future_option":null
        });
        let request: ReductoV3Request = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
        let legacy_wire =
            json!({"document_url":"reducto://document","options":{"enhance":{"agentic":true}}});
        let legacy: ReductoLegacyRequest = serde_json::from_value(legacy_wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(legacy).unwrap(), legacy_wire);
    }

    #[rstest]
    #[case::full(json!({"type":"full","chunks":[{
        "content":"text","embed":"embedding text","enriched":null,"blocks":[{
            "type":"Table","content":"table","bbox":{"left":0.1,"top":0.2,"width":0.8,"height":0.3,"page":2},
            "granular_confidence":{"parse_confidence":0.9,"extract_confidence":null},"future_field":true
        }]
    }]}))]
    #[case::url(json!({"type":"url","url":"https://example.com/result","result_id":"id"}))]
    #[case::future(json!({"type":"future-result","payload":{"nested":null}}))]
    fn native_result_variants_preserve_typed_data_and_extensions(#[case] result: Value) {
        let wire = json!({
            "job_id":"job-id", "duration":1.0, "result":result,
            "usage":{"num_pages":2,"credits":3.5,"credit_breakdown":{"parse":3.5}},"future_field":null
        });
        let response: ReductoParseResponse = serde_json::from_value(wire.clone()).unwrap();
        match response.result.known() {
            Some(ReductoParseResult::Full { chunks, .. }) => {
                assert_eq!(chunks[0].blocks[0].bbox.page, 2)
            }
            Some(ReductoParseResult::Url { result_id, .. }) => {
                assert_eq!(result_id.as_str(), result["result_id"].as_str().unwrap())
            }
            None => assert_eq!(result["type"], "future-result"),
        }
        assert_eq!(serde_json::to_value(response).unwrap(), wire);
    }

    #[rstest]
    #[case::url_missing_id(json!({"type":"url","url":"https://example.com/result"}))]
    #[case::invalid_block(json!({"type":"full","chunks":[{"content":"text","embed":"text","blocks":[{"type":"Text","content":"text","bbox":{"page":"bad"}}]}]}))]
    fn typed_native_result_rejects_malformed_known_shapes(#[case] wire: Value) {
        assert!(serde_json::from_value::<ReductoParseResult>(wire).is_err());
    }
}
