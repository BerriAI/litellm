use std::collections::BTreeMap;

use bytes::Bytes;
use mime::Mime;
use serde_json::Value;
use url::Url;

use super::compiler::CompileError;
use super::policy::OcrCanonicalField;
use super::types::{Field, OcrDialectId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocumentKind {
    Image,
    Pdf,
}

#[derive(Clone, PartialEq)]
pub enum OcrDocument {
    RemoteUrl {
        kind: DocumentKind,
        url: Url,
    },
    Inline {
        kind: DocumentKind,
        media_type: Mime,
        bytes: Bytes,
    },
    ProviderReference {
        provider: OcrDialectId,
        id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageSelection(Vec<u32>);

impl PageSelection {
    pub fn new(pages: impl IntoIterator<Item = u32>) -> Self {
        Self(pages.into_iter().collect())
    }

    pub fn pages(&self) -> &[u32] {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnnotationFormat(Value);

impl AnnotationFormat {
    pub fn new(schema: Value) -> Result<Self, CompileError> {
        if !schema.is_object() {
            return Err(CompileError::InvalidParameter {
                field: "annotation_format",
                reason: "must be a JSON object",
            });
        }
        Ok(Self(schema))
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableFormat {
    Html,
    Markdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfidenceScoresGranularity {
    Page,
    Word,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OcrOutputOptions {
    pub include_image_base64: Field<bool>,
    pub image_limit: Field<u32>,
    pub image_min_size: Field<u32>,
    pub bbox_annotation_format: Field<AnnotationFormat>,
    pub document_annotation_format: Field<AnnotationFormat>,
    pub document_annotation_prompt: Field<String>,
    pub extract_header: Field<bool>,
    pub extract_footer: Field<bool>,
    pub table_format: Field<TableFormat>,
    pub confidence_scores_granularity: Field<ConfidenceScoresGranularity>,
    pub include_blocks: Field<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OcrRequestId(String);

impl OcrRequestId {
    pub fn new(value: impl Into<String>) -> Result<Self, CompileError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(CompileError::InvalidParameter {
                field: "id",
                reason: "must not be blank",
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExplicitProviderExtras {
    dialect: OcrDialectId,
    fields: BTreeMap<String, Value>,
}

impl ExplicitProviderExtras {
    pub fn try_new(
        dialect: OcrDialectId,
        fields: BTreeMap<String, Value>,
    ) -> Result<Self, CompileError> {
        if let Some(field) = fields.keys().find(|field| {
            OcrCanonicalField::from_wire_name(field).is_some() || is_litellm_control(field)
        }) {
            return Err(if OcrCanonicalField::from_wire_name(field).is_some() {
                CompileError::ExtraCollidesWithCanonical(field.clone())
            } else {
                CompileError::ReservedExtra(field.clone())
            });
        }
        Ok(Self { dialect, fields })
    }

    pub fn dialect(&self) -> OcrDialectId {
        self.dialect
    }

    pub fn fields(&self) -> &BTreeMap<String, Value> {
        &self.fields
    }
}

fn is_litellm_control(field: &str) -> bool {
    field.starts_with("litellm_")
        || matches!(
            field,
            "api_base"
                | "api_key"
                | "custom_llm_provider"
                | "fallbacks"
                | "metadata"
                | "mock_response"
                | "num_retries"
                | "request_timeout"
                | "retry_policy"
        )
}

/// ```compile_fail
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<litellm_core::ocr::canonical::CanonicalOcrRequest>();
/// ```
#[derive(Clone, PartialEq)]
pub struct CanonicalOcrRequest {
    pub model: String,
    pub document: OcrDocument,
    pub pages: Field<PageSelection>,
    pub output: OcrOutputOptions,
    pub request_id: Field<OcrRequestId>,
    pub provider_extras: ExplicitProviderExtras,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extras_reject_canonical_field_collisions() {
        let error = ExplicitProviderExtras::try_new(
            OcrDialectId::Mistral,
            BTreeMap::from([("pages".to_string(), json!([0]))]),
        )
        .expect_err("canonical fields cannot enter extras");

        assert_eq!(
            error,
            CompileError::ExtraCollidesWithCanonical("pages".to_string())
        );
    }

    #[test]
    fn extras_reject_litellm_controls() {
        let error = ExplicitProviderExtras::try_new(
            OcrDialectId::Mistral,
            BTreeMap::from([("request_timeout".to_string(), json!(30))]),
        )
        .expect_err("LiteLLM controls cannot enter extras");

        assert_eq!(
            error,
            CompileError::ReservedExtra("request_timeout".to_string())
        );

        let prefixed_error = ExplicitProviderExtras::try_new(
            OcrDialectId::Mistral,
            BTreeMap::from([("litellm_future_control".to_string(), json!(true))]),
        )
        .expect_err("reserved LiteLLM prefixes cannot enter extras");

        assert_eq!(
            prefixed_error,
            CompileError::ReservedExtra("litellm_future_control".to_string())
        );
    }

    #[test]
    fn extras_remain_bound_to_one_dialect() {
        let extras = ExplicitProviderExtras::try_new(
            OcrDialectId::ReductoV3,
            BTreeMap::from([("chunking".to_string(), json!({"size": 1}))]),
        )
        .expect("provider field is accepted");

        assert_eq!(extras.dialect(), OcrDialectId::ReductoV3);
        assert_eq!(extras.fields()["chunking"], json!({"size": 1}));
    }

    #[test]
    fn canonical_fields_preserve_absent_null_and_value() {
        assert_ne!(Field::<PageSelection>::Absent, Field::Null);
        assert_ne!(
            Field::Null,
            Field::Value(PageSelection::new([0_u32, 2_u32]))
        );
    }
}
