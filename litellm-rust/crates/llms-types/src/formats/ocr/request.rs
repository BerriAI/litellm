use std::collections::BTreeMap;

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type")]
pub enum OcrDocument {
    #[serde(rename = "document_url")]
    DocumentUrl {
        document_url: String,
        #[serde(flatten)]
        extra_fields: BTreeMap<String, Option<String>>,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: String,
        #[serde(flatten)]
        extra_fields: BTreeMap<String, Option<String>>,
    },
}

impl OcrDocument {
    pub fn source(&self) -> &str {
        match self {
            Self::DocumentUrl { document_url, .. } => document_url,
            Self::ImageUrl { image_url, .. } => image_url,
        }
    }

    pub fn is_remote(&self) -> bool {
        let source = self.source();
        source.starts_with("http://") || source.starts_with("https://")
    }

    pub fn with_source(self, source: String) -> Self {
        match self {
            Self::DocumentUrl { extra_fields, .. } => Self::DocumentUrl {
                document_url: source,
                extra_fields,
            },
            Self::ImageUrl { extra_fields, .. } => Self::ImageUrl {
                image_url: source,
                extra_fields,
            },
        }
    }
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Default, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OcrResponseFormat {
    #[default]
    Litellm,
    Native,
}

#[cfg(test)]
mod tests {
    use crate::formats::ocr::OcrDocument;

    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    fn document_rejects_non_string_provider_fields() {
        assert!(
            serde_json::from_value::<OcrDocument>(json!({
                "type": "image_url", "image_url": "https://example.com/image", "detail": 42
            }))
            .is_err()
        );
    }

    #[rstest]
    #[case::document_url("document_url", "document_name", "application/pdf")]
    #[case::image_url("image_url", "detail", "image/png")]
    fn document_variants_preserve_provider_fields_when_rewriting_sources(
        #[case] kind: &str,
        #[case] field: &str,
        #[case] mime_type: &str,
        #[values(json!("kept"), Value::Null)] extra: Value,
    ) {
        let original = "https://example.com/input";
        let replacement = format!("data:{mime_type};base64,AA==");
        let document: OcrDocument =
            serde_json::from_value(json!({"type": kind, kind: original, field: extra})).unwrap();
        assert_eq!(document.source(), original);
        assert!(document.is_remote());
        let rewritten = document.with_source(replacement.clone());
        assert!(!rewritten.is_remote());
        assert_eq!(
            serde_json::to_value(rewritten).unwrap(),
            json!({"type": kind, kind: replacement, field: extra})
        );
    }
}
