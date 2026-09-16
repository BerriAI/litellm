use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;

use litellm_auth::{InputSource, Sourced, TokenProviderHandle};

use super::hooks::{NoopOcrHooks, OcrHooks};
use super::provider_config::{OcrConfigKind, resolve_provider_config};
use crate::call_arguments::CallArguments;
use crate::constants::OCR_HTTP_TIMEOUT_SECS;
use crate::serde_compat::{FiniteF64, LaxI64};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OcrDocument {
    #[serde(rename = "document_url")]
    DocumentUrl {
        document_url: String,
        #[serde(flatten)]
        extra_fields: BTreeMap<String, String>,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: String,
        #[serde(flatten)]
        extra_fields: BTreeMap<String, String>,
    },
}

impl OcrDocument {
    pub(crate) fn source(&self) -> &str {
        match self {
            Self::DocumentUrl { document_url, .. } => document_url,
            Self::ImageUrl { image_url, .. } => image_url,
        }
    }

    pub(crate) fn with_source(self, source: String) -> Self {
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrResponseFormat {
    #[default]
    Litellm,
    Native,
}

#[derive(Clone)]
pub struct OcrConnection {
    pub api_key: Option<String>,
    pub dynamic_api_key: Option<Sourced<String>>,
    pub api_key_source: InputSource,
    pub api_base: Option<String>,
    pub dynamic_api_base: Option<Sourced<String>>,
    pub api_base_source: InputSource,
    pub extra_headers: Vec<(String, String)>,
    pub extra_headers_source: InputSource,
    pub timeout: Duration,
    pub max_download_bytes: u64,
    pub max_response_bytes: usize,
    pub poll_timeout: Duration,
}

impl Default for OcrConnection {
    fn default() -> Self {
        Self {
            api_key: None,
            dynamic_api_key: None,
            api_key_source: InputSource::Deployment,
            api_base: None,
            dynamic_api_base: None,
            api_base_source: InputSource::Deployment,
            extra_headers: Vec::new(),
            extra_headers_source: InputSource::Deployment,
            timeout: Duration::from_secs(OCR_HTTP_TIMEOUT_SECS),
            max_download_bytes: crate::constants::OCR_DOWNLOAD_MAX_BYTES,
            max_response_bytes: crate::constants::OCR_RESPONSE_MAX_BYTES,
            poll_timeout: Duration::from_secs(crate::constants::OCR_POLL_TIMEOUT_SECS),
        }
    }
}

pub struct LiteLLMOcrRequest {
    pub model: String,
    pub document: OcrDocument,
    pub connection: OcrConnection,
    pub hooks: Arc<dyn OcrHooks>,
    pub litellm_call_id: Option<String>,
    pub optional_params: CallArguments,
    pub input_sources: BTreeMap<String, InputSource>,
    pub azure_ad_token_provider: Option<TokenProviderHandle>,
    pub(crate) config: OcrConfigKind,
}

impl LiteLLMOcrRequest {
    pub fn new(
        model: String,
        document: OcrDocument,
        custom_llm_provider: Option<&str>,
        optional_params: CallArguments,
    ) -> Result<Self, super::Error> {
        let (model, config) = resolve_provider_config(&model, custom_llm_provider)?;

        Ok(Self {
            model,
            document,
            connection: OcrConnection::default(),
            hooks: Arc::new(NoopOcrHooks),
            litellm_call_id: None,
            optional_params,
            input_sources: BTreeMap::new(),
            azure_ad_token_provider: None,
            config,
        })
    }

    pub(crate) fn response_format(&self) -> Result<OcrResponseFormat, super::Error> {
        self.optional_params
            .get("req_format")
            .filter(|value| !value.is_null())
            .map(|value| {
                serde_json::from_value(value.clone()).map_err(|_| super::Error::RequestFormat)
            })
            .transpose()
            .map(|format| format.unwrap_or_default())
    }

    pub fn provider_name(&self) -> &'static str {
        self.config.provider().into()
    }

    pub fn with_host_hooks(
        self,
        hooks: Arc<dyn OcrHooks>,
        litellm_call_id: Option<String>,
    ) -> Self {
        Self {
            hooks,
            litellm_call_id,
            ..self
        }
    }
}

#[serde_as]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPageDimensions {
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub dpi: Option<i64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub height: Option<i64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub width: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPageImage {
    pub image_base64: Option<String>,
    pub bbox: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[serde_as]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPage {
    #[serde_as(deserialize_as = "LaxI64")]
    pub index: i64,
    pub markdown: String,
    pub images: Option<Vec<OcrPageImage>>,
    pub dimensions: Option<OcrPageDimensions>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[serde_as]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrUsageInfo {
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub pages_processed: Option<i64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub pages_processed_annotation: Option<i64>,
    #[serde_as(deserialize_as = "Option<FiniteF64>")]
    pub credits: Option<f64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub doc_size_bytes: Option<i64>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LiteLLMOcrResponse {
    pub pages: Vec<OcrPage>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<OcrUsageInfo>,
    pub content: Option<String>,
    pub tables: Option<Vec<Map<String, Value>>>,
    #[serde(rename = "keyValuePairs")]
    pub key_value_pairs: Option<Vec<Map<String, Value>>>,
    #[serde(default = "ocr_object")]
    pub object: String,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_native_response: Option<Map<String, Value>>,
}

impl LiteLLMOcrResponse {
    pub fn new(model: impl Into<String>, pages: Vec<OcrPage>) -> Self {
        Self {
            pages,
            model: model.into(),
            document_annotation: None,
            usage_info: None,
            content: None,
            tables: None,
            key_value_pairs: None,
            object: ocr_object(),
            extra_fields: Map::new(),
            provider_native_response: None,
        }
    }

    pub fn into_json(self) -> Value {
        serde_json::to_value(self).expect("OCR response fields are JSON-compatible")
    }
}

fn ocr_object() -> String {
    "ocr".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalized_response_rejects_invalid_shared_fields() {
        for fields in [
            json!({"pages":[{}]}),
            json!({"pages":[{"index":0,"markdown":false}]}),
            json!({"pages":[{"index":0,"markdown":"","images":[{"bbox":[]}]}]}),
            json!({"usage_info":{"pages_processed":1.5}}),
            json!({"tables":[false]}),
            json!({"keyValuePairs":[[]]}),
            json!({"provider_native_response":[]}),
        ] {
            let payload: Map<String, Value> = json!({"model":"model", "pages":[]})
                .as_object()
                .unwrap()
                .iter()
                .chain(fields.as_object().unwrap())
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            assert!(serde_json::from_value::<LiteLLMOcrResponse>(Value::Object(payload)).is_err());
        }
        assert!(
            serde_json::from_value::<OcrDocument>(json!({
                "type":"image_url", "image_url":"https://example.com/image", "detail":42
            }))
            .is_err()
        );
    }

    #[test]
    fn numeric_coercion_preserves_integer_precision_and_rejects_fractional_values() {
        for (value, expected) in [
            (json!("9007199254740993.0"), 9_007_199_254_740_993),
            (json!("+2.000"), 2),
            (json!("1_000"), 1000),
            (json!(true), 1),
            (json!(2.0), 2),
        ] {
            let page: OcrPage =
                serde_json::from_value(json!({"index":value,"markdown":""})).unwrap();
            assert_eq!(page.index, expected);
        }
        for value in [
            json!("1e2"),
            json!(".0"),
            json!("2."),
            json!("_2"),
            json!("2__0"),
            json!(2.5),
            json!(null),
        ] {
            assert!(
                serde_json::from_value::<OcrPage>(json!({"index":value,"markdown":""})).is_err()
            );
        }
    }

    #[test]
    fn document_variants_preserve_provider_fields_when_rewriting_sources() {
        for (value, original, replacement, expected) in [
            (
                json!({
                    "type":"document_url",
                    "document_url":"https://example.com/input.pdf",
                    "document_name":"input.pdf"
                }),
                "https://example.com/input.pdf",
                "data:application/pdf;base64,AA==",
                json!({
                    "type":"document_url",
                    "document_url":"data:application/pdf;base64,AA==",
                    "document_name":"input.pdf"
                }),
            ),
            (
                json!({
                    "type":"image_url",
                    "image_url":"https://example.com/input.png",
                    "detail":"high"
                }),
                "https://example.com/input.png",
                "data:image/png;base64,AA==",
                json!({
                    "type":"image_url",
                    "image_url":"data:image/png;base64,AA==",
                    "detail":"high"
                }),
            ),
        ] {
            let document: OcrDocument = serde_json::from_value(value).unwrap();
            assert_eq!(document.source(), original);
            assert_eq!(
                serde_json::to_value(document.with_source(replacement.into())).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn response_serialization_flattens_extra_fields_and_omits_absent_native_response() {
        let response = LiteLLMOcrResponse {
            extra_fields: json!({"provider_field":"kept"})
                .as_object()
                .unwrap()
                .clone(),
            ..LiteLLMOcrResponse::new("model", vec![])
        };
        let serialized = response.into_json();
        assert_eq!(serialized["provider_field"], "kept");
        assert!(serialized.get("provider_native_response").is_none());
    }
}
