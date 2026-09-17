use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use litellm_auth::{InputSource, Sourced, TokenProviderHandle};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;

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

    pub(crate) fn is_remote(&self) -> bool {
        let source = self.source();
        source.starts_with("http://") || source.starts_with("https://")
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

impl TryFrom<Value> for OcrDocument {
    type Error = super::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        super::json::decode_request_value(value, "document")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OcrDocumentInput {
    Document(OcrDocument),
    Path {
        path: PathBuf,
        mime_type: Option<String>,
    },
    Bytes {
        bytes: Bytes,
        file_name: Option<String>,
        mime_type: Option<String>,
    },
    HostReader {
        mime_type: Option<String>,
    },
}

impl From<OcrDocument> for OcrDocumentInput {
    fn from(document: OcrDocument) -> Self {
        Self::Document(document)
    }
}

impl From<PathBuf> for OcrDocumentInput {
    fn from(path: PathBuf) -> Self {
        Self::Path {
            path,
            mime_type: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OcrFileContent {
    pub bytes: Bytes,
    pub file_name: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrResponseFormat {
    #[default]
    Litellm,
    Native,
}

#[derive(Clone, Default)]
pub struct OcrCredentialInputs {
    pub api_key: Option<Sourced<String>>,
    pub dynamic_api_key: Option<Sourced<String>>,
    pub api_base: Option<Sourced<String>>,
    pub dynamic_api_base: Option<Sourced<String>>,
}

impl OcrCredentialInputs {
    pub fn new(
        api_key: Option<String>,
        api_key_source: InputSource,
        api_base: Option<String>,
        api_base_source: InputSource,
    ) -> Self {
        Self {
            api_key: nonblank(api_key).map(|value| Sourced::new(value, api_key_source)),
            dynamic_api_key: None,
            api_base: nonblank(api_base).map(|value| Sourced::new(value, api_base_source)),
            dynamic_api_base: None,
        }
    }
}

#[derive(Clone)]
pub struct OcrTransportConfig {
    pub extra_headers: Vec<(String, String)>,
    pub extra_headers_source: InputSource,
    pub timeout: Duration,
    pub max_download_bytes: u64,
    pub max_response_bytes: usize,
    pub poll_timeout: Duration,
}

impl Default for OcrTransportConfig {
    fn default() -> Self {
        Self {
            extra_headers: Vec::new(),
            extra_headers_source: InputSource::Deployment,
            timeout: Duration::from_secs(OCR_HTTP_TIMEOUT_SECS),
            max_download_bytes: crate::constants::OCR_DOWNLOAD_MAX_BYTES,
            max_response_bytes: crate::constants::OCR_RESPONSE_MAX_BYTES,
            poll_timeout: Duration::from_secs(crate::constants::OCR_POLL_TIMEOUT_SECS),
        }
    }
}

impl OcrTransportConfig {
    pub fn with_overrides(
        self,
        extra_headers: Vec<(String, String)>,
        extra_headers_source: InputSource,
        timeout: Option<Duration>,
    ) -> Self {
        Self {
            extra_headers,
            extra_headers_source,
            timeout: timeout.unwrap_or(self.timeout),
            ..self
        }
    }
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Caller-supplied connection overrides for a [`LiteLLMOcrRequest`], in the
/// shape hosts receive them: JSON-ish headers, optional timeout, optional
/// credentials, and per-field provenance in `input_sources`.
#[derive(Clone, Debug, Default)]
pub struct OcrConnectionInputs {
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub input_sources: BTreeMap<String, InputSource>,
}

impl OcrConnectionInputs {
    fn source(&self, name: &str) -> InputSource {
        self.input_sources.get(name).copied().unwrap_or_default()
    }

    fn header_pairs(&self) -> Result<Vec<(String, String)>, super::Error> {
        self.extra_headers
            .iter()
            .map(|(name, value)| {
                value
                    .as_str()
                    .map(|value| (name.clone(), value.to_string()))
                    .ok_or_else(|| super::Error::RequestField {
                        path: format!("extra_headers.{name}"),
                    })
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct OcrConnection {
    pub api_key: Option<String>,
    pub api_key_source: InputSource,
    pub api_base: Option<String>,
    pub api_base_source: InputSource,
    pub extra_headers: Vec<(String, String)>,
    pub extra_headers_source: InputSource,
    pub timeout: Duration,
    pub max_download_bytes: u64,
    pub max_response_bytes: usize,
    pub poll_timeout: Duration,
}

impl OcrConnection {
    pub(crate) fn new(credentials: ResolvedOcrCredentials, transport: OcrTransportConfig) -> Self {
        let api_key_source = credentials
            .api_key
            .as_ref()
            .map(Sourced::source)
            .unwrap_or(InputSource::Deployment);
        let api_base_source = credentials
            .api_base
            .as_ref()
            .map(Sourced::source)
            .unwrap_or(InputSource::Deployment);
        Self {
            api_key: credentials.api_key.map(Sourced::into_value),
            api_key_source,
            api_base: credentials.api_base.map(Sourced::into_value),
            api_base_source,
            extra_headers: transport.extra_headers,
            extra_headers_source: transport.extra_headers_source,
            timeout: transport.timeout,
            max_download_bytes: transport.max_download_bytes,
            max_response_bytes: transport.max_response_bytes,
            poll_timeout: transport.poll_timeout,
        }
    }
}

impl Default for OcrConnection {
    fn default() -> Self {
        Self::new(
            ResolvedOcrCredentials::default(),
            OcrTransportConfig::default(),
        )
    }
}

#[derive(Clone, Default)]
pub(crate) struct ResolvedOcrCredentials {
    pub api_key: Option<Sourced<String>>,
    pub api_base: Option<Sourced<String>>,
}

pub struct LiteLLMOcrRequest<D = OcrDocumentInput> {
    pub model: String,
    pub document: D,
    pub credentials: OcrCredentialInputs,
    pub transport: OcrTransportConfig,
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
        document: impl Into<OcrDocumentInput>,
        custom_llm_provider: Option<&str>,
        optional_params: CallArguments,
    ) -> Result<Self, super::Error> {
        let (model, config) = resolve_provider_config(&model, custom_llm_provider)?;
        let default_transport = OcrTransportConfig::default();
        let max_response_bytes = optional_params
            .get("max_response_bytes")
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .filter(|value| *value > 0 && *value <= default_transport.max_response_bytes)
                    .ok_or_else(|| super::Error::RequestField {
                        path: "max_response_bytes".into(),
                    })
            })
            .transpose()?
            .unwrap_or(default_transport.max_response_bytes);
        let transport = OcrTransportConfig {
            max_response_bytes,
            ..default_transport
        };
        let optional_params = optional_params
            .into_iter()
            .filter(|(name, _)| name != "max_response_bytes")
            .collect();

        Ok(Self {
            model,
            document: document.into(),
            credentials: OcrCredentialInputs::default(),
            transport,
            hooks: Arc::new(NoopOcrHooks),
            litellm_call_id: None,
            optional_params,
            input_sources: BTreeMap::new(),
            azure_ad_token_provider: None,
            config,
        })
    }
}

impl<D> LiteLLMOcrRequest<D> {
    pub fn map_document<T, E>(
        self,
        map: impl FnOnce(D) -> Result<T, E>,
    ) -> Result<LiteLLMOcrRequest<T>, E> {
        Ok(LiteLLMOcrRequest {
            model: self.model,
            document: map(self.document)?,
            credentials: self.credentials,
            transport: self.transport,
            hooks: self.hooks,
            litellm_call_id: self.litellm_call_id,
            optional_params: self.optional_params,
            input_sources: self.input_sources,
            azure_ad_token_provider: self.azure_ad_token_provider,
            config: self.config,
        })
    }

    pub fn with_document<T>(self, document: T) -> LiteLLMOcrRequest<T> {
        LiteLLMOcrRequest {
            model: self.model,
            document,
            credentials: self.credentials,
            transport: self.transport,
            hooks: self.hooks,
            litellm_call_id: self.litellm_call_id,
            optional_params: self.optional_params,
            input_sources: self.input_sources,
            azure_ad_token_provider: self.azure_ad_token_provider,
            config: self.config,
        }
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

    pub fn with_connection_inputs(
        self,
        credentials: OcrCredentialInputs,
        transport: OcrTransportConfig,
        input_sources: BTreeMap<String, InputSource>,
    ) -> Self {
        Self {
            credentials,
            transport,
            input_sources,
            ..self
        }
    }
}

impl LiteLLMOcrRequest {
    /// Builds a request from host-shaped inputs in one step: provider
    /// resolution, optional-param validation, header/timeout overrides and
    /// sourced credentials. Hosts should prefer this over sequencing
    /// [`Self::new`], [`OcrTransportConfig::with_overrides`] and
    /// [`Self::with_connection_inputs`] by hand.
    pub fn from_inputs(
        model: String,
        document: impl Into<OcrDocumentInput>,
        custom_llm_provider: Option<&str>,
        optional_params: CallArguments,
        connection: OcrConnectionInputs,
    ) -> Result<Self, super::Error> {
        let request = Self::new(model, document, custom_llm_provider, optional_params)?;
        let transport = request.transport.clone().with_overrides(
            connection.header_pairs()?,
            connection.source("extra_headers"),
            connection.timeout,
        );
        let (api_key_source, api_base_source) =
            (connection.source("api_key"), connection.source("api_base"));
        let credentials = OcrCredentialInputs::new(
            connection.api_key,
            api_key_source,
            connection.api_base,
            api_base_source,
        );
        Ok(request.with_connection_inputs(credentials, transport, connection.input_sources))
    }
}

pub(crate) type ResolvedOcrRequest = LiteLLMOcrRequest<OcrDocument>;

pub(crate) struct PreparedOcrRequest {
    pub model: String,
    pub document: OcrDocument,
    pub connection: OcrConnection,
    pub hooks: Arc<dyn OcrHooks>,
    pub optional_params: CallArguments,
    pub input_sources: BTreeMap<String, InputSource>,
    pub azure_ad_token_provider: Option<TokenProviderHandle>,
    pub(crate) config: OcrConfigKind,
}

impl PreparedOcrRequest {
    pub(crate) fn new(request: ResolvedOcrRequest, connection: OcrConnection) -> Self {
        let LiteLLMOcrRequest {
            model,
            document,
            credentials: _,
            transport: _,
            hooks,
            litellm_call_id: _,
            optional_params,
            input_sources,
            azure_ad_token_provider,
            config,
        } = request;
        Self {
            model,
            document,
            connection,
            hooks,
            optional_params,
            input_sources,
            azure_ad_token_provider,
            config,
        }
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

    pub(crate) fn provider_name(&self) -> &'static str {
        self.config.provider().into()
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
    use serde_json::json;

    use super::*;

    fn document() -> OcrDocument {
        OcrDocument::try_from(
            json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"}),
        )
        .unwrap()
    }

    #[test]
    fn from_inputs_applies_connection_overrides_with_field_sources() {
        let request = LiteLLMOcrRequest::from_inputs(
            "mistral/model".into(),
            document(),
            None,
            Default::default(),
            OcrConnectionInputs {
                api_key: Some(" key ".into()),
                api_base: Some("".into()),
                extra_headers: json!({"x-a": "1"}).as_object().unwrap().clone(),
                timeout: Some(Duration::from_secs(7)),
                input_sources: [
                    ("api_key".to_string(), InputSource::Request),
                    ("extra_headers".to_string(), InputSource::Request),
                ]
                .into(),
            },
        )
        .unwrap();

        let api_key = request.credentials.api_key.as_ref().unwrap();
        assert_eq!(api_key.clone().into_value(), "key");
        assert_eq!(api_key.source(), InputSource::Request);
        assert!(request.credentials.api_base.is_none());
        assert_eq!(
            request.transport.extra_headers,
            vec![("x-a".to_string(), "1".to_string())]
        );
        assert_eq!(request.transport.extra_headers_source, InputSource::Request);
        assert_eq!(request.transport.timeout, Duration::from_secs(7));
        assert_eq!(request.input_sources.len(), 2);

        let defaulted = LiteLLMOcrRequest::from_inputs(
            "mistral/model".into(),
            document(),
            None,
            Default::default(),
            OcrConnectionInputs::default(),
        )
        .unwrap();
        assert_eq!(
            defaulted.transport.timeout,
            OcrTransportConfig::default().timeout
        );
        assert_eq!(
            defaulted.transport.extra_headers_source,
            InputSource::Deployment
        );
    }

    #[test]
    fn from_inputs_rejects_non_string_header_values_by_path() {
        let Err(error) = LiteLLMOcrRequest::from_inputs(
            "mistral/model".into(),
            document(),
            None,
            Default::default(),
            OcrConnectionInputs {
                extra_headers: json!({"x-a": 1}).as_object().unwrap().clone(),
                ..Default::default()
            },
        ) else {
            panic!("non-string header value accepted");
        };
        assert!(matches!(
            error,
            super::super::Error::RequestField { ref path } if path == "extra_headers.x-a"
        ));
    }

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
