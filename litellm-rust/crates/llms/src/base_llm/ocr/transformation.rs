use std::{collections::BTreeMap, future::Future, sync::Arc, time::Duration};

use litellm_auth::{InputSource, SecretValue, Sourced, TokenProviderHandle};
use litellm_core_utils::{
    call_arguments::CallArguments,
    serde_compat::{FiniteF64, LaxI64},
    settings::ProcessEnvironment,
};
use litellm_http::outbound::{OutboundRequest, RequestSigner};
use litellm_secrets::source::Secrets;
use serde::{
    Deserialize, Serialize,
    de::{DeserializeOwned, IntoDeserializer},
};
use serde_json::{Map, Value};
use serde_with::serde_as;

use crate::base_llm::ocr::{
    error::Error,
    handler::{CallHooks, OcrClient, read_response_bytes, transform_request_body},
    settings::OcrSettings,
};

pub const OCR_RESPONSE_MAX_BYTES: usize = 64 * 1024 * 1024;
pub const OCR_INLINE_MAX_BYTES: usize = 50 * 1024 * 1024;
pub const OCR_MAX_FETCH_REDIRECTS: usize = 10;
pub const OCR_POLL_RETRY_SECS: u64 = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

impl TryFrom<Value> for OcrDocument {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        decode_request_value(value, "document")
    }
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
    pub api_key: Option<Sourced<SecretValue>>,
    pub dynamic_api_key: Option<Sourced<SecretValue>>,
    pub api_base: Option<Sourced<String>>,
    pub dynamic_api_base: Option<Sourced<String>>,
}

impl OcrCredentialInputs {
    pub fn new(
        api_key: Option<SecretValue>,
        api_key_source: InputSource,
        api_base: Option<String>,
        api_base_source: InputSource,
    ) -> Self {
        Self {
            api_key: nonblank(api_key.as_ref().map(|key| key.expose().to_string()))
                .map(|value| Sourced::new(SecretValue::new(value), api_key_source)),
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
    pub timeout: Option<Duration>,
    pub max_response_bytes: usize,
}

impl Default for OcrTransportConfig {
    fn default() -> Self {
        Self {
            extra_headers: Vec::new(),
            extra_headers_source: InputSource::Deployment,
            timeout: None,
            max_response_bytes: OCR_RESPONSE_MAX_BYTES,
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
            timeout: timeout.or(self.timeout),
            ..self
        }
    }
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[derive(Clone)]
pub struct OcrConnection {
    pub api_key: Option<SecretValue>,
    pub api_key_source: InputSource,
    pub api_base: Option<String>,
    pub api_base_source: InputSource,
    pub extra_headers: Vec<(String, String)>,
    pub extra_headers_source: InputSource,
    pub timeout: Duration,
    pub max_response_bytes: usize,
    pub settings: OcrSettings,
    pub secrets: Secrets,
}

impl OcrConnection {
    pub fn new(
        credentials: ResolvedOcrCredentials,
        transport: OcrTransportConfig,
        settings: OcrSettings,
        secrets: Secrets,
    ) -> Self {
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
            timeout: transport
                .timeout
                .filter(|timeout| !timeout.is_zero())
                .unwrap_or(settings.request_timeout),
            max_response_bytes: transport.max_response_bytes,
            settings,
            secrets,
        }
    }

    pub fn secret(&self, name: &str) -> Option<String> {
        self.secrets.get(name)
    }
}

impl Default for OcrConnection {
    fn default() -> Self {
        Self::new(
            ResolvedOcrCredentials::default(),
            OcrTransportConfig::default(),
            OcrSettings::default(),
            Arc::new(ProcessEnvironment),
        )
    }
}

#[derive(Clone, Default)]
pub struct ResolvedOcrCredentials {
    pub api_key: Option<Sourced<SecretValue>>,
    pub api_base: Option<Sourced<String>>,
}

pub struct PreparedOcrRequest {
    pub model: String,
    pub document: OcrDocument,
    pub connection: OcrConnection,
    /// Whether the caller handed over the document as is, so the wire body's document
    /// is the caller's own input rather than something the route prepared.
    pub caller_document: bool,
    pub optional_params: CallArguments,
    pub input_sources: BTreeMap<String, InputSource>,
    pub azure_ad_token_provider: Option<TokenProviderHandle>,
}

impl PreparedOcrRequest {
    pub fn response_format(&self) -> Result<OcrResponseFormat, Error> {
        response_format(&self.optional_params)
    }
}

pub fn response_format(optional_params: &CallArguments) -> Result<OcrResponseFormat, Error> {
    optional_params
        .get("req_format")
        .filter(|value| !value.is_null())
        .map(|value| serde_json::from_value(value.clone()).map_err(|_| Error::RequestFormat))
        .transpose()
        .map(|format| format.unwrap_or_default())
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

#[derive(Debug)]
pub struct DecodedOcrResponse<T> {
    pub data: T,
    pub native: Option<Map<String, Value>>,
    pub text: String,
}

pub fn decode_request_value<T: DeserializeOwned>(value: Value, prefix: &str) -> Result<T, Error> {
    serde_path_to_error::deserialize(value.into_deserializer()).map_err(|error| {
        Error::RequestField {
            path: format!("{prefix}.{}", error.path()),
        }
    })
}

pub fn decode_response_value<T: DeserializeOwned>(value: Value, prefix: &str) -> Result<T, Error> {
    serde_path_to_error::deserialize(value.into_deserializer()).map_err(|error| {
        Error::ResponseField {
            path: format!("{prefix}.{}", error.path()),
        }
    })
}

pub fn decode_response<T: DeserializeOwned>(
    bytes: &[u8],
    native: bool,
) -> Result<DecodedOcrResponse<T>, Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let data = serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
        Error::ResponseField {
            path: error.path().to_string(),
        }
    })?;
    deserializer.end().map_err(|_| Error::ResponseField {
        path: "response".into(),
    })?;
    let native = if native {
        Some(
            serde_json::from_slice(bytes).map_err(|_| Error::ResponseField {
                path: "response".into(),
            })?,
        )
    } else {
        None
    };
    Ok(DecodedOcrResponse {
        data,
        native,
        text: String::from_utf8_lossy(bytes).into_owned(),
    })
}

const HEALTH_CHECK_PDF_DATA_URI: &str = "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MKMyAwIG9iago8PC9UeXBlIC9QYWdlCi9QYXJlbnQgMSAwIFIKL01lZGlhQm94IFswIDAgNjEyIDc5Ml0KL0NvbnRlbnRzIDQgMCBSCi9SZXNvdXJjZXMgPDwvRm9udCA8PC9GMSAyIDAgUj4+Pj4+PgplbmRvYmoKNCAwIG9iago8PC9MZW5ndGggNDQ+PgpzdHJlYW0KQlQKL0YxIDI0IFRmCjEwMCA3MDAgVGQKKHRlc3QpIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKMiAwIG9iago8PC9UeXBlIC9Gb250Ci9TdWJ0eXBlIC9UeXBlMQovQmFzZUZvbnQgL0hlbHZldGljYT4+CmVuZG9iagoxIDAgb2JqCjw8L1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDE+PgplbmRvYmoKNSAwIG9iago8PC9UeXBlIC9DYXRhbG9nCi9QYWdlcyAxIDAgUj4+CmVuZG9iagp0cmFpbGVyCjw8L1NpemUgNgovUm9vdCA1IDAgUj4+CnN0YXJ0eHJlZgozMjQKJSVFT0Y=";

/// Output of `validate_environment`: whatever a provider resolves up front
/// (headers at minimum; Vertex also carries the project id).
pub trait OcrEnvironment: Send + Sync {
    fn headers(&self) -> &[(String, String)];

    fn signer(&self) -> Option<&dyn RequestSigner> {
        None
    }
}

impl OcrEnvironment for Vec<(String, String)> {
    fn headers(&self) -> &[(String, String)] {
        self
    }
}

#[derive(Clone, Copy)]
pub struct OcrRequestContext<'a> {
    pub client: &'a OcrClient,
    pub connection: &'a OcrConnection,
}

#[derive(Clone, Copy)]
pub struct OcrResponseContext<'a> {
    pub client: &'a OcrClient,
    pub connection: &'a OcrConnection,
    pub hooks: &'a dyn CallHooks<Error>,
    pub request_format: OcrResponseFormat,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
}

pub trait BaseOcrConfig: Send + Sync + Sized + 'static {
    type OcrParams: Send + Sync;
    type ProviderRequest: Serialize + Send;
    type Environment: OcrEnvironment;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &[]
    }

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        None
    }

    fn secret_names(&self) -> Vec<&'static str>;

    fn resolve_connection_params(&self, inputs: OcrCredentialInputs) -> ResolvedOcrCredentials {
        ResolvedOcrCredentials {
            api_key: inputs
                .dynamic_api_key
                .filter(|value| !value.value().expose().is_empty())
                .or(inputs.api_key),
            api_base: inputs
                .dynamic_api_base
                .filter(|value| !value.value().is_empty())
                .or(inputs.api_base),
        }
    }

    fn get_health_check_document(&self) -> OcrDocument {
        OcrDocument::DocumentUrl {
            document_url: HEALTH_CHECK_PDF_DATA_URI.into(),
            extra_fields: Default::default(),
        }
    }

    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        model: &str,
    ) -> Result<Self::OcrParams, Error>;

    fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> impl Future<Output = Result<Self::Environment, Error>> + Send;

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        optional_params: &Self::OcrParams,
        environment: &Self::Environment,
    ) -> Result<String, Error>;

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        headers: &[(String, String)],
    ) -> Result<Self::ProviderRequest, Error>;

    fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        headers: &[(String, String)],
        _context: OcrRequestContext<'_>,
    ) -> impl Future<Output = Result<Self::ProviderRequest, Error>> + Send {
        async move { self.transform_ocr_request(model, document, optional_params, headers) }
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, Error>;

    fn async_transform_ocr_response(
        &self,
        model: &str,
        raw_response: reqwest::Response,
        context: OcrResponseContext<'_>,
    ) -> impl Future<Output = Result<LiteLLMOcrResponse, Error>> + Send {
        async move {
            let bytes =
                read_response_bytes(raw_response, context.connection.max_response_bytes).await?;
            context.hooks.response_received(&bytes).await?;
            self.transform_ocr_response(model, &bytes, context.request_format)
        }
    }

    fn get_error_class(
        &self,
        error_message: String,
        status_code: u16,
        headers: Vec<(String, String)>,
    ) -> Error {
        Error::Provider {
            status: status_code,
            body: error_message,
            headers,
        }
    }

    /// Provider-specific check applied to the composed body, both before and
    /// after guardrail hooks. Defaults to accepting any body.
    fn validate_request_body(&self, _body: &Value) -> Result<(), Error> {
        Ok(())
    }

    /// Rust counterpart of `BaseLLMHTTPHandler._async_prepare_ocr_request`:
    /// map params, validate environment, build URL, transform, compose body.
    fn prepare_request(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
        hooks: &dyn CallHooks<Error>,
    ) -> impl Future<Output = Result<OutboundRequest, Error>> + Send {
        async move {
            let params = self.map_ocr_params(&request.optional_params, &request.model)?;
            let environment = self.validate_environment(request, client).await?;
            let url = self.get_complete_url(request, &params, &environment)?;
            let headers = environment.headers();
            let body = self
                .async_transform_ocr_request(
                    &request.model,
                    request.document.clone(),
                    &params,
                    headers,
                    OcrRequestContext {
                        client,
                        connection: &request.connection,
                    },
                )
                .await?;
            transform_request_body(
                self,
                request,
                &url,
                headers,
                body,
                environment.signer(),
                hooks,
            )
            .await
        }
    }
}

pub fn decode_and_normalize_response<T: DeserializeOwned>(
    model: &str,
    raw_response: &[u8],
    request_format: OcrResponseFormat,
    normalize: impl FnOnce(&str, T) -> Result<LiteLLMOcrResponse, Error>,
) -> Result<LiteLLMOcrResponse, Error> {
    let decoded = decode_response(raw_response, request_format == OcrResponseFormat::Native)?;
    Ok(LiteLLMOcrResponse {
        provider_native_response: decoded.native,
        ..normalize(model, decoded.data)?
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn connection_timeout_falls_back_to_the_request_timeout_setting_like_a_python_or() {
        let settings = OcrSettings {
            request_timeout: Duration::from_secs(42),
            ..OcrSettings::default()
        };
        let timeout = |call: Option<Duration>| {
            OcrConnection::new(
                ResolvedOcrCredentials::default(),
                OcrTransportConfig {
                    timeout: call,
                    ..OcrTransportConfig::default()
                },
                settings.clone(),
                Arc::new(ProcessEnvironment),
            )
            .timeout
        };
        assert_eq!(timeout(None), Duration::from_secs(42));
        assert_eq!(timeout(Some(Duration::ZERO)), Duration::from_secs(42));
        assert_eq!(
            timeout(Some(Duration::from_secs(5))),
            Duration::from_secs(5)
        );
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

    #[rstest::rstest]
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
        assert_eq!(
            serde_json::to_value(document.with_source(replacement.clone())).unwrap(),
            json!({"type": kind, kind: replacement, field: extra})
        );
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
