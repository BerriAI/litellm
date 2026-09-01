use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;
use url::Url;

use super::canonical::{CanonicalOcrRequest, DocumentKind};
use super::plan::{CompletionPlan, DocumentPlan};
use super::policy::OcrParameterPolicy;
use super::response::NormalizedOcr;
use super::types::OcrDialectId;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CompileError {
    #[error("invalid OCR parameter {field}: {reason}")]
    InvalidParameter {
        field: &'static str,
        reason: &'static str,
    },
    #[error("provider extra collides with canonical OCR field: {0}")]
    ExtraCollidesWithCanonical(String),
    #[error("LiteLLM control cannot enter OCR provider extras: {0}")]
    ReservedExtra(String),
    #[error("OCR dialect is not compiled yet: {0:?}")]
    UnsupportedDialect(OcrDialectId),
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum NormalizeError {
    #[error("invalid terminal OCR response: {0}")]
    InvalidPayload(&'static str),
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct OcrCredentials {
    api_key: Option<String>,
    oauth_token: Option<String>,
}

impl OcrCredentials {
    pub fn new(api_key: Option<String>, oauth_token: Option<String>) -> Self {
        Self {
            api_key,
            oauth_token,
        }
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn oauth_token(&self) -> Option<&str> {
        self.oauth_token.as_deref()
    }
}

/// ```compile_fail
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<litellm_core::ocr::compiler::OcrCredentials>();
/// assert_serialize::<litellm_core::ocr::compiler::ResolvedOcrTarget>();
/// ```
#[derive(Clone, PartialEq)]
pub struct ResolvedOcrTarget {
    dialect: OcrDialectId,
    api_base: Url,
    credentials: OcrCredentials,
}

impl ResolvedOcrTarget {
    pub fn new(dialect: OcrDialectId, api_base: Url, credentials: OcrCredentials) -> Self {
        Self {
            dialect,
            api_base,
            credentials,
        }
    }

    pub fn dialect(&self) -> OcrDialectId {
        self.dialect
    }

    pub fn api_base(&self) -> &Url {
        &self.api_base
    }

    pub fn credentials(&self) -> &OcrCredentials {
        &self.credentials
    }
}

#[derive(Clone, PartialEq)]
pub enum ProviderDocument {
    RemoteUrl {
        kind: DocumentKind,
        url: Url,
    },
    InlineDataUri {
        kind: DocumentKind,
        data_uri: String,
    },
    Reference {
        id: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    Get,
    Post,
}

#[derive(Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct OcrWireBody(pub(crate) Value);

impl OcrWireBody {
    pub fn as_value(&self) -> &Value {
        &self.0
    }
}

#[derive(Clone, PartialEq, Serialize)]
pub struct CompiledHttpRequest {
    pub method: HttpMethod,
    pub url: Url,
    pub headers: BTreeMap<String, String>,
    pub body: OcrWireBody,
}

#[derive(Clone, PartialEq)]
pub struct ProviderPayload(Value);

impl ProviderPayload {
    pub fn new(value: Value) -> Self {
        Self(value)
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrDocumentPolicy {
    Ready,
    FetchRemoteUrlAndInline,
    UploadUnlessProviderReference,
}

pub trait OcrDialectCompiler: Send + Sync {
    fn parameter_policy(&self) -> &'static OcrParameterPolicy;

    fn prepare_document(
        &self,
        document: &super::canonical::OcrDocument,
        target: &ResolvedOcrTarget,
    ) -> Result<DocumentPlan, CompileError>;

    fn compile_submit(
        &self,
        request: &CanonicalOcrRequest,
        document: ProviderDocument,
        target: &ResolvedOcrTarget,
    ) -> Result<CompiledHttpRequest, CompileError>;

    fn completion_plan(&self) -> CompletionPlan;

    fn normalize(
        &self,
        terminal_response: ProviderPayload,
    ) -> Result<NormalizedOcr, NormalizeError>;
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn compiled_http_request_is_the_serializable_wire_boundary() {
        let request = CompiledHttpRequest {
            method: HttpMethod::Post,
            url: Url::parse("https://example.com/ocr").expect("valid URL"),
            headers: BTreeMap::from([("content-type".to_string(), "application/json".to_string())]),
            body: OcrWireBody(json!({"model": "ocr-model"})),
        };

        let serialized = serde_json::to_value(request).expect("wire request serializes");

        assert_eq!(serialized["method"], "POST");
        assert_eq!(serialized["url"], "https://example.com/ocr");
        assert_eq!(serialized["body"]["model"], "ocr-model");
    }
}
