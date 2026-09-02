use std::collections::BTreeMap;

use serde_json::Value;
use thiserror::Error;
use url::Url;

use super::canonical::{CanonicalOcrRequest, DocumentKind};
use super::plan::{CompletionPlan, DocumentPlan};
use super::policy::OcrParameterPolicy;
use super::response::NormalizedOcr;
use super::types::OcrDialectId;
pub use super::wire::{MultipartBodyPlan, MultipartPart, OcrJsonValue, OcrWireBody, OcrWireError};

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
    Inline {
        kind: DocumentKind,
        media_type: mime::Mime,
        bytes: bytes::Bytes,
    },
    Reference {
        id: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

/// ```compile_fail
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<litellm_core::ocr::compiler::CompiledHttpRequest>();
/// assert_serialize::<litellm_core::ocr::compiler::OcrWireBody>();
/// assert_serialize::<litellm_core::ocr::compiler::OcrJsonValue>();
/// ```
///
/// ```compile_fail
/// fn assert_debug<T: std::fmt::Debug>() {}
/// assert_debug::<litellm_core::ocr::compiler::OcrWireBody>();
/// ```
#[derive(Clone, PartialEq)]
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
    use bytes::Bytes;
    use mime::Mime;
    use serde_json::json;

    use super::*;
    use crate::ocr::canonical::OcrDocument;

    #[test]
    fn compilation_preserves_the_inline_media_allocation() {
        let source = Bytes::from_static(b"pdf payload");
        let source_pointer = source.as_ptr();
        let canonical_document = OcrDocument::Inline {
            kind: DocumentKind::Pdf,
            media_type: "application/pdf".parse::<Mime>().expect("valid MIME type"),
            bytes: source.clone(),
        };
        let OcrDocument::Inline {
            kind,
            media_type,
            bytes,
        } = &canonical_document
        else {
            panic!("inline canonical document expected");
        };
        let provider_document = ProviderDocument::Inline {
            kind: *kind,
            media_type: media_type.clone(),
            bytes: bytes.clone(),
        };
        let ProviderDocument::Inline {
            media_type, bytes, ..
        } = provider_document
        else {
            panic!("inline document expected");
        };
        let request = CompiledHttpRequest {
            method: HttpMethod::Post,
            url: Url::parse("https://example.com/ocr").expect("valid URL"),
            headers: BTreeMap::from([("content-type".to_string(), "application/json".to_string())]),
            body: OcrWireBody::JsonWithMedia(OcrJsonValue::Object(BTreeMap::from([
                (
                    "document".to_string(),
                    OcrJsonValue::InlineDataUri { media_type, bytes },
                ),
                ("model".to_string(), OcrJsonValue::Value(json!("ocr-model"))),
            ]))),
        };

        let OcrWireBody::JsonWithMedia(OcrJsonValue::Object(fields)) = &request.body else {
            panic!("media JSON body expected");
        };
        let OcrJsonValue::InlineDataUri { bytes, .. } = &fields["document"] else {
            panic!("inline media expected");
        };
        assert_eq!(bytes.as_ptr(), source_pointer);
    }
}
