use std::time::Duration;

use url::Url;

use super::canonical::{DocumentKind, OcrDocument};
use super::compiler::ProviderDocument;
use super::types::OcrDialectId;

#[derive(Clone, PartialEq)]
pub enum DocumentPlan {
    Ready(ProviderDocument),
    FetchAndInline(FetchPlan),
    Upload(UploadPlan),
}

#[derive(Clone, PartialEq)]
pub struct FetchPlan {
    pub kind: DocumentKind,
    pub url: Url,
    pub max_bytes: usize,
}

#[derive(Clone, PartialEq)]
pub struct UploadPlan {
    pub dialect: OcrDialectId,
    pub document: OcrDocument,
}

#[derive(Clone, PartialEq)]
pub enum CompletionPlan {
    Immediate,
    Poll(PollPlan),
}

#[derive(Clone, PartialEq, Eq)]
pub struct PollPlan {
    pub operation_location_header: &'static str,
    pub interval: Duration,
    pub timeout: Duration,
}

/// ```compile_fail
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<litellm_core::ocr::plan::DocumentPlan>();
/// assert_serialize::<litellm_core::ocr::plan::FetchPlan>();
/// assert_serialize::<litellm_core::ocr::plan::UploadPlan>();
/// assert_serialize::<litellm_core::ocr::plan::CompletionPlan>();
/// assert_serialize::<litellm_core::ocr::plan::PollPlan>();
/// ```
const _: () = ();
