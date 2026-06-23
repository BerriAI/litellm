//! Typed OCR domain model.
//!
//! These types mirror the Python OCR surface (`litellm/ocr/main.py` and the
//! `OCRResponse` pydantic models) one-to-one, but as real Rust types rather than
//! free-form JSON. Serde derives produce the exact wire JSON, so no provider
//! ever hand-builds or hand-parses JSON.
//!
//! `serde_json::Value` appears in exactly three leaves, and only where Python's
//! own types are `Any` / `Dict[str, Any]`: [`AnnotationFormat`] (a user-supplied
//! JSON Schema we forward verbatim), [`OcrResponse::document_annotation`]
//! (`Optional[Any]`), and [`OcrPageImage::bbox`] (`Optional[Dict[str, Any]]`).
//! Everywhere Python is concrete, this module is concrete.
//!
//! Unknown upstream fields are dropped: serde ignores unrecognized keys by
//! default, which is the deliberate "fully strict" behavior (vs Python's
//! `model_config = {"extra": "allow"}` passthrough).

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Which OCR provider to dispatch to.
///
/// The Python shell resolves `(custom_llm_provider, model)` into a precise
/// variant before crossing the bridge — e.g. an `azure_ai` model that is a
/// Document Intelligence model becomes [`OcrProvider::AzureDocumentIntelligence`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrProvider {
    Mistral,
    AzureAi,
    AzureDocumentIntelligence,
    VertexAi,
    VertexDeepseek,
    Reducto,
}

/// The document to OCR.
///
/// `type: "file"` inputs are converted to a base64 data-URI in the Python shell,
/// so the core only ever sees a URL or image URL. Serializes to the provider
/// wire shape, e.g. `{"type": "document_url", "document_url": "…"}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OcrDocument {
    DocumentUrl { document_url: String },
    ImageUrl { image_url: String },
}

/// Extra HTTP headers (Python `extra_headers`). Header values are strings on the
/// wire; `BTreeMap` keeps a stable, testable ordering.
pub type OcrHeaders = BTreeMap<String, String>;

/// Table output format (Mistral `table_format`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableFormat {
    Markdown,
    Html,
}

/// Confidence-score granularity (Mistral `confidence_scores_granularity`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceGranularity {
    Word,
    Page,
}

/// A provider "response format" (a user-supplied JSON Schema, forwarded verbatim
/// to the provider). A JSON Schema is arbitrary JSON by definition, so
/// `serde_json::Value` is its precise type — mirrors Python's `Dict[str, Any]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnnotationFormat(pub Value);

/// OCR options — the superset of provider params (Python `**kwargs`). Every field
/// is optional; serde skips `None`, so each provider's request body carries only
/// the keys that were actually set.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_image_base64: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_min_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox_annotation_format: Option<AnnotationFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_annotation_format: Option<AnnotationFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_annotation_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extract_header: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extract_footer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table_format: Option<TableFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_scores_granularity: Option<ConfidenceGranularity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// A fully-typed OCR request — maps argument-for-argument to `litellm.ocr()` /
/// `litellm.aocr()`, grouped into a struct (idiomatic Rust; also keeps the entry
/// point under clippy's `too_many_arguments` ceiling).
#[derive(Clone, Debug)]
pub struct OcrRequest {
    pub provider: OcrProvider,
    pub model: String,
    pub document: OcrDocument,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: OcrHeaders,
    pub timeout: Option<Duration>,
    pub params: OcrParams,
}

/// Page dimensions (Python `OCRPageDimensions`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPageDimensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dpi: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
}

/// An image extracted from a page (Python `OCRPageImage`). `bbox` is
/// `Optional[Dict[str, Any]]` in Python, so it stays a `Value`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPageImage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Value>,
}

/// A single OCR page (Python `OCRPage`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrPage {
    pub index: u32,
    pub markdown: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<OcrPageImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<OcrPageDimensions>,
}

/// Usage information (Python `OCRUsageInfo`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrUsageInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages_processed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credits: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_size_bytes: Option<u64>,
}

fn default_ocr_object() -> String {
    "ocr".to_string()
}

/// The standardized OCR response (Python `OCRResponse`). `document_annotation` is
/// `Optional[Any]` in Python (it depends on the user's annotation schema), so it
/// stays a `Value`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrResponse {
    #[serde(default)]
    pub pages: Vec<OcrPage>,
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_annotation: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_info: Option<OcrUsageInfo>,
    #[serde(default = "default_ocr_object")]
    pub object: String,
}
