use crate::ocr::compiler::OcrDocumentPolicy;
use crate::ocr::policy::{OcrParameterPolicy, REJECT_CANONICAL_OCR_PARAMETER_POLICY};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexDeepSeekOcrCompiler;

impl VertexDeepSeekOcrCompiler {
    pub fn parameter_policy(&self) -> &'static OcrParameterPolicy {
        &REJECT_CANONICAL_OCR_PARAMETER_POLICY
    }

    pub fn document_policy(&self) -> OcrDocumentPolicy {
        OcrDocumentPolicy::Ready
    }
}
