use crate::ocr::policy::{OcrParameterPolicy, REJECT_CANONICAL_OCR_PARAMETER_POLICY};

use super::response::ReductoOcrResponseNormalizer;
use super::upload::ReductoDocumentAdapter;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReductoLegacyOcrCompiler {
    document_adapter: ReductoDocumentAdapter,
    response_normalizer: ReductoOcrResponseNormalizer,
}

impl ReductoLegacyOcrCompiler {
    pub fn parameter_policy(&self) -> &'static OcrParameterPolicy {
        &REJECT_CANONICAL_OCR_PARAMETER_POLICY
    }

    pub fn document_adapter(&self) -> &ReductoDocumentAdapter {
        &self.document_adapter
    }

    pub fn response_normalizer(&self) -> &ReductoOcrResponseNormalizer {
        &self.response_normalizer
    }
}
