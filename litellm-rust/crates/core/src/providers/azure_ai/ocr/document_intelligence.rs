use std::time::Duration;

use crate::ocr::compiler::OcrDocumentPolicy;
use crate::ocr::plan::{CompletionPlan, PollPlan};
use crate::ocr::policy::{AZURE_DOCUMENT_INTELLIGENCE_PARAMETER_POLICY, OcrParameterPolicy};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AzureDocumentIntelligenceOcrCompiler;

impl AzureDocumentIntelligenceOcrCompiler {
    pub fn parameter_policy(&self) -> &'static OcrParameterPolicy {
        &AZURE_DOCUMENT_INTELLIGENCE_PARAMETER_POLICY
    }

    pub fn document_policy(&self) -> OcrDocumentPolicy {
        OcrDocumentPolicy::Ready
    }

    pub fn completion_plan(&self, interval: Duration, timeout: Duration) -> CompletionPlan {
        CompletionPlan::Poll(PollPlan {
            operation_location_header: "operation-location",
            interval,
            timeout,
        })
    }
}
