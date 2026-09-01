use crate::ocr::compiler::OcrDocumentPolicy;
use crate::providers::mistral::ocr::compiler::MistralOcrCodec;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AzureFoundryMistralOcrCompiler {
    codec: MistralOcrCodec,
}

impl AzureFoundryMistralOcrCompiler {
    pub fn codec(&self) -> &MistralOcrCodec {
        &self.codec
    }

    pub fn document_policy(&self) -> OcrDocumentPolicy {
        OcrDocumentPolicy::FetchRemoteUrlAndInline
    }
}
