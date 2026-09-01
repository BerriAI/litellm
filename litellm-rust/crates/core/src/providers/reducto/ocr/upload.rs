use crate::ocr::compiler::OcrDocumentPolicy;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReductoDocumentAdapter;

impl ReductoDocumentAdapter {
    pub fn document_policy(&self) -> OcrDocumentPolicy {
        OcrDocumentPolicy::UploadUnlessProviderReference
    }
}
