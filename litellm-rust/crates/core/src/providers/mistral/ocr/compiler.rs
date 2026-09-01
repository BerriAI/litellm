use crate::ocr::compiler::OcrDocumentPolicy;
use crate::ocr::policy::{MISTRAL_OCR_PARAMETER_POLICY, OcrParameterPolicy};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MistralOcrCodec;

impl MistralOcrCodec {
    pub fn parameter_policy(&self) -> &'static OcrParameterPolicy {
        &MISTRAL_OCR_PARAMETER_POLICY
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MistralOcrCompiler {
    codec: MistralOcrCodec,
}

impl MistralOcrCompiler {
    pub fn codec(&self) -> &MistralOcrCodec {
        &self.codec
    }

    pub fn document_policy(&self) -> OcrDocumentPolicy {
        OcrDocumentPolicy::Ready
    }
}

#[cfg(test)]
mod tests {
    use crate::ocr::compiler::OcrDocumentPolicy;
    use crate::ocr::policy::OcrCanonicalField;
    use crate::providers::azure_ai::ocr::foundry::AzureFoundryMistralOcrCompiler;
    use crate::providers::vertex_ai::ocr::mistral::VertexMistralOcrCompiler;

    use super::*;

    #[test]
    fn mistral_compatible_dialects_share_the_codec_policy() {
        let mistral = MistralOcrCompiler::default();
        let foundry = AzureFoundryMistralOcrCompiler::default();
        let vertex = VertexMistralOcrCompiler::default();

        for field in OcrCanonicalField::ALL {
            assert_eq!(
                mistral.codec().parameter_policy().disposition(field),
                foundry.codec().parameter_policy().disposition(field)
            );
            assert_eq!(
                mistral.codec().parameter_policy().disposition(field),
                vertex.codec().parameter_policy().disposition(field)
            );
        }
        assert_eq!(mistral.document_policy(), OcrDocumentPolicy::Ready);
        assert_eq!(
            foundry.document_policy(),
            OcrDocumentPolicy::FetchRemoteUrlAndInline
        );
        assert_eq!(
            vertex.document_policy(),
            OcrDocumentPolicy::FetchRemoteUrlAndInline
        );
    }
}
