use super::compiler::OcrDocumentPolicy;
use super::policy::{OcrParameterPolicy, ParameterDisposition};
use super::types::OcrDialectId;

pub const MISTRAL_OCR_PARAMETER_POLICY: OcrParameterPolicy = OcrParameterPolicy {
    pages: ParameterDisposition::Forward,
    include_image_base64: ParameterDisposition::Forward,
    image_limit: ParameterDisposition::Forward,
    image_min_size: ParameterDisposition::Forward,
    bbox_annotation_format: ParameterDisposition::Forward,
    document_annotation_format: ParameterDisposition::Forward,
    document_annotation_prompt: ParameterDisposition::Forward,
    extract_header: ParameterDisposition::Forward,
    extract_footer: ParameterDisposition::Forward,
    table_format: ParameterDisposition::Forward,
    confidence_scores_granularity: ParameterDisposition::Forward,
    include_blocks: ParameterDisposition::Forward,
    request_id: ParameterDisposition::Forward,
};

pub const AZURE_DOCUMENT_INTELLIGENCE_PARAMETER_POLICY: OcrParameterPolicy = OcrParameterPolicy {
    pages: ParameterDisposition::Transform,
    include_image_base64: ParameterDisposition::Reject,
    image_limit: ParameterDisposition::Reject,
    image_min_size: ParameterDisposition::Reject,
    bbox_annotation_format: ParameterDisposition::Reject,
    document_annotation_format: ParameterDisposition::Reject,
    document_annotation_prompt: ParameterDisposition::Reject,
    extract_header: ParameterDisposition::Reject,
    extract_footer: ParameterDisposition::Reject,
    table_format: ParameterDisposition::Reject,
    confidence_scores_granularity: ParameterDisposition::Reject,
    include_blocks: ParameterDisposition::Reject,
    request_id: ParameterDisposition::Reject,
};

pub const REJECT_CANONICAL_OCR_PARAMETER_POLICY: OcrParameterPolicy = OcrParameterPolicy {
    pages: ParameterDisposition::Reject,
    include_image_base64: ParameterDisposition::Reject,
    image_limit: ParameterDisposition::Reject,
    image_min_size: ParameterDisposition::Reject,
    bbox_annotation_format: ParameterDisposition::Reject,
    document_annotation_format: ParameterDisposition::Reject,
    document_annotation_prompt: ParameterDisposition::Reject,
    extract_header: ParameterDisposition::Reject,
    extract_footer: ParameterDisposition::Reject,
    table_format: ParameterDisposition::Reject,
    confidence_scores_granularity: ParameterDisposition::Reject,
    include_blocks: ParameterDisposition::Reject,
    request_id: ParameterDisposition::Reject,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OcrPollingProfile {
    pub operation_location_header: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OcrDialectProfile {
    pub dialect: OcrDialectId,
    pub parameter_policy: &'static OcrParameterPolicy,
    pub document_policy: OcrDocumentPolicy,
    pub polling: Option<OcrPollingProfile>,
}

pub const OCR_DIALECT_PROFILES: [OcrDialectProfile; 7] = [
    OcrDialectProfile {
        dialect: OcrDialectId::Mistral,
        parameter_policy: &MISTRAL_OCR_PARAMETER_POLICY,
        document_policy: OcrDocumentPolicy::Ready,
        polling: None,
    },
    OcrDialectProfile {
        dialect: OcrDialectId::AzureFoundryMistral,
        parameter_policy: &MISTRAL_OCR_PARAMETER_POLICY,
        document_policy: OcrDocumentPolicy::FetchRemoteUrlAndInline,
        polling: None,
    },
    OcrDialectProfile {
        dialect: OcrDialectId::AzureDocumentIntelligence,
        parameter_policy: &AZURE_DOCUMENT_INTELLIGENCE_PARAMETER_POLICY,
        document_policy: OcrDocumentPolicy::Ready,
        polling: Some(OcrPollingProfile {
            operation_location_header: "operation-location",
        }),
    },
    OcrDialectProfile {
        dialect: OcrDialectId::VertexMistral,
        parameter_policy: &MISTRAL_OCR_PARAMETER_POLICY,
        document_policy: OcrDocumentPolicy::FetchRemoteUrlAndInline,
        polling: None,
    },
    OcrDialectProfile {
        dialect: OcrDialectId::VertexDeepSeek,
        parameter_policy: &REJECT_CANONICAL_OCR_PARAMETER_POLICY,
        document_policy: OcrDocumentPolicy::Ready,
        polling: None,
    },
    OcrDialectProfile {
        dialect: OcrDialectId::ReductoV3,
        parameter_policy: &REJECT_CANONICAL_OCR_PARAMETER_POLICY,
        document_policy: OcrDocumentPolicy::UploadUnlessProviderReference,
        polling: None,
    },
    OcrDialectProfile {
        dialect: OcrDialectId::ReductoLegacy,
        parameter_policy: &REJECT_CANONICAL_OCR_PARAMETER_POLICY,
        document_policy: OcrDocumentPolicy::UploadUnlessProviderReference,
        polling: None,
    },
];

pub const fn ocr_dialect_profile(dialect: OcrDialectId) -> &'static OcrDialectProfile {
    match dialect {
        OcrDialectId::Mistral => &OCR_DIALECT_PROFILES[0],
        OcrDialectId::AzureFoundryMistral => &OCR_DIALECT_PROFILES[1],
        OcrDialectId::AzureDocumentIntelligence => &OCR_DIALECT_PROFILES[2],
        OcrDialectId::VertexMistral => &OCR_DIALECT_PROFILES[3],
        OcrDialectId::VertexDeepSeek => &OCR_DIALECT_PROFILES[4],
        OcrDialectId::ReductoV3 => &OCR_DIALECT_PROFILES[5],
        OcrDialectId::ReductoLegacy => &OCR_DIALECT_PROFILES[6],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::policy::{OcrCanonicalField, ParameterDisposition};

    #[test]
    fn mistral_compatible_dialects_share_parameter_rules_but_not_document_rules() {
        let mistral = ocr_dialect_profile(OcrDialectId::Mistral);
        let foundry = ocr_dialect_profile(OcrDialectId::AzureFoundryMistral);
        let vertex = ocr_dialect_profile(OcrDialectId::VertexMistral);

        for field in OcrCanonicalField::ALL {
            let expected = mistral.parameter_policy.disposition(field);
            assert_eq!(foundry.parameter_policy.disposition(field), expected);
            assert_eq!(vertex.parameter_policy.disposition(field), expected);
        }
        assert_eq!(mistral.document_policy, OcrDocumentPolicy::Ready);
        assert_eq!(
            foundry.document_policy,
            OcrDocumentPolicy::FetchRemoteUrlAndInline
        );
        assert_eq!(
            vertex.document_policy,
            OcrDocumentPolicy::FetchRemoteUrlAndInline
        );
    }

    #[test]
    fn non_mistral_profiles_preserve_provider_specific_boundaries() {
        let azure = ocr_dialect_profile(OcrDialectId::AzureDocumentIntelligence);
        assert_eq!(
            azure.parameter_policy.disposition(OcrCanonicalField::Pages),
            ParameterDisposition::Transform
        );
        assert_eq!(
            azure.polling,
            Some(OcrPollingProfile {
                operation_location_header: "operation-location"
            })
        );

        for dialect in [OcrDialectId::ReductoV3, OcrDialectId::ReductoLegacy] {
            let reducto = ocr_dialect_profile(dialect);
            assert_eq!(
                reducto.document_policy,
                OcrDocumentPolicy::UploadUnlessProviderReference
            );
            assert!(OcrCanonicalField::ALL.iter().all(|field| {
                reducto.parameter_policy.disposition(*field) == ParameterDisposition::Reject
            }));
        }
    }

    #[test]
    fn every_dialect_has_exactly_one_profile() {
        for (index, profile) in OCR_DIALECT_PROFILES.iter().enumerate() {
            assert_eq!(ocr_dialect_profile(profile.dialect), profile);
            assert!(
                OCR_DIALECT_PROFILES[index + 1..]
                    .iter()
                    .all(|other| other.dialect != profile.dialect)
            );
        }
    }
}
