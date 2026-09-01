#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterDisposition {
    Forward,
    Rename(&'static str),
    Transform,
    ConsumeAsConfiguration,
    Reject,
}

macro_rules! ocr_parameter_schema {
    ($(($variant:ident, $field:ident, $wire_name:literal)),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum OcrCanonicalField {
            $($variant),+
        }

        impl OcrCanonicalField {
            pub const ALL: [Self; ocr_parameter_schema!(@count $($variant),+)] = [
                $(Self::$variant),+
            ];

            pub const fn wire_name(self) -> &'static str {
                match self {
                    $(Self::$variant => $wire_name),+
                }
            }

            pub fn from_wire_name(value: &str) -> Option<Self> {
                match value {
                    $($wire_name => Some(Self::$variant)),+,
                    _ => None,
                }
            }
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct OcrParameterPolicy {
            $(pub $field: ParameterDisposition),+
        }

        impl OcrParameterPolicy {
            pub const fn disposition(self, field: OcrCanonicalField) -> ParameterDisposition {
                match field {
                    $(OcrCanonicalField::$variant => self.$field),+
                }
            }
        }
    };
    (@count $($item:ident),+) => {
        <[()]>::len(&[$(ocr_parameter_schema!(@replace $item ())),+])
    };
    (@replace $_item:ident $sub:expr) => { $sub };
}

ocr_parameter_schema!(
    (Pages, pages, "pages"),
    (
        IncludeImageBase64,
        include_image_base64,
        "include_image_base64"
    ),
    (ImageLimit, image_limit, "image_limit"),
    (ImageMinSize, image_min_size, "image_min_size"),
    (
        BboxAnnotationFormat,
        bbox_annotation_format,
        "bbox_annotation_format"
    ),
    (
        DocumentAnnotationFormat,
        document_annotation_format,
        "document_annotation_format"
    ),
    (
        DocumentAnnotationPrompt,
        document_annotation_prompt,
        "document_annotation_prompt"
    ),
    (ExtractHeader, extract_header, "extract_header"),
    (ExtractFooter, extract_footer, "extract_footer"),
    (TableFormat, table_format, "table_format"),
    (
        ConfidenceScoresGranularity,
        confidence_scores_granularity,
        "confidence_scores_granularity"
    ),
    (IncludeBlocks, include_blocks, "include_blocks"),
    (RequestId, request_id, "id"),
);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_canonical_field_has_a_mistral_disposition() {
        let classified = OcrCanonicalField::ALL.map(|field| {
            (
                field.wire_name(),
                MISTRAL_OCR_PARAMETER_POLICY.disposition(field),
            )
        });

        assert_eq!(classified.len(), 13);
        assert!(
            classified
                .iter()
                .all(|(_, disposition)| *disposition == ParameterDisposition::Forward)
        );
    }

    #[test]
    fn every_canonical_field_has_an_explicit_non_mistral_disposition() {
        let azure = OcrCanonicalField::ALL
            .map(|field| AZURE_DOCUMENT_INTELLIGENCE_PARAMETER_POLICY.disposition(field));
        let provider_bound = OcrCanonicalField::ALL
            .map(|field| REJECT_CANONICAL_OCR_PARAMETER_POLICY.disposition(field));

        assert_eq!(azure[0], ParameterDisposition::Transform);
        assert!(
            azure[1..]
                .iter()
                .all(|disposition| *disposition == ParameterDisposition::Reject)
        );
        assert!(
            provider_bound
                .iter()
                .all(|disposition| *disposition == ParameterDisposition::Reject)
        );
    }
}
