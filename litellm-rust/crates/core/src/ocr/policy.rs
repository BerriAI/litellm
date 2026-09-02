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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_canonical_field_round_trips_through_its_wire_name() {
        assert_eq!(OcrCanonicalField::ALL.len(), 13);
        for field in OcrCanonicalField::ALL {
            assert_eq!(
                OcrCanonicalField::from_wire_name(field.wire_name()),
                Some(field)
            );
        }
        assert_eq!(OcrCanonicalField::from_wire_name("provider_private"), None);
    }
}
