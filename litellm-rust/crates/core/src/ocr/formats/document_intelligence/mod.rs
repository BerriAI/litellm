mod params;
mod transformation;
pub mod types;

use types::*;

use crate::ocr::formats::ocr_format;
use crate::ocr::types::OcrDocument;

ocr_format! {
    pub AzureDocumentIntelligenceOcrFormat {
        params: DocumentIntelligenceInputParams => DocumentIntelligenceParams,
        validate: params::validate,
        map: params::normalize,
        document: OcrDocument,
        request: DocumentIntelligenceRequest => transformation::request,
        response: AzureDocumentIntelligenceOperation => transformation::response,
    }
}
