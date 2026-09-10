mod transformation;
pub mod types;

use types::*;

use crate::ocr::formats::ocr_format;
use crate::ocr::types::OcrDocument;

ocr_format! {
    pub MistralOcrFormat {
        params: MistralOcrParams,
        document: OcrDocument,
        request: MistralOcrRequest => transformation::request,
        response: MistralOcrResponse => transformation::response,
    }
}
