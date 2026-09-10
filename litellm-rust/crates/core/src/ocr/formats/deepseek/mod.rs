mod transformation;
pub mod types;

use types::*;

use crate::ocr::formats::ocr_format;
use crate::ocr::types::OcrDocument;

ocr_format! {
    pub DeepSeekOcrFormat {
        params: DeepSeekOcrParams,
        document: OcrDocument,
        request: DeepSeekOcrRequest => transformation::request,
        response: DeepSeekOcrResponse => transformation::response,
    }
}
