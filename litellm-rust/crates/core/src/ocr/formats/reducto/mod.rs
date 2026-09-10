mod transformation;
pub mod types;

use types::*;

use crate::ocr::formats::ocr_format;

ocr_format! {
    pub ReductoParseV3Format {
        params: ReductoV3Params,
        document: ReductoFileId,
        request: ReductoV3Request => transformation::v3_request,
        response: ReductoResponse => transformation::response,
    }
}

ocr_format! {
    pub ReductoParseLegacyFormat {
        params: ReductoLegacyParams,
        document: ReductoFileId,
        request: ReductoLegacyRequest => transformation::legacy_request,
        response: ReductoResponse => transformation::response,
    }
}
