use std::collections::BTreeMap;
use std::convert::Infallible;

use serde::Serialize;

use super::types::OcrRequestData;

#[derive(Serialize)]
pub struct OcrPreCall {
    pub model: String,
    pub request: OcrRequestData,
    pub api_base: String,
    pub headers: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct OcrPostCall {
    pub original_response: String,
}

#[macro_export]
macro_rules! ocr_observer_catalog {
    ($consumer:path, $($options:tt)*) => {
        $consumer! {
            $($options)*
            {
                pre_call: PreCall($crate::ocr::observers::OcrPreCall) -> () = direct;
                post_call: PostCall($crate::ocr::observers::OcrPostCall) -> () = direct;
            }
        }
    };
}

ocr_observer_catalog!(crate::define_hooks, pub trait OcrObserver;);

pub struct NoopOcrObserver;

impl OcrObserver for NoopOcrObserver {
    type Error = Infallible;

    async fn pre_call(&mut self, _input: &OcrPreCall) -> Result<(), Infallible> {
        Ok(())
    }

    async fn post_call(&mut self, _input: &OcrPostCall) -> Result<(), Infallible> {
        Ok(())
    }
}
