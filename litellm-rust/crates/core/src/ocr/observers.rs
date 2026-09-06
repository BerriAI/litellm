use std::collections::BTreeMap;
use std::convert::Infallible;

use serde::Serialize;

use crate::provider_callbacks::CallbackDecision;
use crate::request_context::{RequestAttribution, RequestCapabilities};

use super::types::OcrRequestData;

#[derive(Serialize)]
pub struct OcrPreCall {
    pub call_id: Option<String>,
    pub trace_id: Option<String>,
    pub requested_model: Option<String>,
    pub attribution: RequestAttribution,
    pub capabilities: RequestCapabilities,
    pub model: String,
    pub request: OcrRequestData,
    pub api_base: String,
    pub headers: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct OcrPostCall {
    pub call_id: Option<String>,
    pub trace_id: Option<String>,
    pub requested_model: Option<String>,
    pub attribution: RequestAttribution,
    pub capabilities: RequestCapabilities,
    pub original_response: String,
}

#[macro_export]
macro_rules! ocr_observer_catalog {
    ($consumer:path, $($options:tt)*) => {
        $consumer! {
            $($options)*
            {
                pre_call: PreCall($crate::ocr::observers::OcrPreCall) -> $crate::provider_callbacks::CallbackDecision = direct;
                post_call: PostCall($crate::ocr::observers::OcrPostCall) -> $crate::provider_callbacks::CallbackDecision = direct;
            }
        }
    };
}

ocr_observer_catalog!(crate::define_hooks, pub trait OcrObserver;);

pub struct NoopOcrObserver;

impl OcrObserver for NoopOcrObserver {
    type Error = Infallible;

    async fn pre_call(&mut self, _input: &OcrPreCall) -> Result<CallbackDecision, Infallible> {
        Ok(CallbackDecision::Unchanged)
    }

    async fn post_call(&mut self, _input: &OcrPostCall) -> Result<CallbackDecision, Infallible> {
        Ok(CallbackDecision::Unchanged)
    }
}
