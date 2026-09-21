use std::time::Duration;

use litellm_auth::SecretValue;
use litellm_host::machine::RouteMachine;
use litellm_llms::base_llm::base_model_iterator::ResponsesApi;
use litellm_types::responses::streaming::{ResponsesRequest, ResponsesResponse};

use crate::streaming::{StreamingCall, StreamingRoute};

pub struct ResponsesCall {
    pub request: ResponsesRequest,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub extra_headers: Box<[(String, String)]>,
    pub timeout: Option<Duration>,
}

pub struct Responses;

impl StreamingCall for Responses {
    type Contract = ResponsesApi;
    type Request = ResponsesCall;
    type Response = ResponsesResponse;
}

pub fn responses_machine() -> RouteMachine<StreamingRoute<Responses>> {
    todo!(
        "Compose native Responses or Chat-to-Responses conversion with the shared Responses driver and host operations"
    )
}
