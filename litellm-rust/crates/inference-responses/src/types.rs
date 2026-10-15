use std::time::Duration;

use bytes::Bytes;
use litellm_host::call::CallOutput;
use litellm_llms::base_llm::{
    auth::ValidatedEnvironment, responses::transformation::BaseResponsesApiConfig,
};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value};

use super::Error;

pub struct ResponsesCall {
    pub model: String,
    pub input: Value,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub api_version: Option<String>,
    /// The deployment's provider settings, such as AWS credentials and region
    pub litellm_params: LitellmParams,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub struct ResponsesStreamHead {
    pub headers: Vec<(String, String)>,
}

pub type ResponsesOutput = CallOutput<ResponsesApiResponse, ResponsesStreamHead, Bytes, Error>;

pub(super) struct ProviderResponsesRequest {
    pub config: Box<dyn BaseResponsesApiConfig + Send>,
    pub environment: ValidatedEnvironment,
    pub url: String,
    pub body: Value,
    pub context: litellm_host::interceptors::RequestContext,
    pub timeout: Option<Duration>,
}
