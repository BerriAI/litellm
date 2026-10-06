use std::time::Duration;

use bytes::Bytes;
use litellm_cache_response::{CacheKeyInput, Deployment, extra_headers};
use litellm_host::call::CallOutput;
use litellm_inference::{
    RouteError,
    caching::{Cachable, CacheKeyProjection},
};
use litellm_llms::base_llm::{
    auth::ValidatedEnvironment, responses::transformation::BaseResponsesApiConfig,
};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use serde_json::{Map, Value};

use super::Error;

pub struct ResponsesCall {
    pub model: String,
    pub input: Value,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub struct ResponsesStreamHead {
    pub headers: Vec<(String, String)>,
}

pub type ResponsesOutput = CallOutput<ResponsesApiResponse, ResponsesStreamHead, Bytes, Error>;

pub(super) struct ProviderResponsesRequest {
    pub config: &'static dyn BaseResponsesApiConfig,
    pub environment: ValidatedEnvironment,
    pub url: String,
    pub body: Value,
    pub context: litellm_host::interceptors::RequestContext,
    pub timeout: Option<Duration>,
}

impl CacheKeyProjection for ResponsesCall {
    fn cache_key_input(&self) -> Result<CacheKeyInput, RouteError> {
        Ok(CacheKeyInput::forwarded(
            <crate::route::Responses as Cachable>::SURFACE,
            Deployment::new(
                &self.model,
                self.custom_llm_provider.as_deref(),
                self.api_base.as_deref(),
            ),
            self.optional_params
                .clone()
                .into_iter()
                .chain([("input".into(), self.input.clone())]),
            extra_headers(self.extra_headers.as_ref()),
        ))
    }
}
