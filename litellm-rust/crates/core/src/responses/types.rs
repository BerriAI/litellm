use std::time::Duration;

use bytes::Bytes;
use futures_util::stream::BoxStream;
use litellm_llms::openai::responses::transformation::ResponsesResponse;
use serde_json::{Map, Value};

use crate::RouteError;

pub struct ResponsesRequest<'a> {
    pub model: &'a str,
    pub body: Map<String, Value>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub enum ResponsesBody {
    Response(ResponsesResponse),
    Stream(BoxStream<'static, Result<Bytes, RouteError>>),
}

pub struct ResponsesOutput {
    pub headers: reqwest::header::HeaderMap,
    pub body: ResponsesBody,
}
