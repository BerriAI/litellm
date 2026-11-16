use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_llms_types::formats::responses::ResponsesApiResponse;

use super::{
    Error, ResponsesRoute,
    types::{ResponsesPlan, ResponsesStreamHead},
};

pub struct Responses;

impl Protocol for Responses {
    type Response = ResponsesApiResponse;
    type Error = Error;
    type Request = ResponsesPlan;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ResponsesStreamHead;
}

impl ResponsesRoute {
    pub fn machine(
        self,
        plan: ResponsesPlan,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> HostedMachine<Responses> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        hosted_call(
            plan,
            observers,
            move |plan, _, interceptors, observers| async move {
                self.run(plan, cache_options, &interceptors, observers.as_ref())
                    .await
            },
        )
    }
}

impl litellm_inference::caching::Cachable for Responses {
    const SURFACE: &'static str = "responses";

    fn reusable(response: &Self::Response) -> bool {
        response
            .extra
            .get("status")
            .and_then(serde_json::Value::as_str)
            == Some("completed")
    }
}

impl litellm_inference::caching::StreamCachable for Responses {
    const TERMINAL_EVENT: &'static str = "response.completed";

    fn replay(data: bytes::Bytes) -> Option<litellm_host::call::OutputOf<Self>> {
        Some(litellm_host::call::CallOutput::Stream {
            head: ResponsesStreamHead {
                headers: Vec::new(),
            },
            chunks: Box::pin(futures_util::stream::iter([Ok(data)])),
        })
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        chunk.as_ref()
    }
}
