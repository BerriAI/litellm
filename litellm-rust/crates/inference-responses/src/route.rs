use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_llms_types::formats::responses::ResponsesApiResponse;

use super::{Error, ResponsesRoute, types::ResponsesCall};

pub struct Responses;

impl Protocol for Responses {
    type Response = http::Response<ResponsesApiResponse>;
    type Error = Error;
    type Request = ResponsesCall;
    type HostCall = Infallible;
    type Chunk = Bytes;
    /// The upstream status line and headers, as the caller sees them before any chunk.
    type StreamHead = http::response::Parts;
}

impl ResponsesRoute {
    pub fn machine(
        self,
        call: ResponsesCall,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> HostedMachine<Responses> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        hosted_call(
            call,
            observers,
            move |call, _, interceptors, observers| async move {
                self.run(call, cache_options, &interceptors, observers.as_ref())
                    .await
            },
        )
    }
}

impl litellm_inference::caching::Cachable for Responses {
    const SURFACE: &'static str = "responses";
    type Body = ResponsesApiResponse;

    fn reusable(response: &Self::Response) -> bool {
        response
            .body()
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
            head: http::Response::new(()).into_parts().0,
            chunks: Box::pin(futures_util::stream::iter([Ok(data)])),
        })
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        chunk.as_ref()
    }
}
