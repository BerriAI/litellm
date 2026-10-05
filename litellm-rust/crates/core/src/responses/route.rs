use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_llms_types::formats::responses::ResponsesApiResponse;

use super::{
    Error, ResponsesRoute,
    types::{ResponsesCall, ResponsesStreamHead},
};

pub struct Responses;

impl Protocol for Responses {
    type Response = ResponsesApiResponse;
    type Error = Error;
    type Request = ResponsesCall;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ResponsesStreamHead;
}

impl ResponsesRoute {
    pub fn machine(
        self,
        call: ResponsesCall,
        options: impl Into<crate::CallOptions>,
    ) -> HostedMachine<Responses> {
        let crate::CallOptions {
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

impl crate::caching::Cachable for Responses {
    const SURFACE: &'static str = "responses";

    fn reusable(response: &Self::Response) -> bool {
        response
            .extra
            .get("status")
            .and_then(serde_json::Value::as_str)
            == Some("completed")
    }
}

impl crate::caching::StreamCachable for Responses {
    const TERMINAL_EVENT: &'static str = "response.completed";

    fn replay(data: bytes::Bytes) -> Option<litellm_host::call::OutputOf<Self>> {
        Some(litellm_host::call::CallOutput::Stream {
            head: ResponsesStreamHead {
                headers: Vec::new(),
            },
            chunks: Box::pin(futures_util::stream::iter(
                litellm_framer::sse::RawBlocks::new(data).map(Ok),
            )),
        })
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        chunk.as_ref()
    }
}
