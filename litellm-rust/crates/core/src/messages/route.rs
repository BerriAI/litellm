use litellm_http::response::ResponseHead;
use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedCompletion, HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_llms_types::formats::messages::MessagesResponse;

use super::{Error, MessagesCall};

pub type MessagesOutput = HostedCompletion<litellm_http::response::Response<Box<MessagesResponse>>>;

pub struct Messages;

impl Protocol for Messages {
    type Response = litellm_http::response::Response<Box<MessagesResponse>>;
    type Error = Error;
    type Request = MessagesCall;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ResponseHead;
}

pub type MessagesMachine = HostedMachine<Messages>;

impl super::MessagesRoute {
    pub fn machine(
        self,
        request: super::MessagesCall,
        options: impl Into<crate::CallOptions>,
    ) -> MessagesMachine {
        let crate::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        hosted_call(
            request,
            observers,
            move |call, _, interceptors, observers| async move {
                let context = crate::context::CallContext::new(
                    &interceptors,
                    crate::CallOptions {
                        cache: cache_options,
                        observers,
                    },
                );
                self.run(call, context).await
            },
        )
    }
}

impl crate::caching::Cachable for Messages {
    type Body = Box<MessagesResponse>;

    const SURFACE: &'static str = "messages";
}

impl crate::caching::StreamCachable for Messages {
    const TERMINAL_EVENT: &'static str = "message_stop";

    fn replay(data: bytes::Bytes) -> Option<litellm_host::call::OutputOf<Self>> {
        Some(litellm_host::call::CallOutput::Stream {
            head: ResponseHead::cached(),
            chunks: Box::pin(futures_util::stream::iter([Ok(data)])),
        })
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        chunk.as_ref()
    }
}
