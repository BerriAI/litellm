use litellm_host::observation::ObservationSender;
use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_types::responses::main::ResponsesApiResponse;

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
        observers: Option<ObservationSender>,
    ) -> HostedMachine<Responses> {
        hosted_call(
            call,
            observers,
            move |call, _, interceptors, observers| async move {
                self.run(call, &interceptors, observers.as_ref()).await
            },
        )
    }
}
