use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_types::responses::main::ResponsesApiResponse;

use super::{
    Error,
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

impl crate::CoreClient {
    pub fn responses_machine(
        &self,
    ) -> Result<
        impl FnOnce(ResponsesCall) -> HostedMachine<Responses> + Send + Sync + use<>,
        litellm_http::Error,
    > {
        let http = self.provider_http()?;
        let auth = self.resources().auth.clone();
        let secrets = self.secret_source().clone();
        Ok(move |request| {
            hosted_call(request, move |call, _, hooks| async move {
                super::execute(Ok(http), &auth, secrets.as_ref(), call, &hooks).await
            })
        })
    }
}
