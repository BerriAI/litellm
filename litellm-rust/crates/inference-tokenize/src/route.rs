use std::convert::Infallible;

use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    observation::ObservationSender,
    protocol::Protocol,
};
use litellm_llms_types::formats::tokenize::TokenizeResponse;

use crate::{Error, TokenizeRequest, TokenizeRoute};

pub type TokenizeCall = TokenizeRequest;

pub struct Tokenize;

impl Protocol for Tokenize {
    type Response = TokenizeResponse;
    type Error = Error;
    type Request = TokenizeCall;
    type HostCall = Infallible;
    type Chunk = Infallible;
    type StreamHead = Infallible;
}

pub type TokenizeMachine = HostedMachine<Tokenize>;

impl TokenizeRoute {
    pub fn machine(
        self,
        request: TokenizeCall,
        observers: Option<ObservationSender>,
    ) -> TokenizeMachine {
        hosted_call(
            request,
            observers,
            move |request, _, interceptors, observers| async move {
                self.run(request, &interceptors, observers.as_ref())
                    .await
                    .map(CallOutput::Complete)
            },
        )
    }
}
