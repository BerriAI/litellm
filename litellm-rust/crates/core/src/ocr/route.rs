use litellm_auth::ResolvedCredential;
use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    machine::{HostTokenProvider, TokenProtocol},
    protocol::Protocol,
    protocol::Reply,
};
use litellm_llms::base_llm::ocr::{error::Error, transformation::LiteLLMOcrResponse};

use crate::ocr::types::{LiteLLMOcrRequest, OcrDocumentInput};

pub enum OcrOp {
    AcquireAzureAdToken(Reply<ResolvedCredential>),
}

/// The caller's request as the host projects it.
pub struct OcrCall {
    pub request: LiteLLMOcrRequest<OcrDocumentInput>,
    /// The caller passed its own Azure AD token provider, which the host keeps.
    pub caller_token: bool,
}

pub struct Ocr;

impl Protocol for Ocr {
    type Response = LiteLLMOcrResponse;
    type Error = Error;
    type Request = OcrCall;
    type HostCall = OcrOp;
    type Chunk = std::convert::Infallible;
    type StreamHead = std::convert::Infallible;
}

impl TokenProtocol for Ocr {
    fn acquire_token_op(reply: Reply<ResolvedCredential>) -> OcrOp {
        OcrOp::AcquireAzureAdToken(reply)
    }
}

pub type OcrMachine = HostedMachine<Ocr>;

impl crate::ocr::OcrRoute {
    pub fn machine(self, request: OcrCall) -> OcrMachine {
        hosted_call(
            request,
            move |projection: OcrCall, services, hooks| async move {
                let request = LiteLLMOcrRequest {
                    azure_ad_token_provider: projection
                        .caller_token
                        .then(|| HostTokenProvider::handle(services))
                        .or(projection.request.azure_ad_token_provider),
                    ..projection.request
                };
                self.run(request, &hooks).await.map(CallOutput::Complete)
            },
        )
    }
}
