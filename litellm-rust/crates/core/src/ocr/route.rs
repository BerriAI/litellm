use litellm_auth::ResolvedCredential;
use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    host::Reply,
    machine::{HostTokenProvider, TokenProtocol},
    protocol::Protocol,
};
use litellm_llms::base_llm::ocr::{
    error::Error, handler::OcrClient, transformation::LiteLLMOcrResponse,
};

use crate::ocr::types::{LiteLLMOcrRequest, OcrDocumentInput};

pub enum OcrOp {
    AcquireAzureAdToken(Reply<ResolvedCredential>),
}

/// The caller's request as the host projects it.
pub struct OcrProjection {
    pub request: LiteLLMOcrRequest<OcrDocumentInput>,
    /// The caller passed its own Azure AD token provider, which the host keeps.
    pub caller_token: bool,
}

pub struct Ocr;

impl Protocol for Ocr {
    type Response = LiteLLMOcrResponse;
    type Error = Error;
    type Projection = OcrProjection;
    type Op = OcrOp;
    type Chunk = std::convert::Infallible;
    type StreamHead = std::convert::Infallible;
}

impl TokenProtocol for Ocr {
    fn acquire_token_op(reply: Reply<ResolvedCredential>) -> OcrOp {
        OcrOp::AcquireAzureAdToken(reply)
    }
}

pub type OcrMachine = HostedMachine<Ocr>;

pub fn ocr_machine(client: OcrClient) -> OcrMachine {
    hosted_call(move |projection: OcrProjection, host| async move {
        let request = LiteLLMOcrRequest {
            azure_ad_token_provider: projection
                .caller_token
                .then(|| HostTokenProvider::handle(host.clone()))
                .or(projection.request.azure_ad_token_provider),
            ..projection.request
        };
        super::client::perform_with_hooks(&client, request, &host)
            .await
            .map(CallOutput::Complete)
    })
}
