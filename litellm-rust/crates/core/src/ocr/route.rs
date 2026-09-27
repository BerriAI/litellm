use std::sync::{Arc, Mutex};

use litellm_auth::ResolvedCredential;
use litellm_host::{
    event::{CallEvent, RequestContext, WireRequest},
    host::Reply,
    machine::{CallMachine, HostChannel, HostTokenProvider, TokenProtocol},
    protocol::Protocol,
};
use litellm_llms::base_llm::ocr::{
    error::Error, handler::OcrClient, transformation::LiteLLMOcrResponse,
};

use super::handler::perform_ocr_request;
use crate::ocr::types::{LiteLLMOcrRequest, OcrDocumentInput, ResolvedOcrRequest};

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

pub type OcrHost = HostChannel<Ocr>;
pub type OcrMachine = CallMachine<Ocr>;

/// The OCR call as a machine: projection and token acquisition are host operations;
/// everything else runs in Rust.
pub fn ocr_machine(client: OcrClient) -> OcrMachine {
    CallMachine::new(move |host| Box::pin(execute(client, host)))
}

async fn execute(client: OcrClient, host: OcrHost) -> Result<LiteLLMOcrResponse, Error> {
    let OcrProjection {
        request,
        caller_token,
    } = host.project().await?;
    let request = LiteLLMOcrRequest {
        azure_ad_token_provider: caller_token
            .then(|| HostTokenProvider::handle(host.clone()))
            .or(request.azure_ad_token_provider),
        ..request
    };
    let caller_document = matches!(request.document, OcrDocumentInput::Document(_));
    let request = prepare_request_document(request).await?;
    perform_ocr_request(&client, request, &host, caller_document).await
}

async fn prepare_request_document(
    request: LiteLLMOcrRequest<OcrDocumentInput>,
) -> Result<ResolvedOcrRequest, Error> {
    if let OcrDocumentInput::Document(_) = &request.document {
        return request.map_document(super::document::prepare_document);
    }
    tokio::task::spawn_blocking(move || request.map_document(super::document::prepare_document))
        .await
        .map_err(|error| Error::DocumentTask(Arc::new(error)))?
}

type BeforeSend =
    Box<dyn Fn(WireRequest, &RequestContext) -> Result<WireRequest, Error> + Send + Sync>;
type Observer = Box<dyn Fn(&CallEvent) + Send + Sync>;

/// The in-process host for a request that is already in hand: the request answers
/// projection, and the optional observer sees and may rewrite the wire request.
pub struct LocalOcrHost {
    request: Mutex<Option<LiteLLMOcrRequest<OcrDocumentInput>>>,
    before_send: Option<BeforeSend>,
    observer: Option<Observer>,
}

impl LocalOcrHost {
    pub fn new(request: LiteLLMOcrRequest<OcrDocumentInput>) -> Self {
        Self {
            request: Mutex::new(Some(request)),
            before_send: None,
            observer: None,
        }
    }

    pub fn with_before_send(
        self,
        before_send: impl Fn(WireRequest, &RequestContext) -> Result<WireRequest, Error>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            before_send: Some(Box::new(before_send)),
            ..self
        }
    }

    pub fn with_observer(self, observer: impl Fn(&CallEvent) + Send + Sync + 'static) -> Self {
        Self {
            observer: Some(Box::new(observer)),
            ..self
        }
    }
}

impl litellm_host::host::Host<Ocr> for LocalOcrHost {
    async fn project(&self) -> Result<OcrProjection, Error> {
        self.request
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .map(|request| OcrProjection {
                request,
                caller_token: false,
            })
            .ok_or_else(|| Error::InvalidRequest("OCR request was already projected".into()))
    }

    async fn custom_op(&self, op: OcrOp) -> Result<(), Error> {
        match op {
            OcrOp::AcquireAzureAdToken(_) => {
                Err(Error::Auth(litellm_auth::Error::AzureTokenAcquisition(
                    "OCR host has no Azure AD token provider".into(),
                )))
            }
        }
    }

    async fn before_send(
        &self,
        wire: WireRequest,
        context: &RequestContext,
    ) -> Result<WireRequest, Error> {
        match &self.before_send {
            Some(before_send) => before_send(wire, context),
            None => Ok(wire),
        }
    }

    async fn emit(&self, event: &CallEvent) -> Result<(), Error> {
        if let Some(observer) = &self.observer {
            observer(event);
        }
        Ok(())
    }
}
