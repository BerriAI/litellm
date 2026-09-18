use std::sync::{Arc, Mutex};

use litellm_auth::ResolvedCredential;
use litellm_callbacks::{
    event::{CallEvent, RequestContext, WireRequest},
    route::Route,
};

use super::{
    Error, LiteLLMOcrRequest, LiteLLMOcrResponse, OcrClient,
    handler::perform_ocr_request,
    types::{OcrDocumentInput, OcrFileContent, ResolvedOcrRequest},
};
use crate::machine::{HostChannel, HostTokenProvider, MachineFault, RouteMachine, TokenRoute};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrOp {
    ProjectRequest,
    ReadDocument,
    AcquireAzureAdToken,
}

pub enum OcrOpResult {
    Request {
        request: Box<LiteLLMOcrRequest<OcrDocumentInput>>,
        caller_token: bool,
    },
    Document(OcrFileContent),
    AzureAdToken(ResolvedCredential),
}

pub struct Ocr;

impl Route for Ocr {
    type Response = LiteLLMOcrResponse;
    type Error = Error;
    type Op = OcrOp;
    type OpResult = OcrOpResult;
}

impl TokenRoute for Ocr {
    fn acquire_token_op() -> OcrOp {
        OcrOp::AcquireAzureAdToken
    }

    fn token_credential(result: OcrOpResult) -> Option<ResolvedCredential> {
        match result {
            OcrOpResult::AzureAdToken(credential) => Some(credential),
            _ => None,
        }
    }
}

impl From<MachineFault> for Error {
    fn from(fault: MachineFault) -> Self {
        Self::InvalidRequest(match fault {
            MachineFault::Abandoned => "OCR host driver was abandoned".into(),
            MachineFault::Protocol(message) => format!("OCR {message}"),
            MachineFault::Mismatch => "invalid OCR host operation result".into(),
        })
    }
}

pub type OcrHost = HostChannel<Ocr>;
pub type OcrMachine = RouteMachine<Ocr>;

/// The OCR call as a machine: projection, document reading and token acquisition are
/// host operations; everything else runs in Rust.
pub fn ocr_machine(client: OcrClient) -> OcrMachine {
    RouteMachine::new(move |host| Box::pin(execute(client, host)))
}

async fn execute(client: OcrClient, host: OcrHost) -> Result<LiteLLMOcrResponse, Error> {
    let OcrOpResult::Request {
        request,
        caller_token,
    } = host.route(OcrOp::ProjectRequest).await?
    else {
        return Err(MachineFault::Mismatch.into());
    };
    let request = LiteLLMOcrRequest {
        azure_ad_token_provider: caller_token
            .then(|| HostTokenProvider::handle(host.clone()))
            .or(request.azure_ad_token_provider),
        ..*request
    };
    let caller_document = matches!(request.document, OcrDocumentInput::Document(_));
    let request = prepare_request_document(request, &host).await?;
    perform_ocr_request(&client, request, &host, caller_document).await
}

async fn prepare_request_document(
    request: LiteLLMOcrRequest<OcrDocumentInput>,
    host: &OcrHost,
) -> Result<ResolvedOcrRequest, Error> {
    let request = match &request.document {
        OcrDocumentInput::HostReader { mime_type } => {
            let mime_type = mime_type.clone();
            let OcrOpResult::Document(content) = host.route(OcrOp::ReadDocument).await? else {
                return Err(MachineFault::Mismatch.into());
            };
            request.with_document(OcrDocumentInput::Bytes {
                bytes: content.bytes,
                file_name: content.file_name,
                mime_type,
            })
        }
        _ => request,
    };
    if let OcrDocumentInput::Document(_) = &request.document {
        return request.map_document(super::document::prepare_document);
    }
    tokio::task::spawn_blocking(move || request.map_document(super::document::prepare_document))
        .await
        .map_err(|error| Error::DocumentTask(Arc::new(error)))?
}

type Reader = Box<dyn Fn() -> Result<OcrFileContent, Error> + Send + Sync>;
type BeforeSend =
    Box<dyn Fn(WireRequest, &RequestContext) -> Result<WireRequest, Error> + Send + Sync>;
type Observer = Box<dyn Fn(&CallEvent) + Send + Sync>;

/// The in-process host for a request that is already in hand: the request answers
/// projection, and the optional observer sees and may rewrite the wire request.
pub struct LocalOcrHost {
    request: Mutex<Option<LiteLLMOcrRequest<OcrDocumentInput>>>,
    reader: Option<Reader>,
    before_send: Option<BeforeSend>,
    observer: Option<Observer>,
}

impl LocalOcrHost {
    pub fn new(request: LiteLLMOcrRequest<OcrDocumentInput>) -> Self {
        Self {
            request: Mutex::new(Some(request)),
            reader: None,
            before_send: None,
            observer: None,
        }
    }

    pub fn with_reader(
        self,
        reader: impl Fn() -> Result<OcrFileContent, Error> + Send + Sync + 'static,
    ) -> Self {
        Self {
            reader: Some(Box::new(reader)),
            ..self
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

impl litellm_callbacks::host::Host<Ocr> for LocalOcrHost {
    async fn route(&self, op: OcrOp) -> Result<OcrOpResult, Error> {
        match op {
            OcrOp::ProjectRequest => self
                .request
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
                .map(|request| OcrOpResult::Request {
                    request: Box::new(request),
                    caller_token: false,
                })
                .ok_or_else(|| Error::InvalidRequest("OCR request was already projected".into())),
            OcrOp::ReadDocument => self
                .reader
                .as_ref()
                .ok_or_else(|| Error::InvalidRequest("OCR host has no document reader".into()))
                .and_then(|reader| reader())
                .map(OcrOpResult::Document),
            OcrOp::AcquireAzureAdToken => {
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
