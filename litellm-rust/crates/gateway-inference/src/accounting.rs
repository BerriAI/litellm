use crate::{AccountingError, PluginError};
use bytes::Bytes;
use litellm_accounting::{
    ApplyResult, BudgetAdmission, Charges, Cost, Effect, Outcome, Progress, ProviderWork,
    ReportedUsage, Session, Settlement, SettlementStatus, Terminal,
};
use litellm_core::{RouteError, responses::route::Responses};
use litellm_gateway_auth::AuthenticatedCaller;
use litellm_host::{
    HookError,
    hooks::{CallInterceptors, CallOutcome},
    interceptors::{ProviderInterceptors, RawResponse, RequestContext, WireRequest},
};
use litellm_types::responses::main::ResponsesApiResponse;
use serde_json::{Map, Value};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::{mpsc, oneshot};
use tokio_util::task::TaskTracker;
use tracing::{Instrument, instrument::WithSubscriber};

pub type AccountingFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
pub type AccountingSettlement<'a> = Settlement<'a, String, Value, PluginError>;

pub struct AdmissionRequest {
    pub caller: AuthenticatedCaller,
    pub public_model: String,
    pub deployment_model: String,
    pub request: Map<String, Value>,
}

pub struct AdmittedCall {
    pub call_id: String,
    pub budget_receipt: Option<String>,
    pub backend: Box<dyn AccountingBackend>,
}

pub trait AccountingService: Send + Sync {
    fn admit(
        &self,
        request: AdmissionRequest,
    ) -> AccountingFuture<'_, Result<AdmittedCall, PluginError>>;
}

pub enum Fact {
    Prepared(RequestContext),
    ProviderResponse(RawResponse),
    Response(ResponsesApiResponse),
    Chunk(Bytes),
}

pub struct AccountingInputs {
    pub work: ProviderWork,
    pub usage: ReportedUsage<Value>,
}

pub trait AccountingBackend: Send {
    fn observe(&mut self, fact: Fact) -> Result<(), PluginError>;
    fn inputs(&self) -> AccountingInputs;
    fn price(&mut self, outcome: Outcome) -> AccountingFuture<'_, Result<Charges, PluginError>>;
    fn apply<'a>(
        &'a mut self,
        effect: Effect,
        settlement: AccountingSettlement<'a>,
    ) -> AccountingFuture<'a, ApplyResult<PluginError>>;
}

#[derive(Clone)]
pub struct Report {
    pub call_id: String,
    pub terminal: Terminal<Value>,
    pub status: SettlementStatus,
    pub progress: Progress<PluginError>,
    pub assessment_error: Option<PluginError>,
}

pub trait AccountingCallback: Send + Sync {
    fn completed(&self, report: Report) -> AccountingFuture<'_, Result<(), PluginError>>;
}

#[derive(Clone)]
pub struct ResponsesAccounting {
    service: Arc<dyn AccountingService>,
    callback: Option<Arc<dyn AccountingCallback>>,
    tasks: TaskTracker,
    closed: Arc<AtomicBool>,
}

impl ResponsesAccounting {
    pub fn new(service: Arc<dyn AccountingService>) -> Self {
        Self {
            service,
            callback: None,
            tasks: TaskTracker::new(),
            closed: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_callback(mut self, callback: Arc<dyn AccountingCallback>) -> Self {
        self.callback = Some(callback);
        self
    }

    pub async fn shutdown(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.tasks.close();
        self.tasks.wait().await;
    }

    pub(crate) async fn begin(&self, request: AdmissionRequest) -> Result<Call, AccountingError> {
        let token = self.tasks.token();
        if self.closed.load(Ordering::SeqCst) {
            return Err(AccountingError::Closed);
        }
        let (sender, receiver) = mpsc::channel(1);
        let (ready, admitted) = oneshot::channel();
        let service = self.service.clone();
        let callback = self.callback.clone();
        tokio::spawn(async move {
            let _token = token;
            let admission = match service.admit(request).await {
                Ok(admission) => admission,
                Err(error) => {
                    let _ = ready.send(Err(AccountingError::Admission(error)));
                    return;
                }
            };
            let mut session = Session::new(BudgetAdmission::new(admission.budget_receipt));
            let mut backend = admission.backend;
            let delivered = ready.send(Ok(())).is_ok();
            let (outcome, reply, assessment_error) = if delivered {
                collect(receiver, backend.as_mut()).await
            } else {
                (Outcome::Cancelled, None, None)
            };
            let result = settle(&mut session, backend.as_mut(), admission.call_id, outcome, assessment_error).await;
            match result {
                Ok(report) => {
                    if let Some(callback) = callback
                        && callback.completed(report.clone()).await.is_err() {
                        tracing::warn!("accounting terminal callback failed");
                    }
                    let delivery = report_result(&report);
                    if delivery.is_err() {
                        tracing::error!(status = ?report.status, "call accounting needs attention");
                    }
                    if let Some(reply) = reply {
                        let _ = reply.send(delivery);
                    }
                }
                Err(error) => {
                    tracing::error!("accounting session contract failed");
                    if let Some(reply) = reply { let _ = reply.send(Err(error)); }
                }
            }
        }.with_current_subscriber().in_current_span());
        admitted.await.map_err(|_| AccountingError::Closed)??;
        Ok(Call {
            sender: Some(sender),
        })
    }
}

type TerminalReply = oneshot::Sender<Result<(), AccountingError>>;
enum Delivery {
    Fact(Fact),
    Terminal {
        outcome: Outcome,
        reply: TerminalReply,
    },
}

async fn collect(
    mut receiver: mpsc::Receiver<Delivery>,
    backend: &mut dyn AccountingBackend,
) -> (Outcome, Option<TerminalReply>, Option<PluginError>) {
    let mut error = None;
    while let Some(delivery) = receiver.recv().await {
        match delivery {
            Delivery::Fact(fact) => {
                if let Err(failure) = backend.observe(fact) {
                    error.get_or_insert(failure);
                }
            }
            Delivery::Terminal { outcome, reply } => return (outcome, Some(reply), error),
        }
    }
    (Outcome::Cancelled, None, error)
}

async fn settle(
    session: &mut Session<String, Value, PluginError>,
    backend: &mut dyn AccountingBackend,
    call_id: String,
    outcome: Outcome,
    observation_error: Option<PluginError>,
) -> Result<Report, AccountingError> {
    let inputs = backend.inputs();
    let pricing = match observation_error {
        Some(error) => Err(error),
        None => backend.price(outcome).await,
    };
    let checked = pricing.and_then(|charges| {
        Terminal::new(outcome, inputs.work, inputs.usage.clone(), charges)
            .map_err(|error| Arc::new(error) as PluginError)
    });
    let (terminal, assessment_error) = match checked {
        Ok(terminal) => (terminal, None),
        Err(error) => (
            Terminal::new(
                outcome,
                inputs.work,
                inputs.usage,
                Charges::new(Cost::Unknown, Cost::Unknown).map_err(AccountingError::Contract)?,
            )
            .map_err(AccountingError::Contract)?,
            Some(error),
        ),
    };
    session
        .finish(terminal.clone())
        .map_err(AccountingError::Contract)?;
    while let Some(pending) = session.next_effect().map_err(AccountingError::Contract)? {
        let litellm_accounting::PendingEffect { effect, settlement } = pending;
        let result = backend.apply(effect, settlement).await;
        session
            .complete_effect(effect, result)
            .map_err(AccountingError::Contract)?;
    }
    Ok(Report {
        call_id,
        terminal,
        status: session.status(),
        progress: session.progress().clone(),
        assessment_error,
    })
}

fn report_result(report: &Report) -> Result<(), AccountingError> {
    if let Some(error) = &report.assessment_error {
        return Err(AccountingError::Assessment(error.clone()));
    }
    match report.status {
        SettlementStatus::Committed | SettlementStatus::Accepted => Ok(()),
        status => Err(AccountingError::Incomplete { status }),
    }
}

#[derive(Clone)]
pub(crate) struct Call {
    sender: Option<mpsc::Sender<Delivery>>,
}
impl Call {
    async fn fact(&self, fact: Fact) -> Result<(), AccountingError> {
        self.sender
            .as_ref()
            .ok_or(AccountingError::Closed)?
            .send(Delivery::Fact(fact))
            .await
            .map_err(|_| AccountingError::Closed)
    }
}

impl ProviderInterceptors<RouteError> for Call {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, RouteError> {
        let optional_params = match context.optional_params {
            Value::Object(params) => Value::Object(
                params
                    .into_iter()
                    .filter(|(key, _)| !context.secret_fields.contains(key))
                    .collect(),
            ),
            _ => Value::Null,
        };
        self.fact(Fact::Prepared(RequestContext {
            api_key: None,
            optional_params,
            ..context
        }))
        .await
        .map_err(hook_route_error)?;
        Ok(wire)
    }
    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), RouteError> {
        self.fact(Fact::ProviderResponse(raw))
            .await
            .map_err(hook_route_error)
    }
}
fn hook_route_error(error: AccountingError) -> RouteError {
    RouteError::InvalidResponse(litellm_llms::ErrorDetail::failed(
        "accounting fact delivery",
        error,
    ))
}

impl CallInterceptors<Responses> for Call {
    async fn transform_response(
        &mut self,
        response: ResponsesApiResponse,
    ) -> Result<ResponsesApiResponse, HookError> {
        self.fact(Fact::Response(response.clone()))
            .await
            .map_err(HookError::new)?;
        Ok(response)
    }
    fn on_stream_chunk(
        &mut self,
        chunk: &Bytes,
    ) -> impl Future<Output = Result<(), HookError>> + Send {
        let chunk = chunk.clone();
        async move { self.fact(Fact::Chunk(chunk)).await.map_err(HookError::new) }
    }
    async fn on_terminal(&mut self, outcome: CallOutcome) -> Result<(), HookError> {
        let Some(sender) = self.sender.take() else {
            return Ok(());
        };
        let outcome = match outcome {
            CallOutcome::Succeeded => Outcome::Succeeded,
            CallOutcome::Failed => Outcome::Failed,
            CallOutcome::Cancelled => Outcome::Cancelled,
        };
        let (reply, settled) = oneshot::channel();
        sender
            .send(Delivery::Terminal { outcome, reply })
            .await
            .map_err(|_| HookError::new(AccountingError::Closed))?;
        drop(sender);
        settled
            .await
            .map_err(|_| HookError::new(AccountingError::Closed))?
            .map_err(HookError::new)
    }
    fn on_cancel(&mut self) {
        self.sender.take();
    }
}
