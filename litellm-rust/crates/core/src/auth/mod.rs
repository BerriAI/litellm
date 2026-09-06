mod credentials;
mod providers;

pub use credentials::{
    AzureCredentialMethod, Clock, CloudCredentialMethod, CredentialIdentity, CredentialProvenance,
    ExpiryAwareTokenProvider, GoogleCredentialMethod, SecretString, SystemClock, TokenCredential,
    TokenLease, TokenProvider,
};
pub use providers::{
    BearerAuthorizationProvider, BearerCredentialAdapter, CredentialAdapter,
    CredentialAdapterRegistry, CredentialCandidate, CredentialKind, CredentialSpec,
    StaticHeaderAuthorizationProvider, StaticHeaderCredentialAdapter,
};

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Method, StatusCode, Url};
use serde_json::Value;
use tokio::sync::Notify;

use crate::Error;

pub type OperationFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutboundOperationKind {
    Request,
    Upload,
    Submission,
    Poll,
}

#[derive(Clone, Default)]
pub struct OperationCancellation {
    state: Arc<CancellationState>,
}

#[derive(Default)]
struct CancellationState {
    cancelled: AtomicBool,
    notify: Notify,
}

impl OperationCancellation {
    pub fn cancel(&self) {
        self.state.cancelled.store(true, Ordering::Release);
        self.state.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        let notified = self.state.notify.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

#[derive(Clone)]
pub struct OperationControl {
    pub deadline: Instant,
    pub cancellation: OperationCancellation,
}

impl OperationControl {
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            deadline: Instant::now() + timeout,
            cancellation: OperationCancellation::default(),
        }
    }

    async fn run<T>(&self, work: impl Future<Output = Result<T, Error>>) -> Result<T, Error> {
        if self.cancellation.is_cancelled() {
            return Err(cancelled_error());
        }
        if self.deadline <= Instant::now() {
            return Err(deadline_error());
        }
        tokio::select! {
            biased;
            _ = self.cancellation.cancelled() => Err(cancelled_error()),
            result = tokio::time::timeout_at(self.deadline.into(), work) => {
                result.map_err(|_| deadline_error())?
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum OutboundBody {
    Bodyless,
    JsonObject(serde_json::Map<String, Value>),
    Encoded {
        bytes: Vec<u8>,
        content_type: HeaderValue,
    },
}

#[derive(Clone, Debug)]
pub enum BodyDecision {
    Unchanged,
    Replace(Value),
    Reject(String),
}

pub struct OutboundOperation {
    pub method: Method,
    pub url: Url,
    pub headers: HeaderMap,
    pub body: OutboundBody,
    pub operation: OutboundOperationKind,
}

#[derive(Clone, Debug)]
pub struct OutboundRequestView {
    pub method: Method,
    pub url: Url,
    pub headers: HeaderMap,
    pub body: OutboundBody,
}

#[derive(Debug)]
pub struct OutboundResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

pub struct AuthorizationInput<'a> {
    pub method: &'a Method,
    pub url: &'a Url,
    pub headers: &'a HeaderMap,
    pub body: Option<&'a [u8]>,
}

#[derive(Default)]
pub struct AuthorizationMutation {
    pub set_headers: Vec<(HeaderName, HeaderValue)>,
    pub remove_headers: Vec<HeaderName>,
}

pub trait RequestAuthorizer: Send + Sync {
    fn declared_headers(&self) -> &[HeaderName];
    fn authorize<'a>(
        &'a self,
        input: AuthorizationInput<'a>,
    ) -> OperationFuture<'a, AuthorizationMutation>;
}

pub trait AuthorizationProvider: Send + Sync {
    fn prepare(&self) -> OperationFuture<'_, AuthorizationPreparation>;
}

pub struct AuthorizationPreparation {
    pub authorizer: Arc<dyn RequestAuthorizer>,
    pub visible_headers: HeaderMap,
    pub remove_headers: Vec<HeaderName>,
    pub provenance: Option<CredentialProvenance>,
}

pub struct FixedAuthorization {
    authorizer: Arc<dyn RequestAuthorizer>,
    visible_headers: HeaderMap,
    remove_headers: Vec<HeaderName>,
}

impl FixedAuthorization {
    pub fn new(authorizer: Arc<dyn RequestAuthorizer>) -> Self {
        Self {
            authorizer,
            visible_headers: HeaderMap::new(),
            remove_headers: Vec::new(),
        }
    }

    pub fn with_prepared_headers(
        authorizer: Arc<dyn RequestAuthorizer>,
        visible_headers: HeaderMap,
        remove_headers: Vec<HeaderName>,
    ) -> Self {
        Self {
            authorizer,
            visible_headers,
            remove_headers,
        }
    }
}

impl AuthorizationProvider for FixedAuthorization {
    fn prepare(&self) -> OperationFuture<'_, AuthorizationPreparation> {
        Box::pin(async move {
            Ok(AuthorizationPreparation {
                authorizer: self.authorizer.clone(),
                visible_headers: self.visible_headers.clone(),
                remove_headers: self.remove_headers.clone(),
                provenance: None,
            })
        })
    }
}

pub struct NoAuthorization;

impl AuthorizationProvider for NoAuthorization {
    fn prepare(&self) -> OperationFuture<'_, AuthorizationPreparation> {
        Box::pin(async {
            Ok(AuthorizationPreparation {
                authorizer: Arc::new(NoAuthorization),
                visible_headers: HeaderMap::new(),
                remove_headers: Vec::new(),
                provenance: None,
            })
        })
    }
}

impl RequestAuthorizer for NoAuthorization {
    fn declared_headers(&self) -> &[HeaderName] {
        &[]
    }

    fn authorize<'a>(
        &'a self,
        _: AuthorizationInput<'a>,
    ) -> OperationFuture<'a, AuthorizationMutation> {
        Box::pin(async { Ok(AuthorizationMutation::default()) })
    }
}

struct ValidatedRequest {
    method: Method,
    url: Url,
    headers: HeaderMap,
    body: Option<Vec<u8>>,
}

struct AuthorizedRequest(reqwest::Request);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Origin {
    scheme: String,
    host: String,
    port: u16,
}

impl Origin {
    fn parse(url: &Url) -> Result<Self, Error> {
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(destination_error());
        }
        Ok(Self {
            scheme: url.scheme().to_owned(),
            host: url.host_str().ok_or_else(destination_error)?.to_owned(),
            port: url.port_or_known_default().ok_or_else(destination_error)?,
        })
    }
}

pub async fn send_once<Callback, CallbackFuture>(
    client: &Client,
    origin: &Url,
    request: OutboundOperation,
    callback: Callback,
    authorization: &dyn AuthorizationProvider,
    control: &OperationControl,
) -> Result<OutboundResponse, Error>
where
    Callback: FnOnce(OutboundRequestView) -> CallbackFuture,
    CallbackFuture: Future<Output = Result<BodyDecision, Error>>,
{
    let mut request = request;
    if let OutboundBody::Encoded { content_type, .. } = &request.body {
        request
            .headers
            .insert(reqwest::header::CONTENT_TYPE, content_type.clone());
    }
    validate_destination(origin, &request)?;
    let decision = control
        .run(callback(OutboundRequestView {
            method: request.method.clone(),
            url: request.url.clone(),
            headers: request.headers.clone(),
            body: request.body.clone(),
        }))
        .await?;
    let body = resolve_body(request.body, decision)?;
    let body = match body {
        OutboundBody::Bodyless => None,
        OutboundBody::JsonObject(value) => Some(serde_json::to_vec(&value).map_err(|error| {
            Error::InvalidRequest(format!("failed to encode request: {error}"))
        })?),
        OutboundBody::Encoded {
            bytes,
            content_type: _,
        } => Some(bytes),
    };
    let preparation = control.run(authorization.prepare()).await?;
    let mut headers = request.headers;
    apply_preparation(&mut headers, &preparation)?;

    let validated = ValidatedRequest {
        method: request.method,
        url: request.url,
        headers,
        body,
    };
    let authorized = authorize(validated, preparation.authorizer.as_ref(), control).await?;
    let response = control
        .run(async { client.execute(authorized.0).await.map_err(transport_error) })
        .await?;
    let status = response.status();
    let headers = response.headers().clone();
    let body = control
        .run(async {
            response
                .bytes()
                .await
                .map(Vec::from)
                .map_err(transport_error)
        })
        .await?;
    Ok(OutboundResponse {
        status,
        headers,
        body,
    })
}

fn apply_preparation(
    headers: &mut HeaderMap,
    preparation: &AuthorizationPreparation,
) -> Result<(), Error> {
    let declared: HashSet<_> = preparation
        .authorizer
        .declared_headers()
        .iter()
        .cloned()
        .collect();
    if preparation
        .visible_headers
        .keys()
        .any(|name| !declared.contains(name))
        || preparation
            .remove_headers
            .iter()
            .any(|name| !declared.contains(name))
    {
        return Err(Error::Auth(
            "credential preparation attempted an undeclared header mutation".into(),
        ));
    }
    for name in &preparation.remove_headers {
        headers.remove(name);
    }
    for (name, value) in &preparation.visible_headers {
        headers.insert(name, value.clone());
    }
    Ok(())
}

fn validate_destination(origin: &Url, request: &OutboundOperation) -> Result<(), Error> {
    if Origin::parse(origin)? != Origin::parse(&request.url)? {
        return Err(destination_error());
    }
    Ok(())
}

fn resolve_body(body: OutboundBody, decision: BodyDecision) -> Result<OutboundBody, Error> {
    match decision {
        BodyDecision::Unchanged => Ok(body),
        BodyDecision::Reject(message) => Err(Error::InvalidRequest(message)),
        BodyDecision::Replace(value) => match body {
            OutboundBody::Bodyless => Err(Error::InvalidRequest(
                "callback cannot add a body to a bodyless operation".into(),
            )),
            OutboundBody::JsonObject(_) => value
                .as_object()
                .cloned()
                .map(OutboundBody::JsonObject)
                .ok_or_else(|| {
                    Error::InvalidRequest("callback replacement must be a JSON object".into())
                }),
            OutboundBody::Encoded { .. } => Err(Error::InvalidRequest(
                "callback cannot replace an encoded operation body".into(),
            )),
        },
    }
}

async fn authorize(
    request: ValidatedRequest,
    authorizer: &dyn RequestAuthorizer,
    control: &OperationControl,
) -> Result<AuthorizedRequest, Error> {
    let mutation = control
        .run(authorizer.authorize(AuthorizationInput {
            method: &request.method,
            url: &request.url,
            headers: &request.headers,
            body: request.body.as_deref(),
        }))
        .await?;
    let declared: HashSet<_> = authorizer.declared_headers().iter().cloned().collect();
    if mutation
        .set_headers
        .iter()
        .any(|(name, _)| !declared.contains(name))
        || mutation
            .remove_headers
            .iter()
            .any(|name| !declared.contains(name))
    {
        return Err(Error::Auth(
            "authorizer attempted an undeclared header mutation".into(),
        ));
    }

    let mut headers = request.headers;
    for name in mutation.remove_headers {
        headers.remove(name);
    }
    for (name, value) in mutation.set_headers {
        headers.insert(name, value);
    }
    let mut wire = reqwest::Request::new(request.method, request.url);
    *wire.headers_mut() = headers;
    *wire.body_mut() = request.body.map(reqwest::Body::from);
    let remaining = control
        .deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(deadline_error)?;
    *wire.timeout_mut() = Some(remaining);
    Ok(AuthorizedRequest(wire))
}

fn destination_error() -> Error {
    Error::Auth("refusing to authorize a request for this destination".into())
}

fn deadline_error() -> Error {
    Error::Network("Request timed out".into())
}

fn cancelled_error() -> Error {
    Error::Network("Request cancelled".into())
}

fn transport_error(error: reqwest::Error) -> Error {
    Error::Network(if error.is_timeout() {
        "Request timed out".into()
    } else {
        error.to_string()
    })
}

#[cfg(test)]
mod tests;
