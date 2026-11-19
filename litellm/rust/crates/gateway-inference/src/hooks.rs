//! The proxy's callback hooks around one call, at the points `ProxyBaseLLMRequestProcessing`
//! runs them. The [`Host`](crate::Host) the gateway was built with answers them, so a route
//! tells what happened to its call and never learns which callbacks hear it.

use std::{future::Future, sync::Arc};

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::stream::BoxStream;
use litellm_framer::sse::SseBlock;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host::call::CallOutput;
use litellm_inference::Error as InferenceError;
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use serde_json::{Map, Value};

use crate::{
    Ended, Gateway, Host,
    stream_event::{Counts, Read},
    stream_outcome::{self, Outcome},
};

/// The route a call came in on, each named as Python's `call_type` for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum CallType {
    Acompletion,
    AnthropicMessages,
    Aresponses,
    Atranscription,
    Aocr,
}

/// A call a callback refused, as the proxy's `ProxyException` carries it to a route's handler.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Refusal {
    pub status: StatusCode,
    pub message: String,
    /// The exception's `type`, when it is not the one its status gives.
    pub kind: Option<String>,
    /// The exception's `code`, when it is not its status.
    pub code: Option<String>,
    /// In the order Python sends them.
    pub headers: Vec<(String, String)>,
}

impl Refusal {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            ..Self::default()
        }
    }
}

/// A call that was served, as the success log is told it.
#[derive(Clone, Debug, PartialEq)]
pub struct ServedCall {
    /// The model group that served it, an alias resolved, as the proxy logs it.
    pub model_group: String,
    /// As Python's litellm `Usage` counts it, `None` for an answer that carries no usage.
    pub usage: Option<ChatCompletionsUsage>,
}

/// The hooks of one call, told in the order the proxy runs them:
///
/// - [`pre_call`](Self::pre_call) before the call is routed (`pre_call_hook`)
/// - [`post_call_success`](Self::post_call_success) on a complete answer only, as
///   `post_call_success_hook` never sees a stream
/// - [`response_headers`](Self::response_headers) on every served answer
///   (`post_call_response_headers_hook`)
/// - [`succeeded`](Self::succeeded) once the call is done: a complete answer after it goes out, a
///   stream when it ends (the success log)
/// - [`failed`](Self::failed) for a call that ends on an error, a refusal by a hook included
///   (`post_call_failure_hook`, then the failure log)
///
/// A stream the client drops before its end tells neither `succeeded` nor `failed`, so a callback
/// that holds something for the call gives it back when its state for the request is dropped.
/// Accepted difference: an attempt a retry or a fallback answered for is not logged as a
/// failure, since the route only hears how its call ended.
pub trait CallHooks: Send + Sync + 'static {
    /// Every callback sees the call's `body` in turn and may change it or refuse the call. The
    /// body comes back as the callbacks left it.
    fn pre_call(
        &self,
        body: Map<String, Value>,
    ) -> impl Future<Output = Result<Map<String, Value>, Refusal>> + Send;

    fn post_call_success(&self) -> impl Future<Output = Result<(), Refusal>> + Send;

    /// The headers the callbacks add to the answer.
    fn response_headers(&self) -> impl Future<Output = HeaderMap> + Send;

    fn succeeded(&self, served: ServedCall) -> impl Future<Output = ()> + Send;

    fn failed(&self) -> impl Future<Output = ()> + Send;
}

/// A served call's answer with what the success log is told once it goes out, `None` for a
/// stream, which tells its own end.
pub(crate) struct Answer {
    response: Response,
    complete: Option<ServedCall>,
}

/// What the hooks made of a served call before its answer is rendered.
pub(crate) struct Told {
    added: HeaderMap,
    complete: Option<ServedCall>,
}

impl Told {
    /// `response` with the headers the callbacks add.
    pub(crate) fn answer(self, response: impl IntoResponse) -> Answer {
        Answer {
            response: (self.added, response).into_response(),
            complete: self.complete,
        }
    }
}

/// Serves one call of the `call_type` route with the host's hooks around it: `serve` runs on the
/// body the pre-call hook left, a complete answer is logged off the request, as the proxy logs
/// it after it went out, and a call that ends on an error, a refusal by that hook included,
/// tells the hooks it failed.
pub(crate) async fn around<H, F, Fut>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    call_type: CallType,
    body: Map<String, Value>,
    serve: F,
) -> Result<Response, Ended>
where
    H: Host,
    F: FnOnce(Arc<H::Hooks>, Map<String, Value>) -> Fut,
    Fut: Future<Output = Result<Answer, Ended>>,
{
    let hooks = Arc::new(gateway.host.hooks(identity, call_type));
    let served = match hooks.pre_call(body).await {
        Ok(body) => serve(hooks.clone(), body).await,
        Err(refusal) => Err(refusal.into()),
    };
    match served {
        Ok(Answer { response, complete }) => {
            if let Some(served) = complete {
                tokio::spawn(async move { hooks.succeeded(served).await });
            }
            Ok(response)
        }
        Err(ended) => {
            hooks.failed().await;
            Err(ended)
        }
    }
}

/// Tells the hooks about the answer of a route that never streams.
pub(crate) async fn complete<K: CallHooks>(hooks: &K, served: ServedCall) -> Result<Told, Ended> {
    hooks.post_call_success().await?;
    Ok(Told {
        added: hooks.response_headers().await,
        complete: Some(served),
    })
}

/// How a route's stream is read, and what its counts are as Python's litellm `Usage`.
#[derive(Clone, Copy)]
pub(crate) struct Format {
    pub(crate) read: Read,
    pub(crate) usage: fn(Counts) -> ChatCompletionsUsage,
}

type HttpError = litellm_host_http::Error<InferenceError>;
type Blocks = BoxStream<'static, Result<SseBlock, HttpError>>;

/// Tells the hooks about the answer of a route that may stream, which `model_group` served. A
/// complete answer counts what `counted` reads from it, and a stream tells how it went once it
/// ends.
pub(crate) async fn answered<R, Head, K: CallHooks>(
    hooks: &Arc<K>,
    ran: CallOutput<R, Head, SseBlock, HttpError>,
    model_group: String,
    format: Format,
    counted: impl FnOnce(&R) -> Option<Counts>,
) -> Result<(CallOutput<R, Head, SseBlock, HttpError>, Told), Ended> {
    let (head, chunks) = match ran {
        CallOutput::Complete(body) => {
            let served = ServedCall {
                model_group,
                usage: counted(&body).map(format.usage),
            };
            let told = complete(&**hooks, served).await?;
            return Ok((CallOutput::Complete(body), told));
        }
        CallOutput::Stream { head, chunks } => (head, chunks),
    };
    let told = Told {
        added: hooks.response_headers().await,
        complete: None,
    };
    let hooks = hooks.clone();
    let chunks: Blocks = stream_outcome::watched(
        chunks,
        format.read,
        stream_outcome::broke,
        move |outcome: Outcome| {
            let (hooks, model_group) = (hooks.clone(), model_group.clone());
            async move {
                match outcome {
                    Ok(counts) => {
                        let usage = Some((format.usage)(counts));
                        hooks.succeeded(ServedCall { model_group, usage }).await;
                    }
                    Err(_) => hooks.failed().await,
                }
            }
        },
    );
    Ok((CallOutput::Stream { head, chunks }, told))
}
