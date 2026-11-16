//! Admission: everything the gateway settles about a request before any handler runs.
//!
//! [`Routed::plan`] is the only caller of a route's `plan`, so admission is the only place a
//! request declines. A handler takes what `plan` made as a [`Planned`] and has no
//! body left to read and nothing left to decline.

use std::sync::Arc;

use axum::{
    Json,
    body::{Body, Bytes},
    extract::{FromRequestParts, MatchedPath, Request, State},
    http::{Extensions, request::Parts},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{MethodRouter, post},
};
use futures_util::{StreamExt, stream};
use http_body_util::BodyExt;
use litellm_core_utils::decline::Decline;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_inference::Unplanned;
use parking_lot::Mutex;

use crate::{
    BodyError, Error, Gateway, audio_transcription, chat_completions, messages, ocr,
    request::{MAX_BODY_BYTES, RequestId},
    responses,
};

const UNSERVED: Decline = Decline::new("route", "a path the Rust gateway does not implement");
fn unserved(path: &str) -> Decline {
    UNSERVED.with_detail(format!("{path} is not implemented by the Rust gateway"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    Messages,
    Ocr,
    ChatCompletions,
    ChatCompletionsForModel,
    AudioTranscription,
    Responses,
    Unserved,
}

/// Every path the gateway mounts. The router and admission both read this table and nothing else
/// lists a native route.
const ROUTES: &[(&str, Route)] = &[
    ("/v1/messages", Route::Messages),
    ("/ocr", Route::Ocr),
    ("/v1/ocr", Route::Ocr),
    ("/chat/completions", Route::ChatCompletions),
    ("/v1/chat/completions", Route::ChatCompletions),
    (
        "/engines/{model}/chat/completions",
        Route::ChatCompletionsForModel,
    ),
    (
        "/openai/deployments/{model}/chat/completions",
        Route::ChatCompletionsForModel,
    ),
    ("/audio/transcriptions", Route::AudioTranscription),
    ("/v1/audio/transcriptions", Route::AudioTranscription),
    ("/responses", Route::Responses),
    ("/v1/responses", Route::Responses),
    ("/embeddings", Route::Unserved),
    ("/v1/embeddings", Route::Unserved),
    ("/completions", Route::Unserved),
    ("/v1/completions", Route::Unserved),
    ("/engines/{model}/embeddings", Route::Unserved),
    ("/engines/{model}/completions", Route::Unserved),
    ("/openai/deployments/{model}/embeddings", Route::Unserved),
    ("/openai/deployments/{model}/completions", Route::Unserved),
];

impl Route {
    fn matched(extensions: &Extensions) -> Option<Self> {
        let matched = extensions.get::<MatchedPath>()?.as_str();
        ROUTES
            .iter()
            .find_map(|(path, route)| (*path == matched).then_some(*route))
    }

    fn handler(self) -> MethodRouter<Arc<Gateway>> {
        match self {
            Self::Messages => post(messages::create),
            Self::Ocr => post(ocr::create),
            Self::ChatCompletions | Self::ChatCompletionsForModel => post(chat_completions::create),
            Self::AudioTranscription => post(audio_transcription::create),
            Self::Responses => post(responses::create),
            Self::Unserved => post(|| async { Error::Internal }),
        }
    }

    /// The decline that needs no body: a path nothing serves. Found before a byte is read.
    fn unread(self, path: &str) -> Option<Decline> {
        match self {
            Self::Unserved => Some(unserved(path)),
            Self::Messages
            | Self::Ocr
            | Self::ChatCompletions
            | Self::ChatCompletionsForModel
            | Self::AudioTranscription
            | Self::Responses => None,
        }
    }

    async fn plan(
        self,
        gateway: &Gateway,
        identity: &AuthenticatedRequest,
        parts: &mut Parts,
        body: Bytes,
    ) -> Result<(), Refusal> {
        match self {
            Self::Messages => {
                let call = messages::plan(gateway, identity, parts, &body).await?;
                hand(parts, call);
            }
            Self::ChatCompletions => {
                let call = chat_completions::plan(gateway, identity, None, &body).await?;
                hand(parts, call);
            }
            Self::ChatCompletionsForModel => {
                let crate::request::Path(model) =
                    crate::request::Path::<String>::from_request_parts(parts, &()).await?;
                let call = chat_completions::plan(gateway, identity, Some(model), &body).await?;
                hand(parts, call);
            }
            Self::Responses => {
                let call = responses::plan(gateway, identity, &body).await?;
                hand(parts, call);
            }
            Self::Ocr => {
                let call = ocr::plan(gateway, identity, parts, body).await?;
                hand(parts, call);
            }
            Self::AudioTranscription => {
                let call = audio_transcription::plan(gateway, identity, parts, body).await?;
                hand(parts, call);
            }
            Self::Unserved => return Err(unserved(parts.uri.path()).into()),
        }
        Ok(())
    }
}

/// Why the gateway does not serve a request.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Refusal {
    /// There is no Rust way to serve it.
    #[error(transparent)]
    Declined(#[from] Decline),
    /// The gateway has its own answer for it.
    #[error(transparent)]
    Rejected(#[from] Error),
}

impl From<BodyError> for Refusal {
    fn from(error: BodyError) -> Self {
        Self::Rejected(error.into())
    }
}

impl From<Unplanned> for Refusal {
    fn from(unplanned: Unplanned) -> Self {
        match unplanned {
            Unplanned::Declined(decline) => Self::Declined(decline),
            Unplanned::Rejected(error) => Self::Rejected(Error::Route(error)),
        }
    }
}

impl Refusal {
    /// The gateway's own answer: the error it found, or a 501 naming the decline's code, since
    /// nothing is behind the gateway to serve a declined request instead.
    fn answer(self, route: Option<Route>, request_id: Option<&str>) -> Response {
        let error = match self {
            Self::Declined(decline) => Error::NotImplemented(decline),
            Self::Rejected(error) => error,
        };
        match route {
            Some(Route::Messages) => (error.status(), Json(error.body(request_id))).into_response(),
            _ => error.into_response(),
        }
    }
}

/// A request on a route that serves it, waiting for its caller. Its body is still unread.
struct Routed {
    route: Route,
    parts: Parts,
    body: Body,
}

impl Routed {
    /// Matches a request on one of [`routes`] to its route. A request no handler would serve
    /// comes back untouched, with why.
    fn of(request: Request) -> Result<Self, (Box<Request>, Refusal)> {
        let (parts, body) = request.into_parts();
        let Some(route) = Route::matched(&parts.extensions) else {
            return Err(handed_back(parts, body, UNSERVED));
        };
        match route.unread(parts.uri.path()) {
            Some(decline) => Err(handed_back(parts, body, decline)),
            None => Ok(Self { route, parts, body }),
        }
    }

    /// Plans the call for `identity`: reads the body up to the native cap, parses it into the
    /// route's request, resolves and authorizes the model, and asks the route core for its plan.
    /// Nothing here writes. The request that comes out carries the caller and the plan for its
    /// handler. One that does not plan comes back as the client sent it, and a body over the cap
    /// has the bytes already read chained back on, so nothing is buffered without a bound.
    async fn plan(
        self,
        gateway: &Gateway,
        identity: AuthenticatedRequest,
    ) -> Result<Request, (Box<Request>, Refusal)> {
        let Self {
            route,
            mut parts,
            body,
        } = self;
        let body = match buffered(body).await {
            Ok(body) => body,
            Err(Unbuffered::TooLarge(body)) => {
                return Err(handed_back(parts, body, Error::BodyTooLarge));
            }
            Err(Unbuffered::Unreadable(body, error)) => {
                return Err(handed_back(parts, body, BodyError::Unreadable(error)));
            }
        };
        match route
            .plan(gateway, &identity, &mut parts, body.clone())
            .await
        {
            Ok(()) => {
                parts.extensions.insert(identity);
                Ok(Request::from_parts(parts, Body::empty()))
            }
            Err(refusal) => Err(handed_back(parts, Body::from(body), refusal)),
        }
    }
}

fn handed_back(parts: Parts, body: Body, refusal: impl Into<Refusal>) -> (Box<Request>, Refusal) {
    (Box::new(Request::from_parts(parts, body)), refusal.into())
}

enum Unbuffered {
    TooLarge(Body),
    Unreadable(Body, axum::Error),
}

/// The whole body, or the body put back together once it turns out not to fit.
async fn buffered(mut body: Body) -> Result<Bytes, Unbuffered> {
    let mut read: Vec<Bytes> = Vec::new();
    let mut length = 0;
    loop {
        match body.frame().await {
            None => return Ok(read.concat().into()),
            Some(Err(error)) => {
                return Err(Unbuffered::Unreadable(replayed(read, Body::empty()), error));
            }
            Some(Ok(frame)) => {
                let Ok(data) = frame.into_data() else {
                    continue;
                };
                length += data.len();
                read.push(data);
                if length > MAX_BODY_BYTES {
                    return Err(Unbuffered::TooLarge(replayed(read, body)));
                }
            }
        }
    }
}

fn replayed(read: Vec<Bytes>, rest: Body) -> Body {
    Body::from_stream(
        stream::iter(read)
            .map(Ok::<_, axum::Error>)
            .chain(rest.into_data_stream()),
    )
}

/// What a route's `plan` made of a request, taken by its handler exactly once.
pub(crate) struct Planned<T>(pub T);

struct Handed<T>(Arc<Mutex<Option<T>>>);

impl<T> Clone for Handed<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

fn hand<T: Send + Sync + 'static>(parts: &mut Parts, planned: T) {
    parts
        .extensions
        .insert(Handed(Arc::new(Mutex::new(Some(planned)))));
}

impl<S: Send + Sync, T: Send + Sync + 'static> FromRequestParts<S> for Planned<T> {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Error> {
        parts
            .extensions
            .get::<Handed<T>>()
            .and_then(|handed| handed.0.lock().take())
            .map(Self)
            .ok_or(Error::Internal)
    }
}

/// The inference routes for requests [`Routed::plan`] let in. A handler takes
/// the plan from the request and fails without one.
fn routes(gateway: Arc<Gateway>) -> axum::Router {
    ROUTES
        .iter()
        .fold(axum::Router::new(), |router, (path, route)| {
            router.route(path, route.handler())
        })
        .with_state(gateway)
}

/// The inference routes with their own admission, for a server with nothing behind it: a request
/// the gateway does not serve gets the gateway's answer, a 501 naming the decline's code when
/// there is no Rust way to serve it. The mounting server authenticates the caller first.
pub fn router(gateway: Arc<Gateway>) -> axum::Router {
    routes(Arc::clone(&gateway)).route_layer(axum::middleware::from_fn_with_state(gateway, admit))
}

async fn admit(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    RequestId(request_id): RequestId,
    request: Request,
    next: Next,
) -> Response {
    let planned = match Routed::of(request) {
        Ok(routed) => routed.plan(&gateway, identity).await,
        Err(refused) => Err(refused),
    };
    match planned {
        Ok(request) => next.run(request).await,
        Err((request, refusal)) => answered(&request, refusal, request_id.as_deref()),
    }
}

fn answered(request: &Request, refusal: Refusal, request_id: Option<&str>) -> Response {
    refusal.answer(Route::matched(request.extensions()), request_id)
}
