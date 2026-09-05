use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::{Client, ClientBuilder, Method, Request, RequestBuilder, Response, Url};

use super::{AuthError, AuthErrorKind, AuthFuture};

pub struct AuthPreparation {
    pub authorizer: Arc<dyn RequestAuthorizer>,
    pub headers: reqwest::header::HeaderMap,
    pub remove_headers: Vec<reqwest::header::HeaderName>,
}

pub trait RequestAuthorizer: Send + Sync {
    fn prepare(&self) -> AuthFuture<'_, Option<AuthPreparation>> {
        Box::pin(async { Ok(None) })
    }

    fn authorize(&self, request: Request) -> AuthFuture<'_, Request>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayPolicy {
    Never,
    SameOrigin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthOperation {
    Initial,
    FollowUp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Origin {
    scheme: String,
    host: String,
    port: u16,
}

impl Origin {
    fn from_url(url: &Url) -> Result<Self, AuthError> {
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(forbidden_destination());
        }
        Ok(Self {
            scheme: url.scheme().to_owned(),
            host: url.host_str().ok_or_else(forbidden_destination)?.to_owned(),
            port: url
                .port_or_known_default()
                .ok_or_else(forbidden_destination)?,
        })
    }
}

pub struct AuthSession {
    authorizer: Arc<dyn RequestAuthorizer>,
    origin: Origin,
    replay: ReplayPolicy,
}

impl AuthSession {
    pub fn new(
        authorizer: Arc<dyn RequestAuthorizer>,
        url: &Url,
        replay: ReplayPolicy,
    ) -> Result<Self, AuthError> {
        Ok(Self {
            authorizer,
            origin: Origin::from_url(url)?,
            replay,
        })
    }

    pub fn check_destination(&self, url: &Url, operation: AuthOperation) -> Result<(), AuthError> {
        if self.origin != Origin::from_url(url)?
            || (operation == AuthOperation::FollowUp && self.replay == ReplayPolicy::Never)
        {
            return Err(forbidden_destination());
        }
        Ok(())
    }

    pub(crate) async fn prepare(
        &self,
        url: &Url,
        operation: AuthOperation,
        deadline: Instant,
    ) -> Result<
        (
            Self,
            reqwest::header::HeaderMap,
            Vec<reqwest::header::HeaderName>,
        ),
        AuthError,
    > {
        self.check_destination(url, operation)?;
        if deadline <= Instant::now() {
            return Err(deadline_error());
        }
        let prepared = tokio::time::timeout_at(deadline.into(), self.authorizer.prepare())
            .await
            .map_err(|_| deadline_error())??;
        let (authorizer, headers, remove_headers) = match prepared {
            Some(prepared) => (
                prepared.authorizer,
                prepared.headers,
                prepared.remove_headers,
            ),
            None => (
                self.authorizer.clone(),
                reqwest::header::HeaderMap::new(),
                Vec::new(),
            ),
        };
        Ok((
            Self {
                authorizer,
                origin: self.origin.clone(),
                replay: self.replay,
            },
            headers,
            remove_headers,
        ))
    }

    async fn authorize(
        &self,
        request: Request,
        operation: AuthOperation,
    ) -> Result<Request, AuthError> {
        self.check_destination(request.url(), operation)?;
        let request = self.authorizer.authorize(request).await?;
        self.check_destination(request.url(), operation)?;
        Ok(request)
    }
}

#[derive(Clone)]
pub struct AuthHttpClient(Client);

impl AuthHttpClient {
    pub fn new(
        builder: ClientBuilder,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<Self, crate::Error> {
        if connect_timeout.is_zero() || request_timeout.is_zero() {
            return Err(crate::Error::InvalidRequest(
                "HTTP timeouts must be positive".into(),
            ));
        }
        builder
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(connect_timeout)
            .timeout(request_timeout)
            .build()
            .map(Self)
            .map_err(|_| crate::Error::Network("failed to build auth HTTP client".into()))
    }

    pub fn request(&self, method: Method, url: Url) -> RequestBuilder {
        self.0.request(method, url)
    }

    pub async fn send(
        &self,
        session: &AuthSession,
        request: Request,
        operation: AuthOperation,
        deadline: Instant,
    ) -> Result<Response, crate::Error> {
        let (session, _, _) = session.prepare(request.url(), operation, deadline).await?;
        self.send_prepared(&session, request, operation, deadline)
            .await
    }

    pub(crate) async fn send_prepared(
        &self,
        session: &AuthSession,
        request: Request,
        operation: AuthOperation,
        deadline: Instant,
    ) -> Result<Response, crate::Error> {
        if deadline <= Instant::now() {
            return Err(deadline_error().into());
        }
        let work = async {
            let mut request = session.authorize(request, operation).await?;
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .ok_or_else(deadline_error)?;
            let timeout = request
                .timeout()
                .copied()
                .map_or(remaining, |timeout| timeout.min(remaining));
            *request.timeout_mut() = Some(timeout);
            self.0.execute(request).await.map_err(|error| {
                crate::Error::Network(
                    if error.is_timeout() {
                        "Request timed out"
                    } else {
                        "authenticated request failed"
                    }
                    .into(),
                )
            })
        };
        tokio::time::timeout_at(deadline.into(), work)
            .await
            .map_err(|_| {
                AuthError::new(
                    AuthErrorKind::DeadlineExceeded,
                    "auth_deadline_exceeded",
                    "Request timed out",
                )
            })?
    }
}

fn forbidden_destination() -> AuthError {
    AuthError::new(
        AuthErrorKind::ForbiddenDestination,
        "forbidden_auth_destination",
        "refusing to send provider credentials to this destination",
    )
}

fn deadline_error() -> AuthError {
    AuthError::new(
        AuthErrorKind::DeadlineExceeded,
        "auth_deadline_exceeded",
        "Request timed out",
    )
}
