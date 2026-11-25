use std::time::{Duration, SystemTime};

use bytes::Bytes;
use litellm_auth_aws::{Credentials, SigningScope, sign_s3_headers};
use reqwest::Method;

pub struct SignedRequest<'a> {
    pub method: Method,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub body: Bytes,
    pub region: &'a str,
    pub credentials: &'a Credentials,
    pub time: SystemTime,
    pub read_timeout: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    pub status: u16,
    pub body: Bytes,
}

impl Reply {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SendError {
    #[error("the request cannot be signed: {0}")]
    Signing(#[from] litellm_auth_aws::Error),
    #[error("the request was not sent: {0}")]
    Request(String),
    #[error("{kind}: {message}")]
    Transport { kind: &'static str, message: String },
}

impl SendError {
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::Transport { .. })
    }

    pub fn is_reconnectable(&self) -> bool {
        matches!(
            self,
            Self::Transport {
                kind: "ConnectError" | "RemoteProtocolError",
                ..
            }
        )
    }
}

pub fn signature_headers(request: &SignedRequest<'_>) -> Result<Vec<(String, String)>, SendError> {
    Ok(sign_s3_headers(
        request.method.as_str(),
        request.url,
        &request.body,
        request.headers,
        SigningScope {
            region: request.region,
            service: "s3",
            credentials: request.credentials,
            time: request.time,
        },
    )?)
}

pub async fn send(
    client: &litellm_http::Client,
    request: SignedRequest<'_>,
) -> Result<Reply, SendError> {
    let signature = signature_headers(&request)?;
    let mut builder = client.request(request.method.clone(), request.url);
    for (name, value) in request.headers.iter().chain(&signature) {
        builder = builder.header(name, value);
    }
    if !request.body.is_empty() || request.method == Method::PUT {
        builder = builder.body(request.body.clone());
    }
    let mut response = timed(request.read_timeout, builder.send()).await?;
    let status = response.status().as_u16();
    let mut body = Vec::new();
    while let Some(chunk) = timed(request.read_timeout, response.chunk()).await? {
        body.extend_from_slice(&chunk);
    }
    Ok(Reply {
        status,
        body: Bytes::from(body),
    })
}

async fn timed<T>(
    limit: Option<Duration>,
    pending: impl Future<Output = Result<T, reqwest::Error>>,
) -> Result<T, SendError> {
    let finished = match limit {
        Some(limit) => tokio::time::timeout(limit, pending)
            .await
            .map_err(|elapsed| SendError::Transport {
                kind: "ReadTimeout",
                message: elapsed.to_string(),
            })?,
        None => pending.await,
    };
    finished.map_err(transport)
}

fn transport(error: reqwest::Error) -> SendError {
    let message = error.to_string();
    if error.is_builder() || error.is_redirect() {
        return SendError::Request(message);
    }
    let kind = if error.is_timeout() {
        "Timeout"
    } else if error.is_connect() {
        "ConnectError"
    } else if remote_protocol(&error) {
        "RemoteProtocolError"
    } else {
        "TransportError"
    };
    SendError::Transport { kind, message }
}

fn remote_protocol(error: &reqwest::Error) -> bool {
    let mut cause = std::error::Error::source(error);
    while let Some(current) = cause {
        if let Some(hyper) = current.downcast_ref::<hyper::Error>() {
            return hyper.is_incomplete_message() || hyper.is_parse();
        }
        cause = current.source();
    }
    false
}
