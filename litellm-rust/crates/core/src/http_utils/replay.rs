use std::error::Error as _;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Method, Response, StatusCode, Url};

use crate::Error;
use crate::constants::{
    JSON_BODY_MAX_REDIRECTS, JSON_BODY_PROTOCOL_RETRIES, MESSAGES_CONNECT_TIMEOUT_SECS,
};

use super::body::PreparedJsonBody;

pub type BodySigner<'a> =
    dyn Fn(&Url, &str, &HeaderMap) -> Result<HeaderMap, Error> + Send + Sync + 'a;

pub fn replay_client() -> Result<&'static Client, Error> {
    static CLIENT: OnceLock<Result<Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            Client::builder()
                .connect_timeout(Duration::from_secs(MESSAGES_CONNECT_TIMEOUT_SECS))
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| Error::Network(error.clone()))
}

pub async fn send_json(
    client: &Client,
    url: &str,
    body: &PreparedJsonBody,
    headers: &[(String, String)],
    timeout: Duration,
    signer: Option<&BodySigner<'_>>,
) -> Result<Response, Error> {
    let started = Instant::now();
    let initial_url =
        Url::parse(url).map_err(|_| Error::InvalidRequest("invalid provider URL".into()))?;
    let digest = signer.map(|_| body.sha256());
    let headers = headers
        .iter()
        .map(|(key, value)| {
            let key = HeaderName::from_bytes(key.as_bytes())
                .map_err(|_| Error::InvalidRequest("invalid request header name".into()))?;
            let value = HeaderValue::from_str(value)
                .map_err(|_| Error::InvalidRequest("invalid request header value".into()))?;
            Ok((key, value))
        })
        .collect::<Result<HeaderMap, Error>>()?;
    if !body.is_streamed() {
        let headers = signed_headers(signer, &initial_url, digest.as_deref(), &headers)?;
        return client
            .post(initial_url)
            .headers(headers)
            .body(request_body(body))
            .timeout(remaining(started, timeout)?)
            .send()
            .await
            .map_err(network_error);
    }
    let mut url = initial_url;
    let mut headers = headers;
    let mut signing = signer;
    let mut method = Method::POST;
    let mut redirects = 0;
    let mut retries = 0;
    loop {
        let mut attempt_headers = headers.clone();
        let request = if method == Method::POST {
            attempt_headers.remove(reqwest::header::TRANSFER_ENCODING);
            attempt_headers.insert(CONTENT_LENGTH, HeaderValue::from(body.content_length()));
            attempt_headers
                .entry(CONTENT_TYPE)
                .or_insert(HeaderValue::from_static("application/json"));
            client
                .request(method.clone(), url.clone())
                .body(request_body(body))
        } else {
            client.request(method.clone(), url.clone())
        };
        let attempt_headers = if method == Method::POST {
            signed_headers(signing, &url, digest.as_deref(), &attempt_headers)?
        } else {
            attempt_headers
        };
        let response = request
            .headers(attempt_headers)
            .timeout(remaining(started, timeout)?)
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(error)
                if retries < JSON_BODY_PROTOCOL_RETRIES && retryable_protocol_error(&error) =>
            {
                retries += 1;
                continue;
            }
            Err(error) => return Err(network_error(error)),
        };
        let status = response.status();
        if !matches!(status.as_u16(), 301 | 302 | 303 | 307 | 308) {
            return Ok(response);
        }
        let Some(location) = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
        else {
            return Ok(response);
        };
        let Ok(next_url) = url.join(location) else {
            return Ok(response);
        };
        if !matches!(next_url.scheme(), "http" | "https") {
            return Err(Error::Network("unsupported redirect scheme".into()));
        }
        if redirects >= JSON_BODY_MAX_REDIRECTS {
            return Err(Error::Network("too many provider redirects".into()));
        }
        redirects += 1;
        if matches!(
            status,
            StatusCode::MOVED_PERMANENTLY | StatusCode::FOUND | StatusCode::SEE_OTHER
        ) {
            method = Method::GET;
            for name in [
                "content-type",
                "content-length",
                "content-encoding",
                "transfer-encoding",
            ] {
                headers.remove(name);
            }
        }
        if url.origin() != next_url.origin() {
            for name in [
                "authorization",
                "proxy-authorization",
                "cookie",
                "www-authenticate",
                "x-api-key",
                "api-key",
                "x-amz-security-token",
                "x-amz-date",
                "x-amz-content-sha256",
                "ocp-apim-subscription-key",
            ] {
                headers.remove(name);
            }
            signing = None;
        }
        headers.remove(reqwest::header::HOST);
        if !(url.scheme() == "https" && next_url.scheme() == "http") {
            let mut referer = url.clone();
            let _ = referer.set_username("");
            let _ = referer.set_password(None);
            referer.set_fragment(None);
            if let Ok(value) = HeaderValue::from_str(referer.as_str()) {
                headers.insert(reqwest::header::REFERER, value);
            }
        } else {
            headers.remove(reqwest::header::REFERER);
        }
        url = next_url;
    }
}

fn signed_headers(
    signer: Option<&BodySigner<'_>>,
    url: &Url,
    digest: Option<&str>,
    headers: &HeaderMap,
) -> Result<HeaderMap, Error> {
    match (signer, digest) {
        (Some(signer), Some(digest)) => signer(url, digest, headers),
        _ => Ok(headers.clone()),
    }
}

fn remaining(started: Instant, timeout: Duration) -> Result<Duration, Error> {
    timeout
        .checked_sub(started.elapsed())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| Error::Network("request timed out".into()))
}

fn network_error(error: reqwest::Error) -> Error {
    if error.is_connect() || error.is_builder() {
        Error::Connect(error.to_string())
    } else {
        Error::Network(error.to_string())
    }
}

fn retryable_protocol_error(error: &reqwest::Error) -> bool {
    let mut source = error.source();
    while let Some(error) = source {
        if let Some(error) = error.downcast_ref::<h2::Error>() {
            return error.is_remote()
                && ((error.is_go_away() && error.reason() == Some(h2::Reason::NO_ERROR))
                    || (error.is_reset() && error.reason() == Some(h2::Reason::REFUSED_STREAM)));
        }
        source = error.source();
    }
    false
}

#[cfg(test)]
#[path = "replay_tests.rs"]
mod tests;

fn request_body(body: &PreparedJsonBody) -> reqwest::Body {
    if let Some(bytes) = body.buffered_bytes() {
        return bytes.into();
    }
    reqwest::Body::wrap_stream(futures_util::stream::iter(
        body.chunks().map(Ok::<_, std::io::Error>),
    ))
}
