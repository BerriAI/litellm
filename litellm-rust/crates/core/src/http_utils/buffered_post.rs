use std::sync::OnceLock;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::constants::BUFFERED_POST_CONNECT_TIMEOUT_SECS;
use crate::error::Error;

pub struct Request {
    pub url: String,
    pub headers: Vec<(Vec<u8>, Vec<u8>)>,
    pub body: Vec<u8>,
    pub timeout_seconds: f64,
}

pub struct Response {
    pub status: u16,
    pub headers: Vec<(Vec<u8>, Vec<u8>)>,
    pub content: Vec<u8>,
}

pub async fn send(request: Request) -> Result<Response, Error> {
    let timeout = Duration::try_from_secs_f64(request.timeout_seconds)
        .ok()
        .filter(|timeout| !timeout.is_zero())
        .ok_or_else(|| Error::InvalidRequest("timeout must be positive and finite".into()))?;
    let mut headers = HeaderMap::new();
    for (name, value) in request.headers {
        let name = HeaderName::from_bytes(&name)
            .map_err(|_| Error::InvalidRequest("invalid header name".into()))?;
        let value = HeaderValue::from_bytes(&value)
            .map_err(|_| Error::InvalidRequest("invalid header value".into()))?;
        headers.append(name, value);
    }

    static CLIENT: OnceLock<Result<reqwest::Client, reqwest::Error>> = OnceLock::new();
    let client = CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(BUFFERED_POST_CONNECT_TIMEOUT_SECS))
                .redirect(reqwest::redirect::Policy::none())
                .no_gzip()
                .no_brotli()
                .no_deflate()
                .no_zstd()
                .build()
        })
        .as_ref()
        .map_err(|_| Error::Network("could not initialize HTTP client".into()))?;
    let response = client
        .post(request.url)
        .headers(headers)
        .body(request.body)
        .timeout(timeout)
        .send()
        .await
        .map_err(|_| Error::Network("transport failed".into()))?;
    let status = response.status().as_u16();
    let headers = response
        .headers()
        .iter()
        .map(|(name, value)| (name.as_str().as_bytes().to_vec(), value.as_bytes().to_vec()))
        .collect();
    let content = response
        .bytes()
        .await
        .map_err(|_| Error::Network("could not read response".into()))?
        .to_vec();
    Ok(Response {
        status,
        headers,
        content,
    })
}

#[cfg(test)]
mod tests;
