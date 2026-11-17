use bytes::Bytes;
use futures_util::{future::BoxFuture, stream::BoxStream};

use crate::Error;

pub struct TransportRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
    pub stream: bool,
}

pub enum Delivered {
    Complete(String),
    Stream(BoxStream<'static, Result<Bytes, Error>>),
    Rejected { text: String, error: Error },
}

pub struct TransportResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub delivered: Delivered,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnencodableHeader {
    Sent(Vec<u8>),
    Received(Vec<u8>),
}

impl std::fmt::Display for UnencodableHeader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a header value is not ASCII, which Python's HTTP client cannot encode")
    }
}

impl std::error::Error for UnencodableHeader {}

impl UnencodableHeader {
    pub(crate) fn received(headers: &reqwest::header::HeaderMap) -> Option<Self> {
        headers
            .values()
            .find(|value| !value.as_bytes().is_ascii())
            .map(|value| Self::Received(value.as_bytes().to_vec()))
    }
}

pub trait Transport: Send + Sync {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'static, Result<TransportResponse, Error>>;
}
