use bytes::Bytes;
use futures_util::{future::BoxFuture, stream::BoxStream};
use serde_json::Value;

use crate::Error;

pub enum TransportBody {
    Json(Value),
    Signed(Bytes),
}

pub struct TransportRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: TransportBody,
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

pub trait Transport: Send + Sync {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'static, Result<TransportResponse, Error>>;
}
