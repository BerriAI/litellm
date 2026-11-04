use std::time::Duration;

use crate::Error;

/// One attempt. Where to fetch from is the `Fetch`, so the request carries no URL
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchRequest {
    pub if_none_match: Option<String>,
    pub timeout: Duration,
}

#[derive(Debug)]
pub enum Reply {
    /// An HTTP, S3 or file answer
    Response {
        status: u16,
        /// Every `ETag` value, joined with `", "`
        etag: Option<String>,
        /// Every `Retry-After` value, joined with `", "`
        retry_after: Option<String>,
        body: Vec<u8>,
    },
    /// No answer reached us, and another attempt may get one
    Transport { error: Error },
    /// The request itself cannot succeed, so retrying is pointless
    Refused { error: Error },
}

/// One source of the catalog's bytes
pub trait Fetch: Sync {
    /// The source as the caller wrote it, for failure reasons
    fn location(&self) -> &str;
    fn fetch(&self, request: &FetchRequest) -> impl Future<Output = Reply> + Send;
}
