use std::time::Duration;

use litellm_model_catalog::{Catalog, Located};

use crate::source::Source;

/// One attempt. Where to fetch from is the `Source`, so the request carries no URL
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchRequest {
    pub if_none_match: Option<String>,
    pub timeout: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// An HTTP, S3 or file answer. The body is read only on 200
    Response {
        status: u16,
        etag: Option<String>,
        retry_after: Option<Duration>,
        body: Option<Vec<u8>>,
    },
    /// No answer reached us
    Transport { reason: String },
}

/// What the fetch state machine asks its driver to do next
#[derive(Debug)]
pub enum Step {
    Fetch(FetchRequest),
    Sleep(Duration),
    /// A parsed document to install, with the row errors decision 15 rejects
    Adopt {
        catalog: Box<Catalog>,
        installed: Installed,
        errors: Vec<Located>,
    },
    /// 304, a final status, or retries exhausted
    Done {
        reason: String,
    },
}

/// What is installed now, the input to the next revalidation
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Installed {
    pub origin: Origin,
    pub etag: Option<String>,
    /// `git_blob_id` of the installed bytes
    pub revision: String,
    pub model_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    Bundled,
    Remote { source: Source },
}
