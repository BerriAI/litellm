use std::io::ErrorKind;
use std::path::Path;

use crate::error::Repr;
use crate::step::Reply;

/// A file has no validator, so every read is a fresh 200
pub(crate) async fn fetch(path: &Path) -> Reply {
    match tokio::fs::read(path).await {
        Ok(body) => Reply::Response {
            status: 200,
            etag: None,
            retry_after: None,
            body: Some(body),
        },
        Err(error) if error.kind() == ErrorKind::NotFound => Reply::Response {
            status: 404,
            etag: None,
            retry_after: None,
            body: None,
        },
        Err(source) => Reply::Failed(
            Repr::File {
                path: path.to_owned(),
                source,
            }
            .into(),
        ),
    }
}
