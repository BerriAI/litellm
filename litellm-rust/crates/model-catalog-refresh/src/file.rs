use std::path::PathBuf;

use crate::{Fetch, FetchRequest, Reply};

/// A catalog in a local file, answered as a 200 with no `ETag`
#[derive(Clone, Debug)]
pub struct FileFetch {
    location: String,
    path: PathBuf,
}

impl FileFetch {
    pub fn new(location: String, path: PathBuf) -> Self {
        Self { location, path }
    }
}

impl Fetch for FileFetch {
    fn location(&self) -> &str {
        &self.location
    }

    async fn fetch(&self, _: &FetchRequest) -> Reply {
        match tokio::fs::read(&self.path).await {
            Ok(body) => Reply::Response {
                status: 200,
                etag: None,
                retry_after: None,
                body,
            },
            Err(error) => Reply::Refused {
                error: error.into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;

    use super::*;

    fn request() -> FetchRequest {
        FetchRequest {
            if_none_match: None,
            timeout: Duration::from_secs(5),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn reads_the_file_and_refuses_a_missing_one() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("map.json");
        std::fs::write(&path, b"{}").unwrap();
        let present = FileFetch::new("file://map".to_owned(), path.clone());
        assert!(matches!(
            present.fetch(&request()).await,
            Reply::Response { status: 200, ref body, .. } if body == b"{}"
        ));
        let missing = FileFetch::new(
            "file://missing".to_owned(),
            directory.path().join("missing.json"),
        );
        assert!(matches!(
            missing.fetch(&request()).await,
            Reply::Refused { .. }
        ));
    }
}
