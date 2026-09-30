use std::future::Future;

use crate::Error;

pub trait Fetch: Sync {
    fn get(&self, url: &str) -> impl Future<Output = Result<Vec<u8>, Error>> + Send;
}

pub struct HttpFetch {
    client: reqwest::Client,
    github_token: Option<String>,
}

impl HttpFetch {
    pub fn new(github_token: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            github_token,
        }
    }

    pub fn from_env() -> Self {
        Self::new(std::env::var("GITHUB_TOKEN").ok())
    }
}

impl Fetch for HttpFetch {
    async fn get(&self, url: &str) -> Result<Vec<u8>, Error> {
        let request = self
            .client
            .get(url)
            .header("user-agent", "litellm-testkit")
            .header("accept", "application/json, application/octet-stream");
        let request = match (
            &self.github_token,
            url.starts_with("https://api.github.com/"),
        ) {
            (Some(token), true) => request.bearer_auth(token),
            _ => request,
        };
        let request_error = |source| Error::Request {
            url: url.to_owned(),
            source,
        };
        let response = request.send().await.map_err(request_error)?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Status {
                url: url.to_owned(),
                status: status.as_u16(),
            });
        }
        Ok(response.bytes().await.map_err(request_error)?.to_vec())
    }
}
