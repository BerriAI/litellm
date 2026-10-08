use crate::{Error, wire};
use http::Method;
use litellm_http::Client;
use serde::{Serialize, de::DeserializeOwned};
use std::{sync::Arc, time::Duration};
use tokio::sync::Semaphore;
use url::Url;

const MAX_RESPONSE: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub struct Control {
    client: Client,
    base: Url,
    token: Arc<str>,
    model_slots: Arc<Semaphore>,
    attempt: Option<u64>,
}

impl Control {
    pub fn new(client: Client, mut base: Url, token: String) -> Self {
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path()));
        }
        Self {
            client,
            base,
            token: token.into(),
            model_slots: Arc::new(Semaphore::new(16)),
            attempt: None,
        }
    }

    pub fn url(&self, path: &str) -> Result<Url, Error> {
        self.base
            .join(path.trim_start_matches('/'))
            .map_err(|_| Error::InvalidRequest)
    }

    pub async fn request<T: DeserializeOwned>(
        &self,
        method: Method,
        url: Url,
        body: Option<&impl Serialize>,
        timeout: Duration,
    ) -> Result<T, Error> {
        let is_model = url.path().ends_with("/model");
        let request = self
            .client
            .request(method, url)
            .bearer_auth(&*self.token)
            .timeout(timeout);
        let request = match body {
            Some(body) => request.json(body),
            None => request,
        };
        let request = match self.attempt {
            Some(attempt) => request.header("x-litellm-lens-attempt", attempt),
            None => request,
        };
        let mut response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok());
            let diagnostic = if is_model {
                model_diagnostic(&mut response).await
            } else {
                None
            };
            return Err(Error::Control {
                status: status.as_u16(),
                retry_after,
                diagnostic,
            });
        }
        let finish_reason = response
            .headers()
            .get("x-litellm-lens-finish-reason")
            .cloned();
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if body.len().saturating_add(chunk.len()) > MAX_RESPONSE {
                return Err(Error::TooLarge);
            }
            body.extend_from_slice(&chunk);
        }
        if body.is_empty() {
            body.extend_from_slice(b"null");
        }
        let mut value: serde_json::Value = serde_json::from_slice(&body)?;
        if let Some(reason) = finish_reason.and_then(|v| v.to_str().ok().map(str::to_owned))
            && matches!(reason.as_str(), "length" | "content_filter")
            && let Some(object) = value.as_object_mut()
        {
            object.insert("finish_reason".into(), reason.into());
        }
        Ok(serde_json::from_value(value)?)
    }

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, Error> {
        self.request(
            Method::GET,
            self.url(path)?,
            None::<&()>,
            Duration::from_secs(180),
        )
        .await
    }

    pub async fn post<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, Error> {
        self.request(
            Method::POST,
            self.url(path)?,
            Some(body),
            Duration::from_secs(180),
        )
        .await
    }
}

async fn model_diagnostic(response: &mut reqwest::Response) -> Option<String> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.ok()? {
        if body.len().saturating_add(chunk.len()) > 16 * 1024 {
            return None;
        }
        body.extend_from_slice(&chunk);
    }
    let value: serde_json::Value = serde_json::from_slice(&body).ok()?;
    let diagnostic = value.pointer("/detail/lens_error")?.as_str()?;
    (diagnostic.len() <= 4096).then(|| diagnostic.to_owned())
}

#[derive(Clone)]
pub struct JobClient {
    pub control: Control,
    prefix: String,
    model_slots: Arc<Semaphore>,
}

impl JobClient {
    pub fn with_attempt(mut self, attempt: u64) -> Self {
        self.control.attempt = Some(attempt);
        self
    }

    pub fn new(
        control: Control,
        lens_id: &str,
        job_id: &str,
        concurrency: usize,
    ) -> Result<Self, Error> {
        if [lens_id, job_id].iter().any(|id| {
            id.is_empty()
                || !id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        }) {
            return Err(Error::InvalidRequest);
        }
        Ok(Self {
            control,
            prefix: format!("lens/worker/{lens_id}/{job_id}"),
            model_slots: Arc::new(Semaphore::new(concurrency.clamp(1, 16))),
        })
    }

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, Error> {
        self.control.get(&format!("{}/{path}", self.prefix)).await
    }

    pub async fn post<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, Error> {
        self.control
            .post(&format!("{}/{path}", self.prefix), body)
            .await
    }

    pub async fn content(
        &self,
        execution_id: &str,
        cursor: &str,
        offset: usize,
    ) -> Result<wire::ExecutionContent, Error> {
        let mut url = self.control.url(&format!("{}/content", self.prefix))?;
        url.query_pairs_mut()
            .append_pair("execution_id", execution_id)
            .append_pair("cursor", cursor)
            .append_pair("offset", &offset.to_string());
        self.control
            .request(Method::GET, url, None::<&()>, Duration::from_secs(180))
            .await
    }

    pub async fn model(&self, body: &wire::ModelRequest) -> Result<wire::ModelResult, Error> {
        let _permit = self
            .model_slots
            .acquire()
            .await
            .map_err(|_| Error::Unavailable)?;
        let url = self.control.url(&format!("{}/model", self.prefix))?;
        let _global_permit = self
            .control
            .model_slots
            .acquire()
            .await
            .map_err(|_| Error::Unavailable)?;
        for attempt in 0..=4 {
            let result = self
                .control
                .request(
                    Method::POST,
                    url.clone(),
                    Some(body),
                    Duration::from_secs(1800),
                )
                .await;
            match result {
                Err(ref error) if error.retryable() && attempt < 4 => {
                    let requested = match error {
                        Error::Control { retry_after, .. } => retry_after.unwrap_or_default(),
                        _ => 0,
                    };
                    tokio::time::sleep(Duration::from_secs(requested.max(1 << attempt).min(60)))
                        .await;
                }
                result => return result,
            }
        }
        Err(Error::Unavailable)
    }

    pub async fn progress(&self, progress: &wire::Progress) -> Result<(), Error> {
        let _: serde_json::Value = self.post("progress", progress).await?;
        Ok(())
    }
}
