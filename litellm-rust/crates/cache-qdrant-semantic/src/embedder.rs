use std::time::Duration;

use litellm_cache::{Error, semantic::Embedder};
use reqwest::Client;
use serde_json::Value;

pub struct OpenAiEmbedder {
    client: Client,
    api_base: String,
    api_key: String,
    model: String,
    timeout: Option<Duration>,
}

pub struct OpenAiEmbedderConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub timeout: Option<Duration>,
}

impl OpenAiEmbedder {
    pub fn new(client: Client, config: OpenAiEmbedderConfig) -> Self {
        Self {
            client,
            api_base: config.api_base.trim_end_matches('/').to_owned(),
            api_key: config.api_key,
            model: config.model,
            timeout: config.timeout,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

/// An OpenAI-compatible `/embeddings` call. It has no router to route on, so `metadata` is
/// unused, and it only embeds asynchronously: sync cache calls block on the cache's runtime.
impl Embedder for OpenAiEmbedder {
    async fn async_embed(&self, input: &str, _metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        let request = self
            .client
            .post(format!("{}/embeddings", self.api_base))
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "model": self.model,
                "input": input,
                "encoding_format": "float",
            }));
        let response = if let Some(timeout) = self.timeout {
            request.timeout(timeout)
        } else {
            request
        }
        .send()
        .await
        .map_err(|_| Error::Unavailable)?
        .error_for_status()
        .map_err(|_| Error::Unavailable)?;
        let body: Value = response.json().await.map_err(|_| Error::Unavailable)?;
        body.get("data")
            .and_then(Value::as_array)
            .and_then(|data| data.first())
            .and_then(|item| item.get("embedding"))
            .and_then(Value::as_array)
            .and_then(|embedding| {
                embedding
                    .iter()
                    .map(|value| value.as_f64().map(|value| value as f32))
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or(Error::Unavailable)
    }
}
