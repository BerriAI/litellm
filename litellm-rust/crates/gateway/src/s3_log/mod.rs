//! The `s3_v2` callback: every request's `StandardLoggingPayload` written to S3 as a JSON
//! object, under the proxy's key layout, as the proxy's `S3Logger` writes it.
//!
//! It is an observer, so nothing here runs on the request path. Each finished request is turned
//! into its object and queued; every flush interval the queue is uploaded with bounded
//! concurrency, or written as one NDJSON object per folder with `s3_batch_file_upload`. What
//! fails for now goes back on the queue for the next flush, until it is older than
//! `s3_max_retry_age_seconds` or the queue overflows.

mod config;
mod key;
mod payload;
mod upload;

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex, Weak},
    time::{Instant, SystemTime},
};

use axum::body::Bytes;
use futures_util::{StreamExt, future::BoxFuture};
use litellm_config::Config;
use litellm_gateway_hooks::{Facts, Observer, Record};
use litellm_gateway_inference::Gateway;
use serde_json::Value;

use self::{
    config::Settings,
    key::{Utc, download_name, logging_id, object_key, parent},
    payload::{Redaction, prompts_only, standard_logging_payload, strip_base64_files},
    upload::{Upload, Uploader},
};
use crate::Error;

const JSON: &str = "application/json";
const NDJSON: &str = "application/x-ndjson";

/// One object waiting to be written.
#[derive(Clone, Debug)]
struct Element {
    key: String,
    download_name: String,
    body: Bytes,
    content_type: &'static str,
    /// A batch file, already holding many requests; never grouped again.
    batch: bool,
    /// When this object first failed to upload.
    retrying_since: Option<Instant>,
}

#[derive(Default)]
struct Queue {
    elements: VecDeque<Element>,
    /// How many at the front were put back by the last failed flush.
    requeued: usize,
}

/// The `s3_v2` observer.
pub struct S3Log {
    settings: Arc<Settings>,
    uploader: Uploader,
    queue: Mutex<Queue>,
    /// Serializes flushes, so the periodic one and the one at shutdown never interleave.
    flushing: tokio::sync::Mutex<()>,
}

impl S3Log {
    /// Starts the observer and its flush timer, or `Ok(None)` when no callback list names
    /// `s3_v2`. Must be called inside a tokio runtime.
    pub async fn from_config(
        config: &Config,
        inference: &Gateway,
    ) -> Result<Option<Arc<Self>>, Error> {
        let env = |name: &str| std::env::var(name).ok();
        let Some(settings) =
            Settings::from_config(config, inference.secrets.as_ref(), &env).await?
        else {
            return Ok(None);
        };
        let settings = Arc::new(settings);
        let log = Arc::new(Self {
            uploader: Uploader::new(
                settings.clone(),
                inference.resources.pool.clone(),
                &inference.http,
            ),
            settings,
            queue: Mutex::default(),
            flushing: tokio::sync::Mutex::new(()),
        });
        tokio::spawn(run(Arc::downgrade(&log)));
        Ok(Some(log))
    }

    fn element(&self, record: &Record) -> Element {
        let settings = &self.settings;
        let redaction = Redaction {
            messages: settings.turn_off_message_logging,
            user_api_key_info: settings.redact_user_api_key_info,
            user_agent_tags: settings.user_agent_tags,
        };
        let mut payload = standard_logging_payload(record, redaction);
        if settings.strip_base64_files {
            strip_base64_files(&mut payload);
        }
        let metadata = |field: &str| {
            payload["metadata"][field]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        let prefix: String = [
            settings
                .team_prefix
                .then(|| metadata("user_api_key_team_alias")),
            settings.key_prefix.then(|| metadata("user_api_key_alias")),
        ]
        .into_iter()
        .flatten()
        .flatten()
        .map(|segment| format!("{segment}/"))
        .collect();
        let id = payload["id"].as_str().unwrap_or_default().to_owned();
        let start = Utc::new(record.request.started_at);
        if settings.prompts_only {
            prompts_only(&mut payload);
        }
        Element {
            key: object_key(
                &settings.path,
                &prefix,
                &start,
                &logging_id(&start, &id),
                settings.granularity,
            ),
            download_name: download_name(&start, &id),
            body: Bytes::from(Value::Object(payload).to_string()),
            content_type: JSON,
            batch: false,
            retrying_since: None,
        }
    }

    fn enqueue(&self, element: Element) {
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        queue.elements.push_back(element);
        let excess = queue
            .elements
            .len()
            .saturating_sub(self.settings.max_queue_size);
        if excess > 0 {
            queue.elements.drain(..excess);
            queue.requeued = queue.requeued.saturating_sub(excess);
            tracing::warn!(
                dropped = excess,
                max_queue_size = self.settings.max_queue_size,
                "s3 logging: queue full; dropped the oldest events"
            );
        }
    }

    /// Uploads everything queued; what fails for now is put back at the front of the queue.
    pub async fn flush(&self) {
        let _flushing = self.flushing.lock().await;
        let (batch, requeued) = {
            let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            let requeued = std::mem::take(&mut queue.requeued);
            (Vec::from(std::mem::take(&mut queue.elements)), requeued)
        };
        if batch.is_empty() {
            return;
        }
        let batch_len = batch.len();
        let uploads = match self.settings.batch_file_upload {
            true => batch_files(batch),
            false => batch,
        };
        // What failed last time goes last, so a backlog never starves new requests.
        let stale = match uploads.len() == batch_len {
            true => requeued.min(uploads.len()),
            false => 0,
        };
        let order: Vec<usize> = (stale..uploads.len()).chain(0..stale).collect();
        let mut outcomes = vec![Upload::Retry; uploads.len()];
        let mut finished = futures_util::stream::iter(order)
            .map(|index| {
                let uploads = &uploads;
                async move { (index, self.uploader.upload(&uploads[index]).await) }
            })
            .buffer_unordered(self.settings.max_concurrent_uploads);
        while let Some((index, outcome)) = finished.next().await {
            outcomes[index] = outcome;
        }
        drop(finished);
        let delivered = outcomes
            .iter()
            .filter(|outcome| **outcome == Upload::Delivered)
            .count();
        let total = uploads.len();
        let failed = total - delivered;
        if failed == 0 {
            return;
        }
        let now = Instant::now();
        let max_age = self.settings.max_retry_age;
        let failures = uploads
            .into_iter()
            .zip(outcomes)
            .filter(|(_, outcome)| *outcome != Upload::Delivered);
        // When nothing got through the bucket itself is failing: keep everything, unaged.
        let kept: Vec<Element> = match delivered {
            0 => failures.map(|(element, _)| element).collect(),
            _ => failures
                .filter(|(element, outcome)| {
                    *outcome == Upload::Retry
                        && !matches!(
                            (max_age, element.retrying_since),
                            (Some(max_age), Some(since)) if now.duration_since(since) > max_age
                        )
                })
                .map(|(element, _)| Element {
                    retrying_since: element.retrying_since.or(max_age.map(|_| now)),
                    ..element
                })
                .collect(),
        };
        tracing::warn!(
            failed,
            kept = kept.len(),
            total,
            "s3 logging: uploads failed this flush; the kept ones are retried on the next, the rest \
             were rejected by S3 or retried longer than s3_max_retry_age_seconds"
        );
        if kept.is_empty() {
            return;
        }
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let arrivals = std::mem::take(&mut queue.elements);
        let kept_len = kept.len();
        let mut elements: VecDeque<Element> = kept.into_iter().chain(arrivals).collect();
        let overflow = elements.len().saturating_sub(self.settings.max_queue_size);
        if overflow > 0 {
            elements.drain(..overflow);
            tracing::warn!(
                dropped = overflow,
                max_queue_size = self.settings.max_queue_size,
                "s3 logging: queue exceeded max_queue_size after a failed flush; dropped the oldest events"
            );
        }
        queue.elements = elements;
        queue.requeued = kept_len.saturating_sub(overflow);
    }
}

/// `s3_batch_file_upload`: one NDJSON object per folder, named for the flush.
fn batch_files(batch: Vec<Element>) -> Vec<Element> {
    let now = Utc::new(SystemTime::now());
    let (ready, singles): (Vec<Element>, Vec<Element>) =
        batch.into_iter().partition(|element| element.batch);
    let mut folders: BTreeMap<String, Vec<Element>> = BTreeMap::new();
    for element in singles {
        folders
            .entry(parent(&element.key).to_owned())
            .or_default()
            .push(element);
    }
    let files = folders.into_iter().map(|(folder, elements)| {
        let name = format!("batch_{}_{}", now.clock(), uuid::Uuid::new_v4().simple());
        let lines: Vec<&[u8]> = elements.iter().map(|element| &element.body[..]).collect();
        Element {
            key: match folder.is_empty() {
                true => format!("{name}.jsonl"),
                false => format!("{folder}/{name}.jsonl"),
            },
            download_name: format!("{name}.jsonl"),
            body: Bytes::from(lines.join(&b'\n')),
            content_type: NDJSON,
            batch: true,
            retrying_since: elements
                .iter()
                .filter_map(|element| element.retrying_since)
                .min(),
        }
    });
    ready.into_iter().chain(files).collect()
}

impl Observer for S3Log {
    fn name(&self) -> &'static str {
        "s3_v2"
    }

    fn reads(&self) -> Facts {
        Facts::KEY
            | Facts::END_USER
            | Facts::CLIENT_HEADERS
            | Facts::DEPLOYMENT
            | Facts::USAGE
            | Facts::COST
            | Facts::TIMINGS
            | Facts::ANNOTATIONS
            | Facts::REQUEST_BODY
            | Facts::RESPONSE_BODY
    }

    fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
        let logged = match record.succeeded() {
            true => self.settings.log_success,
            false => self.settings.log_failure,
        };
        if logged {
            self.enqueue(self.element(record));
        }
        Box::pin(async {})
    }

    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(S3Log::flush(self))
    }
}

/// Flushes every flush interval until the observer is dropped.
async fn run(log: Weak<S3Log>) {
    let interval = match log.upgrade() {
        Some(log) => log.settings.flush_interval,
        None => return,
    };
    loop {
        tokio::time::sleep(interval).await;
        let Some(log) = log.upgrade() else {
            return;
        };
        log.flush().await;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use litellm_gateway_hooks::Route;
    use serde_json::json;

    use super::*;

    fn element(key: &str, body: &str) -> Element {
        Element {
            key: key.into(),
            download_name: "d.json".into(),
            body: Bytes::from(body.to_owned()),
            content_type: JSON,
            batch: false,
            retrying_since: None,
        }
    }

    #[test]
    fn batch_files_hold_one_line_per_request_per_folder() {
        let earlier = Instant::now();
        let mut retried = element("logs/2026-03-04/b.json", "{\"b\":1}");
        retried.retrying_since = Some(earlier);
        let files = batch_files(vec![
            element("logs/2026-03-04/a.json", "{\"a\":1}"),
            element("logs/team/2026-03-04/c.json", "{\"c\":1}"),
            retried,
        ]);
        assert_eq!(files.len(), 2);
        let day = &files[0];
        let (folder, name) = day.key.rsplit_once('/').unwrap();
        assert_eq!(folder, "logs/2026-03-04");
        assert!(
            name.starts_with("batch_") && name.ends_with(".jsonl"),
            "{name}"
        );
        assert_eq!(day.download_name, name);
        assert_eq!(&day.body[..], b"{\"a\":1}\n{\"b\":1}");
        assert_eq!(day.content_type, NDJSON);
        assert_eq!(day.retrying_since, Some(earlier));
        assert!(files[1].key.starts_with("logs/team/2026-03-04/batch_"));
        // A batch file kept for a retry is sent again as it is.
        let again = batch_files(files.clone());
        assert_eq!(
            again.iter().map(|file| &file.key).collect::<Vec<_>>(),
            files.iter().map(|file| &file.key).collect::<Vec<_>>()
        );
    }

    async fn log(params: &str) -> S3Log {
        let settings = config::tests::settings_with(
            &format!(
                "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n    s3_path: audit\n{params}"
            ),
            &[],
        )
        .await
        .unwrap()
        .unwrap();
        let settings = Arc::new(settings);
        S3Log {
            uploader: Uploader::new(
                settings.clone(),
                Arc::new(litellm_http::HttpClientPool::new(Arc::new(
                    litellm_http::media::PublicDnsResolver,
                ))),
                &litellm_http::Resolution::from(&litellm_http::HttpSettings::default()).config,
            ),
            settings,
            queue: Mutex::default(),
            flushing: tokio::sync::Mutex::new(()),
        }
    }

    fn record() -> Record {
        payload::tests::record(
            Route::ChatCompletions,
            200,
            false,
            json!({"messages": [{"role": "user", "content": "hi"}]}),
            Some(payload::tests::COMPLETION),
        )
    }

    #[rstest::rstest]
    #[case::plain("", "audit/1970-01-01/time-00-16-40-000000_chatcmpl-1.json")]
    #[case::team(
        "    s3_use_team_prefix: true\n",
        "audit/research/1970-01-01/time-00-16-40-000000_chatcmpl-1.json"
    )]
    #[case::team_and_key(
        "    s3_use_team_prefix: true\n    s3_use_key_prefix: true\n    s3_partition_granularity: hour\n",
        "audit/research/team-key/1970-01-01/00/time-00-16-40-000000_chatcmpl-1.json"
    )]
    #[tokio::test]
    async fn objects_are_keyed_by_team_key_and_time(#[case] params: &str, #[case] expected: &str) {
        let element = log(params).await.element(&record());
        assert_eq!(element.key, expected);
        assert_eq!(
            element.download_name,
            "time-1970-01-01T00-16-40-000000_chatcmpl-1.json"
        );
    }

    #[tokio::test]
    async fn prompts_only_objects_have_no_response() {
        let element = log("    s3_log_prompts_only: true\n")
            .await
            .element(&record());
        let body: Value = serde_json::from_slice(&element.body).unwrap();
        assert_eq!(body["response"], Value::Null);
        assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
        assert!(element.key.ends_with("_chatcmpl-1.json"));
    }

    #[tokio::test]
    async fn only_the_listed_outcomes_are_queued() {
        let settings = config::tests::settings_with(
            "litellm_settings:\n  failure_callback: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n",
            &[],
        )
        .await
        .unwrap()
        .unwrap();
        let mut log = log("").await;
        log.settings = Arc::new(settings);
        log.observe(&record()).await;
        assert!(log.queue.lock().unwrap().elements.is_empty());
        let mut failed = record();
        failed.outcome.status = 500;
        log.observe(&failed).await;
        assert_eq!(log.queue.lock().unwrap().elements.len(), 1);
    }

    #[tokio::test]
    async fn a_full_queue_drops_the_oldest() {
        let log = log("    s3_max_queue_size: 2\n").await;
        for key in ["a", "b", "c"] {
            log.enqueue(element(key, "{}"));
        }
        let queue = log.queue.lock().unwrap();
        let keys: Vec<_> = queue
            .elements
            .iter()
            .map(|element| element.key.as_str())
            .collect();
        assert_eq!(keys, ["b", "c"]);
    }

    #[tokio::test]
    async fn an_unreachable_bucket_keeps_everything_for_the_next_flush() {
        let log = log("    s3_endpoint_url: http://127.0.0.1:9\n    s3_aws_access_key_id: k\n    s3_aws_secret_access_key: s\n    s3_max_retry_age_seconds: 1\n").await;
        let mut old = element("a", "{}");
        old.retrying_since = Some(Instant::now() - Duration::from_secs(60));
        log.enqueue(old);
        log.enqueue(element("b", "{}"));
        log.flush().await;
        let queue = log.queue.lock().unwrap();
        let kept: Vec<_> = queue
            .elements
            .iter()
            .map(|element| element.key.as_str())
            .collect();
        assert_eq!(kept, ["a", "b"]);
        assert_eq!(queue.requeued, 2);
    }
}
