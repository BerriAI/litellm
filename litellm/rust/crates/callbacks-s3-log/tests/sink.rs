use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Callback, Field, FieldSet, FieldValues, Hook, HookInput, LoggedKeyIdentity, MetadataKind,
    PayloadHost, PayloadRef, RenderOptions, Rendered, Scalar,
};
use litellm_callbacks_s3_log::{BatchUploadFailure, Constants, Host, S3Log, S3LogParts, Settings};
use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{method, path_regex},
};

#[derive(Default)]
struct Recorder {
    failures: AtomicUsize,
    next: AtomicU64,
    payloads: Mutex<HashMap<u64, Option<String>>>,
}

const PANICS: &str = "panics while rendering";

impl Recorder {
    fn stash(self: &Arc<Self>, body: Option<&str>) -> PayloadRef {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.payloads
            .lock()
            .unwrap()
            .insert(id, body.map(str::to_owned));
        PayloadRef::new(id, Arc::clone(self) as Arc<dyn PayloadHost>)
    }

    fn rewrite(&self, id: u64, body: &str) {
        self.payloads
            .lock()
            .unwrap()
            .insert(id, Some(body.to_owned()));
    }

    fn held(&self) -> usize {
        self.payloads.lock().unwrap().len()
    }

    fn failures(&self) -> usize {
        self.failures.load(Ordering::Relaxed)
    }
}

impl PayloadHost for Recorder {
    fn render(&self, payloads: Vec<u64>, options: RenderOptions) -> BoxFuture<'static, Rendered> {
        let bodies: Vec<Option<String>> = payloads
            .iter()
            .map(|payload| {
                self.payloads
                    .lock()
                    .unwrap()
                    .get(payload)
                    .cloned()
                    .flatten()
            })
            .collect();
        let rendered = bodies
            .into_iter()
            .map(|body| match body {
                Some(body) if body == PANICS => panic!("dictionary changed size during iteration"),
                Some(body) if options.null_keys.is_empty() => Ok(body.into_bytes()),
                Some(body) => {
                    Ok(format!("{body} without {}", options.null_keys.join(",")).into_bytes())
                }
                None => Err("the payload cannot be encoded".to_owned()),
            })
            .collect();
        Box::pin(std::future::ready(rendered))
    }

    fn release(&self, payload: u64) {
        self.payloads.lock().unwrap().remove(&payload);
    }
}

impl Host for Recorder {
    fn callback_failed(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    fn web_identity_token(&self, reference: &str) -> Result<String, String> {
        Ok(reference.to_owned())
    }
}

fn fixed_clock() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_763_024_707)
}

fn no_env(_: &str) -> Option<String> {
    None
}

fn sink(server: &MockServer, extra: Value) -> (Arc<S3Log>, Arc<Recorder>) {
    sink_with(server, extra, Constants::default())
}

fn sink_with(
    server: &MockServer,
    extra: Value,
    constants: Constants,
) -> (Arc<S3Log>, Arc<Recorder>) {
    sink_from(server, extra, constants, false, None)
}

fn sink_from(
    server: &MockServer,
    extra: Value,
    constants: Constants,
    cold_storage_is_s3_v2: bool,
    region_from_environment: Option<&str>,
) -> (Arc<S3Log>, Arc<Recorder>) {
    let mut params = json!({
        "s3_bucket_name": "logs",
        "s3_region_name": "us-east-1",
        "s3_endpoint_url": server.uri(),
        "s3_aws_access_key_id": "AKIDEXAMPLE",
        "s3_aws_secret_access_key": "secret",
        "s3_path": "base",
    });
    params
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    params
        .as_object_mut()
        .unwrap()
        .retain(|_, value| !value.is_null());
    let host = Arc::new(Recorder::default());
    let sink = S3Log::new(S3LogParts {
        settings: Settings::from_params(params.as_object().unwrap(), &constants),
        client: litellm_http::Client::for_test(),
        download_client: litellm_http::Client::for_test(),
        environment: Arc::new(no_env),
        clock: fixed_clock,
        backoff: Arc::new(|_| Duration::ZERO),
        host: Arc::clone(&host) as Arc<dyn Host>,
        cold_storage_is_s3_v2,
        region_from_environment: region_from_environment.map(str::to_owned),
        logger: Default::default(),
    });
    (sink, host)
}

const READS: [Field; 6] = [
    Field::DeprecatedKwargs,
    Field::StandardLoggingPayloadRef,
    Field::LogId,
    Field::ColdStorageObjectKey,
    Field::LoggedKeyIdentity,
    Field::AttemptTiming,
];

struct Event {
    payload: Option<PayloadRef>,
    id: &'static str,
    identity: Value,
    cold_storage_object_key: Option<&'static str>,
}

impl Event {
    fn input(self) -> HookInput {
        let mut fields = FieldValues::new(FieldSet::of(&READS))
            .with(Field::DeprecatedKwargs, Value::Null)
            .with(Field::LogId, json!(self.id))
            .with(Field::LoggedKeyIdentity, self.identity)
            .with(
                Field::AttemptTiming,
                json!({"start": {"year": 2026, "month": 11, "day": 13, "hour": 9, "minute": 5, "second": 7, "microsecond": 42}}),
            );
        if let Some(key) = self.cold_storage_object_key {
            fields = fields.with(Field::ColdStorageObjectKey, json!(key));
        }
        if let Some(payload) = self.payload {
            fields.set_payload(payload);
        }
        HookInput {
            hook: Hook::AttemptEnded,
            fields,
            call: None,
        }
    }
}

fn event(host: &Arc<Recorder>, id: &'static str, body: &str) -> Event {
    event_with(host.stash(Some(body)), id)
}

fn event_with(payload: PayloadRef, id: &'static str) -> Event {
    Event {
        payload: Some(payload),
        id,
        identity: json!({"metadata_kind": "mapping", "team_alias": "team-a"}),
        cold_storage_object_key: None,
    }
}

async fn submit(sink: &S3Log, event: Event) {
    sink.handle(&event.input(), None).await.unwrap();
}

fn header(request: &Request, name: &str) -> Option<String> {
    request
        .headers
        .get(name)
        .map(|value| value.to_str().unwrap().to_owned())
}

#[rstest]
#[tokio::test]
async fn a_flush_puts_one_object_per_event_with_the_python_headers() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(
        &server,
        json!({"s3_use_team_prefix": true, "s3_sse_kms_key_id": "kms-1"}),
    );
    submit(
        &sink,
        event(&host, "chatcmpl-1", "{\"id\": \"chatcmpl-1\"}"),
    )
    .await;
    let failure = sink.flush().await;

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let put = &requests[0];
    assert_eq!(
        put.url.path(),
        "/logs/base/team-a/2026-11-13/time-09-05-07-000042_chatcmpl-1.json"
    );
    assert_eq!(put.body, b"{\"id\": \"chatcmpl-1\"}");
    assert_eq!(
        header(put, "content-type").as_deref(),
        Some("application/json")
    );
    assert_eq!(
        header(put, "content-md5").as_deref(),
        Some("Ma6yKrRUR7fU10HikmP0wA==")
    );
    assert_eq!(
        header(put, "content-disposition").as_deref(),
        Some("inline; filename=\"time-2026-11-13T09-05-07-000042_chatcmpl-1.json\"")
    );
    assert_eq!(
        header(put, "x-amz-server-side-encryption").as_deref(),
        Some("aws:kms")
    );
    assert_eq!(
        header(put, "x-amz-date").as_deref(),
        Some("20251113T090507Z")
    );
    assert!(
        header(put, "authorization").unwrap().starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20251113/us-east-1/s3/aws4_request"
        )
    );
    assert!(sink.queued_keys().is_empty());
    assert_eq!(host.failures.load(Ordering::Relaxed), 0);
    assert_eq!(failure, None);
}

#[rstest]
#[tokio::test]
async fn a_bucket_that_fails_every_upload_keeps_the_queue() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({}));
    submit(&sink, event(&host, "a", "{}")).await;
    submit(&sink, event(&host, "b", "{}")).await;
    let failure = sink.flush().await;

    assert_eq!(server.received_requests().await.unwrap().len(), 6);
    assert_eq!(sink.queued_keys().len(), 2);
    assert_eq!(host.failures.load(Ordering::Relaxed), 2);
    assert_eq!(
        failure,
        Some(BatchUploadFailure {
            queued: 2,
            failed: 2,
            total: 2
        })
    );
}

#[rstest]
#[tokio::test]
async fn a_terminal_rejection_is_dropped_when_another_upload_lands() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path_regex("_bad\\.json$"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_string("<Error><Code>EntityTooLarge</Code></Error>"),
        )
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path_regex("_slow\\.json$"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({}));
    for id in ["ok", "bad", "slow"] {
        submit(&sink, event(&host, id, "{}")).await;
    }
    sink.flush().await;

    assert_eq!(
        sink.queued_keys(),
        ["base/2026-11-13/time-09-05-07-000042_slow.json"]
    );
    assert_eq!(host.held(), 1);
}

#[rstest]
#[tokio::test]
async fn batch_files_group_one_flush_by_folder() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({"s3_batch_file_upload": "yes"}));
    submit(&sink, event(&host, "a", "{\"n\": 1}")).await;
    submit(&sink, event(&host, "b", "{\"n\": 2}")).await;
    sink.flush().await;

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0]
            .url
            .path()
            .starts_with("/logs/base/2026-11-13/batch_09-05-07_")
    );
    assert_eq!(requests[0].body, b"{\"n\": 1}\n{\"n\": 2}");
    assert_eq!(
        header(&requests[0], "content-type").as_deref(),
        Some("application/x-ndjson")
    );
}

#[rstest]
#[tokio::test]
async fn an_event_python_would_reject_is_counted_and_skipped() {
    let server = MockServer::start().await;
    let (sink, host) = sink(&server, json!({"s3_use_key_prefix": true}));
    let mut bad = event(&host, "a", "{}");
    bad.identity["key_alias"] = json!(true);
    submit(&sink, bad).await;
    submit(
        &sink,
        Event {
            payload: None,
            id: "b",
            identity: json!({"metadata_kind": "mapping"}),
            cold_storage_object_key: None,
        },
    )
    .await;

    assert!(sink.queued_keys().is_empty());
    assert_eq!(host.failures(), 1);
    assert_eq!(host.held(), 0);
}

#[rstest]
#[tokio::test]
async fn the_body_is_encoded_at_flush_and_released_once_delivered() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({}));
    let payload = host.stash(Some("{\"n\": 1}"));
    let handle = payload.handle();
    submit(&sink, event_with(payload, "a")).await;
    host.rewrite(handle, "{\"n\": 2}");
    sink.flush().await;

    assert_eq!(
        server.received_requests().await.unwrap()[0].body,
        b"{\"n\": 2}"
    );
    assert_eq!(host.held(), 0);
}

#[rstest]
#[tokio::test]
async fn a_body_that_cannot_be_encoded_stays_queued_and_counts_a_failure() {
    let server = MockServer::start().await;
    let (sink, host) = sink(&server, json!({}));
    submit(&sink, event_with(host.stash(None), "a")).await;
    sink.flush().await;

    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(sink.queued_keys().len(), 1);
    assert_eq!(host.failures(), 1);
}

#[rstest]
#[case::environment(Some("ap-south-1"), Some("/ap-south-1/"))]
#[case::default(None, Some("/us-west-2/"))]
#[case::invalid(Some("Ap_South"), None)]
#[tokio::test]
async fn without_a_region_param_the_constructor_region_signs(
    #[case] region: Option<&str>,
    #[case] scope: Option<&str>,
) {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink_from(
        &server,
        json!({"s3_region_name": null, "s3_aws_session_token": "session"}),
        Constants::default(),
        false,
        region,
    );
    submit(&sink, event(&host, "a", "{}")).await;
    sink.flush().await;

    let requests = server.received_requests().await.unwrap();
    let signed = requests
        .first()
        .and_then(|put| header(put, "authorization"))
        .map(|authorization| scope.is_some_and(|scope| authorization.contains(scope)));
    assert_eq!(signed, scope.map(|_| true));
    assert_eq!(sink.queued_keys().len(), usize::from(scope.is_none()));
}

#[rstest]
#[tokio::test]
async fn a_reply_slower_than_the_attempt_timeout_is_retried_then_kept() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(600)))
        .mount(&server)
        .await;
    let (sink, host) = sink_with(
        &server,
        json!({}),
        Constants {
            upload_attempt_timeout_seconds: 0.1,
            ..Constants::default()
        },
    );
    submit(&sink, event(&host, "a", "{}")).await;
    sink.flush().await;

    assert_eq!(server.received_requests().await.unwrap().len(), 3);
    assert_eq!(sink.queued_keys().len(), 1);
    assert_eq!(host.failures(), 1);
}

#[rstest]
#[tokio::test]
async fn a_panic_while_rendering_keeps_that_object_and_the_flush_goes_on() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({}));
    let racing = host.stash(Some(PANICS));
    let handle = racing.handle();
    submit(&sink, event_with(racing, "a")).await;
    submit(&sink, event(&host, "b", "{}")).await;
    sink.flush().await;

    let paths: Vec<String> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|put| put.url.path().to_owned())
        .collect();
    assert_eq!(paths, ["/logs/base/2026-11-13/time-09-05-07-000042_b.json"]);
    assert_eq!(
        sink.queued_keys(),
        ["base/2026-11-13/time-09-05-07-000042_a.json"]
    );
    assert_eq!(host.failures(), 1);

    host.rewrite(handle, "{}");
    sink.flush().await;
    assert!(sink.queued_keys().is_empty());
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[rstest]
#[tokio::test]
async fn a_panic_while_rendering_a_batch_file_keeps_the_whole_queue() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({"s3_batch_file_upload": true}));
    let racing = host.stash(Some(PANICS));
    let handle = racing.handle();
    submit(&sink, event_with(racing, "a")).await;
    submit(&sink, event(&host, "b", "{}")).await;
    sink.flush().await;

    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(sink.queued_keys().len(), 2);

    host.rewrite(handle, "{}");
    sink.flush().await;
    assert!(sink.queued_keys().is_empty());
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[rstest]
#[case::bool_alias(json!({"metadata_kind": "mapping", "team_alias": true}))]
#[case::int_alias(json!({"metadata_kind": "mapping", "team_alias": 7}))]
#[case::nan_alias(serde_json::to_value(LoggedKeyIdentity {
    metadata_kind: Some(MetadataKind::Mapping),
    team_alias: Some(Scalar::Float(f64::NAN)),
    ..LoggedKeyIdentity::default()
}).unwrap())]
#[case::list_alias(json!({"metadata_kind": "mapping", "team_alias": {"text": "['team-a']", "truthy": true, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"}}))]
#[case::alias_whose_str_raises(json!({"metadata_kind": "mapping", "rejected": ["team_alias"]}))]
#[case::null_metadata(json!({"metadata_kind": "null"}))]
#[case::metadata_without_get(json!({"metadata_kind": "other"}))]
#[tokio::test]
async fn a_team_prefix_python_cannot_build_is_counted_and_released(#[case] identity: Value) {
    let server = MockServer::start().await;
    let (sink, host) = sink(&server, json!({"s3_use_team_prefix": true}));
    let mut bad = event(&host, "a", "{}");
    bad.identity = identity;
    submit(&sink, bad).await;

    assert!(sink.queued_keys().is_empty());
    assert_eq!(host.failures(), 1);
    assert_eq!(host.held(), 0);
}

#[rstest]
#[case::empty_text(json!(""))]
#[case::zero(json!(0))]
#[case::zero_float(json!(0.0))]
#[case::false_flag(json!(false))]
#[case::null(Value::Null)]
#[case::empty_list(json!({"text": "[]", "truthy": false, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"}))]
#[tokio::test]
async fn a_falsy_team_alias_adds_no_prefix(#[case] alias: Value) {
    let server = MockServer::start().await;
    let (sink, host) = sink(&server, json!({"s3_use_team_prefix": true}));
    let mut falsy = event(&host, "a", "{}");
    falsy.identity = json!({"metadata_kind": "mapping", "team_alias": alias});
    submit(&sink, falsy).await;

    assert_eq!(
        sink.queued_keys(),
        ["base/2026-11-13/time-09-05-07-000042_a.json"]
    );
    assert_eq!(host.failures(), 0);
}

#[rstest]
#[case::prompts_only(json!({"s3_log_prompts_only": true}), "{} without response")]
#[case::whole_payload(json!({}), "{}")]
#[tokio::test]
async fn prompts_only_renders_the_response_as_null(#[case] extra: Value, #[case] body: &str) {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, extra);
    submit(&sink, event(&host, "a", "{}")).await;
    sink.flush().await;

    assert_eq!(
        server.received_requests().await.unwrap()[0].body,
        body.as_bytes()
    );
}

#[rstest]
#[tokio::test]
async fn a_region_param_wins_over_the_constructor_region() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink_from(
        &server,
        json!({}),
        Constants::default(),
        false,
        Some("ap-south-1"),
    );
    submit(&sink, event(&host, "a", "{}")).await;
    sink.flush().await;

    let requests = server.received_requests().await.unwrap();
    assert!(
        header(&requests[0], "authorization")
            .unwrap()
            .contains("/us-east-1/")
    );
}

const HOUR_KEY: &str = "base/2026-11-13/09/time-09-05-07-000042_a.json";
const DAY_KEY: &str = "base/2026-11-13/time-09-05-07-000042_a.json";

#[rstest]
#[case::s3_v2_reuses_its_hour_key(true, json!({"metadata_kind": "mapping"}), Some(HOUR_KEY))]
#[case::another_owner_keeps_the_day_key(false, json!({"metadata_kind": "mapping"}), Some(DAY_KEY))]
#[case::s3_v2_rejects_metadata_without_get(true, json!({"metadata_kind": "other"}), None)]
#[case::another_owner_ignores_metadata_without_get(false, json!({"metadata_kind": "other"}), Some(DAY_KEY))]
#[tokio::test]
async fn the_constructor_cold_storage_owner_decides_the_cold_key(
    #[case] cold_storage_is_s3_v2: bool,
    #[case] identity: Value,
    #[case] key: Option<&str>,
) {
    let server = MockServer::start().await;
    let (sink, host) = sink_from(
        &server,
        json!({}),
        Constants::default(),
        cold_storage_is_s3_v2,
        None,
    );
    let mut cold = event(&host, "a", "{}");
    cold.identity = identity;
    cold.cold_storage_object_key = Some(HOUR_KEY);
    submit(&sink, cold).await;

    assert_eq!(sink.queued_keys(), key.into_iter().collect::<Vec<_>>());
    assert_eq!(host.failures(), usize::from(key.is_none()));
}

#[rstest]
#[case::s3_v2_owner(true, "/logs/base/2026-11-13/time-09-05-07-000042_a.json")]
#[case::another_owner(false, "/logs/base/2026-11-13/batch_09-05-07_")]
#[tokio::test]
async fn the_constructor_cold_storage_owner_turns_batch_files_off(
    #[case] cold_storage_is_s3_v2: bool,
    #[case] path: &str,
) {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink_from(
        &server,
        json!({"s3_batch_file_upload": true}),
        Constants::default(),
        cold_storage_is_s3_v2,
        None,
    );
    submit(&sink, event(&host, "a", "{}")).await;
    sink.flush().await;

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].url.path().starts_with(path));
}
