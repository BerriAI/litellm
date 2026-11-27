//! The `StandardLoggingPayload` of each call the gateway serves, as the callbacks that read
//! `Field::StandardLoggingPayloadRef` hold it by handle: the Rust counterpart of the Python
//! payload host in `callbacks-python/src/payload.rs`.
//!
//! A payload is built once per hook from the call's record, when a callback subscribed to that
//! hook reads it, and kept built until its last reference is released. It renders in the layout
//! of Python's `json.dumps`, so a callback reads the same bytes from either host. Building it at render time
//! instead would mean keeping a snapshot of the whole record until a flush renders it, and the
//! logged response's `created` and a cache hit's id would carry the flush time, where Python
//! stamps them when the call is logged.

use std::{
    collections::HashMap,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::SystemTime,
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{PayloadHost, PayloadRef, RenderOptions, Rendered};
use litellm_gateway_inference::{CallType, record::CallState};
use litellm_python_compat::json;
use litellm_redaction::SecretRedactor;
use litellm_router_core::Router;
use litellm_standard_logging::StandardLoggingPayload;
use litellm_token_counter::TokenCounter;
use parking_lot::Mutex;
use serde_json::Value;

use super::standard_logging::{Context, failed_on, standard_logging_payload};

/// `get_default_encoding()`, the tiktoken encoding `token_counter` falls back to.
const DEFAULT_ENCODING: &str = "cl100k_base";
/// `redact_string`'s shortest custom key.
const MINIMUM_CUSTOM_KEY_LENGTH: usize = 16;

/// The built payloads callbacks hold by handle until they release them.
#[derive(Default)]
pub struct RustPayloads {
    next: AtomicU64,
    held: Mutex<HashMap<u64, StandardLoggingPayload>>,
}

impl RustPayloads {
    /// Holds `payload` until the returned reference, and every clone of its `Arc`, is dropped.
    pub fn hold(self: &Arc<Self>, payload: StandardLoggingPayload) -> PayloadRef {
        let handle = self.next.fetch_add(1, Ordering::Relaxed);
        self.held.lock().insert(handle, payload);
        PayloadRef::new(handle, Arc::clone(self) as Arc<dyn PayloadHost>)
    }

    /// How many payloads are held.
    #[cfg(test)]
    pub fn held(&self) -> usize {
        self.held.lock().len()
    }

    /// The payload's wire JSON, each of `options.null_keys` set to `null` as the Python host's
    /// dict copy sets it: in place when the payload logs the key, and last when it does not.
    /// Written as `json.dumps` writes it, the way the Python host's encoder lays out the dict.
    fn rendered(&self, handle: u64, options: &RenderOptions) -> Result<Vec<u8>, String> {
        let held = self.held.lock();
        let payload = held
            .get(&handle)
            .ok_or_else(|| "the payload was already released".to_owned())?;
        if options.null_keys.is_empty() {
            return json::dumps_serde(payload).map_err(|error| error.to_string());
        }
        let mut document = serde_json::to_value(payload).map_err(|error| error.to_string())?;
        drop(held);
        if let Value::Object(keys) = &mut document {
            for key in &options.null_keys {
                keys.insert(key.clone(), Value::Null);
            }
        }
        json::dumps_serde(&document).map_err(|error| error.to_string())
    }
}

impl PayloadHost for RustPayloads {
    fn render(&self, handles: Vec<u64>, options: RenderOptions) -> BoxFuture<'static, Rendered> {
        let rendered = handles
            .iter()
            .map(|handle| self.rendered(*handle, &options))
            .collect();
        Box::pin(async move { rendered })
    }

    fn release(&self, handle: u64) {
        let released = self.held.lock().remove(&handle);
        drop(released);
    }
}

/// A call's payload for one hook, and the host a callback's reference to it points into.
#[derive(Clone, Copy)]
pub struct Logged<'a> {
    pub payload: &'a StandardLoggingPayload,
    pub host: &'a Arc<RustPayloads>,
}

/// What builds a call's payload besides its record, and the host its references point into.
pub struct LoggingPayloads {
    host: Arc<RustPayloads>,
    /// Where the deployment a failed call last ran on is looked up. `None` logs no deployment
    /// for a failure.
    router: Option<Router>,
    redactor: SecretRedactor,
    counter: OnceLock<Option<TokenCounter>>,
}

impl Default for LoggingPayloads {
    fn default() -> Self {
        Self {
            host: Arc::default(),
            router: None,
            redactor: SecretRedactor::new(MINIMUM_CUSTOM_KEY_LENGTH),
            counter: OnceLock::new(),
        }
    }
}

impl LoggingPayloads {
    /// Payloads whose failures name the deployment `router` registers under the id the call
    /// last ran on.
    pub fn over(router: Router) -> Self {
        Self {
            router: Some(router),
            ..Self::default()
        }
    }

    pub fn host(&self) -> &Arc<RustPayloads> {
        &self.host
    }

    /// `token_counter(text=..., count_response_tokens=True)` over the default encoding, 0 when
    /// it cannot be loaded.
    fn count_tokens(&self, text: &str) -> u64 {
        self.counter
            .get_or_init(|| TokenCounter::from_tiktoken(DEFAULT_ENCODING).ok())
            .as_ref()
            .and_then(|counter| counter.count_text(text).ok())
            .map_or(0, |count| u64::try_from(count).unwrap_or(u64::MAX))
    }

    /// The payload of the call `entries` records, as of now. `None` for a call that has not
    /// ended, for a call of a route other than `/v1/messages`, which the builder does not shape
    /// yet, and for one whose answer is not logged the way Python logs it, which is logged here.
    pub fn build(&self, entries: &CallState) -> Option<StandardLoggingPayload> {
        if entries.call_type != Some(CallType::AnthropicMessages) {
            return None;
        }
        let deployment = failed_on(entries)
            .zip(self.router.as_ref())
            .and_then(|(id, router)| router.registry().deployment(id));
        let count_tokens = |text: &str| self.count_tokens(text);
        let context = Context {
            failed_on: deployment.as_deref(),
            redactor: &self.redactor,
            now: SystemTime::now(),
            count_tokens: &count_tokens,
        };
        match standard_logging_payload(entries, &context) {
            Ok(payload) => Some(payload),
            Err(super::standard_logging::Unbuilt::NotEnded) => None,
            Err(error) => {
                tracing::error!(error = %litellm_error::Report::new(&error), "the standard logging payload was not built");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;
    use litellm_gateway_inference::{
        CallType,
        record::{self, CallRecord},
    };
    use rstest::{fixture, rstest};
    use serde_json::{Map, json};

    use super::*;

    /// The payload of a `/v1/messages` call that failed before it was routed.
    #[fixture]
    fn payload() -> StandardLoggingPayload {
        let built = || {
            record::write(|entries| {
                entries.call_type = Some(CallType::AnthropicMessages);
                entries.call_id = Some("call-1".to_owned());
                entries.hold(Map::from_iter([("model".to_owned(), json!("claude"))]));
                entries.release();
                entries.fail();
                LoggingPayloads::default().build(entries)
            })
            .flatten()
        };
        record::scope(CallRecord::new(), async { built() })
            .now_or_never()
            .unwrap()
            .unwrap()
    }

    fn rendered(payloads: &Arc<RustPayloads>, reference: &PayloadRef, nulled: &[&str]) -> Value {
        let options = RenderOptions {
            null_keys: nulled.iter().map(|key| (*key).to_owned()).collect(),
        };
        let bytes = reference.render(options).now_or_never().unwrap().unwrap();
        assert_eq!(payloads.held(), 1);
        serde_json::from_slice(&bytes).unwrap()
    }

    #[rstest]
    fn a_held_payload_renders_as_json_dumps_writes_its_wire_json(payload: StandardLoggingPayload) {
        let payloads = Arc::new(RustPayloads::default());
        let reference = payloads.hold(payload.clone());
        let bytes = reference
            .render(RenderOptions::default())
            .now_or_never()
            .unwrap()
            .unwrap();
        assert_eq!(bytes, json::dumps_serde(&payload).unwrap());
    }

    #[rstest]
    #[case::without_null_keys(&[])]
    #[case::with_null_keys(&["response", "not_logged"])]
    fn a_rendered_payload_has_pythons_spacing_and_ascii_escapes(
        payload: StandardLoggingPayload,
        #[case] nulled: &[&str],
    ) {
        let payloads = Arc::new(RustPayloads::default());
        let reference = payloads.hold(StandardLoggingPayload {
            id: "caf\u{e9} \u{1f600}".to_owned(),
            response_time: 1e-6,
            ..payload
        });
        let options = RenderOptions {
            null_keys: nulled.iter().map(|key| (*key).to_owned()).collect(),
        };
        let text =
            String::from_utf8(reference.render(options).now_or_never().unwrap().unwrap()).unwrap();
        assert!(
            text.starts_with(r#"{"id": "caf\u00e9 \ud83d\ude00", "litellm_call_id": "call-1", "#)
                && text.contains(r#""response_time": 1e-06, "#),
            "{text}"
        );
    }

    /// The keys of a rendered document, in order.
    fn keys(document: &Value) -> Vec<String> {
        document.as_object().unwrap().keys().cloned().collect()
    }

    #[rstest]
    fn a_nulled_key_the_payload_logs_keeps_its_place(payload: StandardLoggingPayload) {
        let payloads = Arc::new(RustPayloads::default());
        let whole = serde_json::to_value(&payload).unwrap();
        let reference = payloads.hold(payload);
        let nulled = rendered(&payloads, &reference, &["response"]);
        assert_eq!(
            (&nulled["response"], keys(&nulled)),
            (&Value::Null, keys(&whole))
        );
    }

    #[rstest]
    fn a_nulled_key_the_payload_does_not_log_is_added_last(payload: StandardLoggingPayload) {
        let payloads = Arc::new(RustPayloads::default());
        let reference = payloads.hold(payload);
        let nulled = rendered(&payloads, &reference, &["not_logged"]);
        assert_eq!(
            (
                &nulled["not_logged"],
                keys(&nulled).last().map(String::as_str)
            ),
            (&Value::Null, Some("not_logged"))
        );
    }

    #[rstest]
    fn a_released_payload_is_no_longer_held_or_rendered(payload: StandardLoggingPayload) {
        let payloads = Arc::new(RustPayloads::default());
        let reference = payloads.hold(payload);
        let handle = reference.handle();
        reference.release();
        assert_eq!(
            (
                payloads.held(),
                payloads.rendered(handle, &RenderOptions::default())
            ),
            (0, Err("the payload was already released".to_owned()))
        );
    }

    #[rstest]
    fn several_payloads_render_in_one_call_in_order(payload: StandardLoggingPayload) {
        let payloads = Arc::new(RustPayloads::default());
        let first = payloads.hold(payload.clone());
        let second = payloads.hold(StandardLoggingPayload {
            id: "second".to_owned(),
            ..payload
        });
        let rendered = litellm_callbacks::render_all(&[&first, &second], RenderOptions::default())
            .now_or_never()
            .unwrap()
            .unwrap();
        let ids: Vec<Value> = rendered
            .iter()
            .map(|bytes| serde_json::from_slice::<Value>(bytes).unwrap()["id"].clone())
            .collect();
        assert_eq!(ids, [json!("call-1"), json!("second")]);
    }

    #[rstest]
    #[case::not_ended(Some(CallType::AnthropicMessages), false)]
    #[case::an_ocr_call(Some(CallType::Aocr), true)]
    #[case::of_no_route(None, true)]
    fn a_call_the_builder_does_not_shape_has_no_payload(
        #[case] call_type: Option<CallType>,
        #[case] ended: bool,
    ) {
        let built = || {
            record::write(|entries| {
                entries.call_type = call_type;
                if ended {
                    entries.fail();
                }
                LoggingPayloads::default().build(entries)
            })
        };
        let built = record::scope(CallRecord::new(), async { built() })
            .now_or_never()
            .unwrap();
        assert!(matches!(built, Some(None)));
    }
}
