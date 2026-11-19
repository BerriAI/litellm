//! The payloads callbacks hold by handle until they render them.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{PayloadHost, RenderOptions, Rendered};
use litellm_standard_logging::StandardLoggingPayload;
use parking_lot::Mutex;
use serde_json::Value;

#[derive(Default)]
pub(super) struct Payloads {
    next: AtomicU64,
    held: Mutex<HashMap<u64, StandardLoggingPayload>>,
}

impl Payloads {
    pub(super) fn hold(&self, payload: StandardLoggingPayload) -> u64 {
        let handle = self.next.fetch_add(1, Ordering::Relaxed);
        self.held.lock().insert(handle, payload);
        handle
    }

    fn rendered(&self, handle: u64, options: &RenderOptions) -> Result<Vec<u8>, String> {
        let document = match self.held.lock().get(&handle) {
            Some(payload) => serde_json::to_value(payload).map_err(|error| error.to_string())?,
            None => return Err("the payload was released before it was rendered".to_owned()),
        };
        let document = match document {
            Value::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .chain(
                        options
                            .null_keys
                            .iter()
                            .map(|key| (key.clone(), Value::Null)),
                    )
                    .collect(),
            ),
            other => other,
        };
        serde_json::to_vec(&document).map_err(|error| error.to_string())
    }
}

impl PayloadHost for Payloads {
    fn render(&self, handles: Vec<u64>, options: RenderOptions) -> BoxFuture<'static, Rendered> {
        let rendered = handles
            .iter()
            .map(|handle| self.rendered(*handle, &options))
            .collect();
        Box::pin(async move { rendered })
    }

    fn release(&self, handle: u64) {
        self.held.lock().remove(&handle);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::test_support::payload;

    #[rstest]
    fn a_held_payload_renders_with_the_keys_a_callback_nulls_until_it_is_released() {
        let payloads = Payloads::default();
        let handle = payloads.hold(payload());
        let options = RenderOptions {
            null_keys: vec!["response".to_owned()],
        };
        let rendered: Value =
            serde_json::from_slice(&payloads.rendered(handle, &options).unwrap()).unwrap();
        assert_eq!(
            (
                &rendered["id"],
                &rendered["response"],
                &rendered["messages"]
            ),
            (&json!("msg_1"), &Value::Null, &json!("kept"))
        );
        payloads.release(handle);
        assert!(payloads.rendered(handle, &options).is_err());
    }
}
