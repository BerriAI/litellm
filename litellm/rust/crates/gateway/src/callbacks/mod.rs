//! The config's logging callbacks, each handed the calls the inference routes log.

mod fields;
mod payloads;
mod s3_log;

use std::{sync::Arc, time::Duration};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    ErasedCallback, Field, FieldSet, FieldValues, Hook, HookInput, PayloadRef,
};
use litellm_config::{Config, Environment};
use litellm_gateway_inference::CallLog;
use litellm_standard_logging::StandardLoggingPayload;
use tokio::task::JoinHandle;

use self::payloads::Payloads;
use crate::{Error, Resources};

/// One callback of the config and how often what it queued is flushed.
pub(crate) struct Configured {
    pub(crate) callback: Arc<dyn ErasedCallback>,
    pub(crate) flush_every: Duration,
}

pub struct Callbacks {
    callbacks: Vec<Arc<dyn ErasedCallback>>,
    payloads: Arc<Payloads>,
    flushers: Vec<JoinHandle<()>>,
}

/// Whether `config` names the callback `name` among those every call or every success runs.
fn named(config: &Config, name: &str) -> bool {
    let settings = &config.litellm_settings;
    [&settings.callbacks, &settings.success_callback]
        .into_iter()
        .flatten()
        .flatten()
        .any(|callback| callback.as_str() == Some(name))
}

fn flusher(configured: &Configured) -> JoinHandle<()> {
    let callback = configured.callback.clone();
    let every = configured.flush_every;
    tokio::spawn(async move {
        let mut ticks = tokio::time::interval(every);
        loop {
            ticks.tick().await;
            callback.flush().await;
        }
    })
}

fn reads_when_an_attempt_ends(callback: &dyn ErasedCallback) -> impl Iterator<Item = FieldSet> {
    callback
        .manifest()
        .subscriptions
        .iter()
        .filter(|subscription| subscription.hook == Hook::AttemptEnded)
        .map(|subscription| subscription.reads)
}

impl Callbacks {
    /// The callbacks `config` names that the gateway runs itself, and nothing when it names none.
    pub(crate) async fn build(
        config: &Config,
        resources: &Resources,
        environment: &Environment,
    ) -> Result<Option<Arc<Self>>, Error> {
        let configured: Vec<Configured> = s3_log::build(config, resources, environment)
            .await?
            .into_iter()
            .collect();
        Ok((!configured.is_empty()).then(|| Arc::new(Self::of(configured))))
    }

    fn of(configured: Vec<Configured>) -> Self {
        Self {
            flushers: configured.iter().map(flusher).collect(),
            callbacks: configured
                .into_iter()
                .map(|configured| configured.callback)
                .collect(),
            payloads: Arc::default(),
        }
    }

    fn fields(&self, reads: FieldSet, payload: &StandardLoggingPayload) -> FieldValues {
        let fields = reads
            .iter()
            .filter(|field| !field.by_reference())
            .filter_map(|field| Some((field, fields::derived(field, payload)?)))
            .fold(FieldValues::new(reads), |fields, (field, value)| {
                fields.with(field, value)
            });
        if !reads.contains(Field::StandardLoggingPayloadRef) {
            return fields;
        }
        fields.with_payload(PayloadRef::new(
            self.payloads.hold(payload.clone()),
            self.payloads.clone(),
        ))
    }

    /// Flushes what every callback still has queued, for a gateway that is shutting down.
    pub(crate) async fn close(&self) {
        for flusher in &self.flushers {
            flusher.abort();
        }
        for callback in &self.callbacks {
            callback.flush().await;
        }
    }
}

impl CallLog for Callbacks {
    fn record(&self, payload: StandardLoggingPayload) -> BoxFuture<'static, ()> {
        let inputs: Vec<(Arc<dyn ErasedCallback>, HookInput)> = self
            .callbacks
            .iter()
            .flat_map(|callback| {
                reads_when_an_attempt_ends(callback.as_ref()).map(|reads| {
                    let input = HookInput {
                        hook: Hook::AttemptEnded,
                        fields: self.fields(reads, &payload),
                        call: None,
                    };
                    (callback.clone(), input)
                })
            })
            .collect();
        Box::pin(async move {
            for (callback, input) in &inputs {
                if let Err(error) = callback.handle(input, None).await {
                    tracing::error!(
                        callback = %callback.manifest().selection.name,
                        %error,
                        "the callback did not take the call"
                    );
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use litellm_callbacks::{
        Callback, HookOutput, Manifest, RenderOptions, Selection, Subscription,
    };
    use parking_lot::Mutex;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::test_support::payload;

    struct Recording {
        manifest: Manifest,
        seen: Mutex<Vec<(Option<Value>, Option<Value>)>>,
        flushed: Mutex<u32>,
    }

    impl Recording {
        fn on(hook: Hook, reads: &[Field]) -> Arc<Self> {
            Arc::new(Self {
                manifest: Manifest {
                    selection: Selection::named("recording"),
                    subscriptions: vec![Subscription {
                        reads: FieldSet::of(reads),
                        awaited: true,
                        ..Subscription::on(hook)
                    }],
                    has_request_state: false,
                },
                seen: Mutex::default(),
                flushed: Mutex::default(),
            })
        }
    }

    impl Callback for Recording {
        type State = ();

        fn manifest(&self) -> &Manifest {
            &self.manifest
        }

        fn handle<'a>(
            &'a self,
            input: &'a HookInput,
            _state: Option<&'a ()>,
        ) -> BoxFuture<'a, HookOutput> {
            Box::pin(async move {
                let rendered = match input.fields.payload() {
                    Ok(Some(payload)) => payload
                        .render(RenderOptions::default())
                        .await
                        .ok()
                        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                        .map(|document| document["id"].clone()),
                    _ => None,
                };
                let log_id = input.fields.raw(Field::LogId).ok().flatten().cloned();
                self.seen.lock().push((log_id, rendered));
                Ok(None)
            })
        }

        fn flush(&self) -> BoxFuture<'_, ()> {
            *self.flushed.lock() += 1;
            Box::pin(async {})
        }
    }

    fn callbacks(recordings: &[Arc<Recording>]) -> Callbacks {
        Callbacks::of(
            recordings
                .iter()
                .map(|recording| Configured {
                    callback: recording.clone(),
                    flush_every: Duration::from_secs(3600),
                })
                .collect(),
        )
    }

    #[rstest]
    #[case::callbacks("  callbacks: [\"s3_v2\"]\n", true)]
    #[case::one_callback("  callbacks: s3_v2\n", true)]
    #[case::success_callback("  success_callback: [\"s3_v2\"]\n", true)]
    #[case::failure_callback("  failure_callback: [\"s3_v2\"]\n", false)]
    #[case::another_callback("  callbacks: [\"langfuse\"]\n", false)]
    #[case::none("  drop_params: true\n", false)]
    fn a_callback_is_named_by_the_callbacks_or_the_success_callbacks(
        #[case] litellm_settings: &str,
        #[case] expected: bool,
    ) {
        let config = Config::from_yaml(&format!(
            "model_list: []\nlitellm_settings:\n{litellm_settings}"
        ))
        .unwrap();
        assert_eq!(named(&config, "s3_v2"), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn each_callback_gets_only_the_fields_it_reads_when_an_attempt_ends() {
        let by_reference = Recording::on(
            Hook::AttemptEnded,
            &[Field::LogId, Field::StandardLoggingPayloadRef],
        );
        let id_only = Recording::on(Hook::AttemptEnded, &[Field::LogId]);
        let another_hook = Recording::on(Hook::LogPrepare, &[Field::LogId]);
        let callbacks = callbacks(&[by_reference.clone(), id_only.clone(), another_hook.clone()]);

        callbacks.record(payload()).await;

        assert_eq!(
            (
                by_reference.seen.lock().clone(),
                id_only.seen.lock().clone(),
                another_hook.seen.lock().clone()
            ),
            (
                vec![(Some(json!("msg_1")), Some(json!("msg_1")))],
                vec![(Some(json!("msg_1")), None)],
                vec![]
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn closing_flushes_every_callback() {
        let first = Recording::on(Hook::AttemptEnded, &[]);
        let second = Recording::on(Hook::AttemptEnded, &[]);
        let callbacks = callbacks(&[first.clone(), second.clone()]);
        tokio::task::yield_now().await;
        let before = (*first.flushed.lock(), *second.flushed.lock());

        callbacks.close().await;

        assert_eq!(
            (*first.flushed.lock(), *second.flushed.lock()),
            (before.0 + 1, before.1 + 1)
        );
    }
}
