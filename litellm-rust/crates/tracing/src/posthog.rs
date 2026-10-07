use posthog_rs::{Client, ClientOptionsBuilder, Event};
use serde_json::Value;

use crate::{
    Error, ExportPolicy, ExportSink, Metadata, Processor, Record, Sink,
    export::{scrub, source_timestamp},
};

pub struct PostHogSink {
    client: Client,
    service_name: String,
    policy: ExportPolicy,
    processor: Processor,
}

impl PostHogSink {
    pub fn new(
        api_key: String,
        host: String,
        service_name: String,
        policy: ExportPolicy,
    ) -> Result<Self, Error> {
        let options = ClientOptionsBuilder::default()
            .api_key(api_key)
            .host(host)
            .request_timeout_seconds(5)
            .shutdown_timeout_ms(5_000)
            .max_queue_size(2_048)
            .disable_geoip(true)
            .build()?;
        Ok(Self {
            client: posthog_rs::client(options),
            service_name,
            policy,
            processor: Processor::new(16),
        })
    }
}

impl ExportSink for PostHogSink {
    fn force_flush(&self) -> Result<(), Error> {
        self.client.flush();
        Ok(())
    }

    fn shutdown(&self) -> Result<(), Error> {
        self.client.shutdown();
        Ok(())
    }
}

impl Sink for PostHogSink {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.policy.minimum_level_accepts(metadata)
    }

    fn emit(&self, record: &Record) {
        if !self.policy.accepts(record) {
            return;
        }
        let Ok(message) = self.processor.redact_text(&record.message) else {
            return;
        };
        let Ok(fields) = scrub(&self.processor, None, &Value::Object(record.fields.clone())) else {
            return;
        };
        let mut event = Event::new(
            if record.metadata.target() == crate::payload::TARGET {
                "llm.payload.shape"
            } else {
                "litellm diagnostic"
            }
            .to_owned(),
            self.service_name.clone(),
        );
        let timestamp = source_timestamp(record)
            .and_then(|timestamp| timestamp.duration_since(std::time::UNIX_EPOCH).ok())
            .and_then(|duration| {
                chrono::DateTime::<chrono::Utc>::from_timestamp(
                    duration.as_secs().try_into().ok()?,
                    duration.subsec_nanos(),
                )
            });
        if let Some(timestamp) = timestamp {
            let _ = event.set_timestamp(timestamp);
        }
        let properties = [
            ("$process_person_profile", Value::Bool(false)),
            ("service_name", self.service_name.clone().into()),
            ("message", message.into()),
            ("level", record.metadata.level().to_string().into()),
            ("target", record.metadata.target().to_owned().into()),
            ("fields", fields),
        ];
        for (key, value) in properties {
            if event.insert_prop(key, value).is_err() {
                return;
            }
        }
        self.client.capture(event);
    }
}
