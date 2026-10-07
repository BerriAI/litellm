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
    analytics: bool,
}

impl PostHogSink {
    pub fn new(
        api_key: String,
        host: String,
        service_name: String,
        policy: ExportPolicy,
    ) -> Result<Self, Error> {
        Self::build(api_key, host, service_name, policy, false)
    }

    fn build(
        api_key: String,
        host: String,
        service_name: String,
        policy: ExportPolicy,
        analytics: bool,
    ) -> Result<Self, Error> {
        let mut builder = ClientOptionsBuilder::default();
        builder
            .api_key(api_key)
            .host(host)
            .request_timeout_seconds(5)
            .shutdown_timeout_ms(5_000)
            .max_queue_size(2_048)
            .disable_geoip(true)
            .is_server(true);
        if analytics {
            builder.before_send(|mut event| {
                let excluded = event
                    .properties()
                    .keys()
                    .filter(|key| {
                        !matches!(
                            key.as_str(),
                            "schema_version"
                                | "event"
                                | "surface"
                                | "version"
                                | "route"
                                | "$process_person_profile"
                                | "$geoip_disable"
                                | "$is_server"
                        )
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                for key in excluded {
                    event.remove_prop(&key);
                }
                Some(event)
            });
        }
        let options = builder.build()?;
        Ok(Self {
            client: posthog_rs::client(options),
            service_name,
            policy,
            processor: Processor::new(16),
            analytics,
        })
    }
    pub fn for_analytics(api_key: String, host: String) -> Result<Self, Error> {
        Self::build(
            api_key,
            host,
            format!("{:032x}", rand::random::<u128>()),
            ExportPolicy::new(tracing::Level::INFO, vec![], 1.0)?,
            true,
        )
    }

    fn capture_analytics(&self, record: &Record) {
        let Ok(projected) = serde_json::from_value::<crate::analytics::AnalyticsEvent>(
            Value::Object(record.fields.clone()),
        ) else {
            return;
        };
        if projected.schema_version != 1
            || projected.version.len() > 64
            || !projected
                .version
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b".-+_".contains(&byte))
        {
            return;
        }
        let name = match projected.event {
            crate::analytics::AnalyticsEventName::RuntimeStarted => "litellm.runtime.started",
            crate::analytics::AnalyticsEventName::ApiUsed => "litellm.api.used",
        };
        let mut event = Event::new(name.to_owned(), self.service_name.clone());
        let Ok(Value::Object(properties)) = serde_json::to_value(projected) else {
            return;
        };
        for (key, value) in properties {
            if event.insert_prop(key, value).is_err() {
                return;
            }
        }
        if event.insert_prop("$process_person_profile", false).is_err() {
            return;
        }
        self.client.capture(event);
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
        if self.analytics {
            self.capture_analytics(record);
            return;
        }
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
