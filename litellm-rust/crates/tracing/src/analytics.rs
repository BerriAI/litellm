use std::sync::{Arc, Mutex, RwLock};

use litellm_serde_compat::{InvalidBoolean, parse_env_bool};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Error, ExportSink, Metadata, Record, Sink};

pub const TARGET: &str = "litellm_analytics";

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyticsInputs {
    pub do_not_track: Option<String>,
    pub explicit: Option<String>,
    pub license_configured: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AnalyticsDecision {
    pub enabled: bool,
    pub invalid_setting: bool,
}

impl AnalyticsInputs {
    pub fn from_sources(lookup: impl Fn(&str) -> Option<String>, declared: bool) -> Self {
        Self {
            do_not_track: lookup("DO_NOT_TRACK"),
            explicit: lookup("LITELLM_TELEMETRY"),
            license_configured: declared || lookup("LITELLM_LICENSE").is_some(),
        }
    }

    pub fn decision(&self) -> AnalyticsDecision {
        match self.do_not_track.as_deref().map(parse_env_bool) {
            Some(Ok(true)) => {
                return AnalyticsDecision {
                    enabled: false,
                    invalid_setting: false,
                };
            }
            Some(Err(InvalidBoolean)) => {
                return AnalyticsDecision {
                    enabled: false,
                    invalid_setting: true,
                };
            }
            _ => {}
        }
        match self.explicit.as_deref().map(parse_env_bool) {
            Some(Ok(enabled)) => AnalyticsDecision {
                enabled,
                invalid_setting: false,
            },
            Some(Err(InvalidBoolean)) => AnalyticsDecision {
                enabled: false,
                invalid_setting: true,
            },
            None => AnalyticsDecision {
                enabled: !self.license_configured,
                invalid_setting: false,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticsSurface {
    PythonSdk,
    PythonGateway,
    RustSdk,
    RustGateway,
}

impl AnalyticsSurface {
    fn rust(self) -> bool {
        matches!(self, Self::RustSdk | Self::RustGateway)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticsRoute {
    #[serde(alias = "completion", alias = "acompletion")]
    ChatCompletions,
    #[serde(alias = "anthropic_messages", alias = "aanthropic_messages")]
    Messages,
    #[serde(alias = "aresponses")]
    Responses,
    #[serde(alias = "aembedding", alias = "embedding")]
    Embeddings,
    #[serde(alias = "aocr")]
    Ocr,
    #[serde(
        alias = "transcription",
        alias = "atranscription",
        alias = "audio_transcription",
        alias = "aaudio_transcription"
    )]
    AudioTranscription,
    Other,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticsEventName {
    RuntimeStarted,
    ApiUsed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyticsEvent {
    pub schema_version: u8,
    pub event: AnalyticsEventName,
    pub surface: AnalyticsSurface,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<AnalyticsRoute>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyticsProject {
    pub endpoint: String,
    pub token: String,
}

impl AnalyticsProject {
    pub fn builtin() -> Option<Self> {
        if !cfg!(feature = "posthog") {
            return None;
        }
        let token = option_env!("LITELLM_ANALYTICS_PROJECT_TOKEN")?.trim();
        if token.is_empty() {
            return None;
        }
        Some(Self {
            token: token.into(),
            endpoint: option_env!("LITELLM_ANALYTICS_ENDPOINT")
                .unwrap_or("https://us.i.posthog.com")
                .into(),
        })
    }
}

#[derive(Clone)]
struct Session {
    surface: AnalyticsSurface,
    version: String,
    sink: Option<Arc<dyn ExportSink>>,
}

#[derive(Clone, Default)]
pub struct Analytics {
    session: Arc<RwLock<Option<Session>>>,
    setup: Arc<Mutex<()>>,
}

impl Analytics {
    pub fn initialize(
        &self,
        decision: AnalyticsDecision,
        surface: AnalyticsSurface,
        version: String,
        project: Option<AnalyticsProject>,
    ) -> Result<bool, Error> {
        self.initialize_with(decision.enabled, surface, version, || {
            #[cfg(feature = "posthog")]
            if let Some(project) = project {
                return Ok(Some(Arc::new(crate::PostHogSink::for_analytics(
                    project.token,
                    project.endpoint,
                )?) as Arc<dyn ExportSink>));
            }
            #[cfg(not(feature = "posthog"))]
            let _ = project;
            Ok(None)
        })
    }

    pub fn initialize_with(
        &self,
        enabled: bool,
        surface: AnalyticsSurface,
        version: String,
        factory: impl FnOnce() -> Result<Option<Arc<dyn ExportSink>>, Error>,
    ) -> Result<bool, Error> {
        let _setup = self.setup.lock().unwrap_or_else(|error| error.into_inner());
        if self.snapshot().is_some() {
            return Ok(self.active());
        }
        let sink = if enabled { factory()? } else { None };
        let active = sink.is_some();
        *self
            .session
            .write()
            .unwrap_or_else(|error| error.into_inner()) = Some(Session {
            surface,
            version,
            sink,
        });
        if active {
            runtime_started(surface.rust());
        }
        Ok(active)
    }

    fn snapshot(&self) -> Option<Session> {
        self.session
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    pub fn initialized(&self) -> bool {
        self.snapshot().is_some()
    }

    pub fn active(&self) -> bool {
        self.snapshot()
            .is_some_and(|session| session.sink.is_some())
    }

    pub fn force_flush(&self) -> Result<(), Error> {
        self.snapshot()
            .and_then(|session| session.sink)
            .map_or(Ok(()), |sink| sink.force_flush())
    }

    pub fn shutdown(&self) -> Result<(), Error> {
        let _setup = self.setup.lock().unwrap_or_else(|error| error.into_inner());
        let session = self
            .session
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        session
            .and_then(|session| session.sink)
            .map_or(Ok(()), |sink| sink.shutdown())
    }
}

impl Sink for Analytics {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.snapshot().is_some_and(|session| {
            session.sink.is_some()
                && (metadata.target() == TARGET
                    || (session.surface.rust()
                        && metadata.is_span()
                        && metadata.target().starts_with("litellm_inference")
                        && metadata.name() == "litellm.route"))
        })
    }

    fn emit(&self, record: &Record) {
        let Some(session) = self.snapshot() else {
            return;
        };
        let Some(sink) = session.sink else {
            return;
        };
        let event = if record.metadata.target() == TARGET {
            let origin = record
                .fields
                .get("analytics.origin")
                .and_then(Value::as_str);
            if origin
                != Some(if session.surface.rust() {
                    "rust"
                } else {
                    "python"
                })
            {
                return;
            }
            match record.fields.get("analytics.event").and_then(Value::as_str) {
                Some("runtime_started") => AnalyticsEventName::RuntimeStarted,
                Some("api_used") => AnalyticsEventName::ApiUsed,
                _ => return,
            }
        } else if session.surface.rust()
            && record.metadata.is_span()
            && record.metadata.target().starts_with("litellm_inference")
            && record.metadata.name() == "litellm.route"
        {
            AnalyticsEventName::ApiUsed
        } else {
            return;
        };
        let route = (event == AnalyticsEventName::ApiUsed).then(|| {
            record
                .fields
                .get("route")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok())
                .unwrap_or(AnalyticsRoute::Other)
        });
        let fields = serde_json::to_value(AnalyticsEvent {
            schema_version: 1,
            event,
            surface: session.surface,
            version: session.version,
            route,
        });
        let Ok(Value::Object(fields)) = fields else {
            return;
        };
        sink.emit(&Record {
            metadata: record.metadata,
            message: "analytics".into(),
            fields,
        });
    }
}

pub fn runtime_started(rust: bool) {
    let fields = json!({"analytics.event": "runtime_started", "analytics.origin": if rust { "rust" } else { "python" }});
    tracing::info!(target: TARGET, diagnostic_fields = %fields, "analytics");
}

pub fn api_used(route: &str) {
    let route =
        serde_json::from_value::<AnalyticsRoute>(json!(route)).unwrap_or(AnalyticsRoute::Other);
    let fields =
        json!({"analytics.event": "api_used", "analytics.origin": "python", "route": route});
    tracing::info!(target: TARGET, diagnostic_fields = %fields, "analytics");
}
