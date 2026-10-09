use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock},
};

use crate::{
    DestinationConfig, DiagnosticsConfig, Error, ExportSink, Logger, Metadata, OtlpSink, Record,
    Sink,
};

#[derive(Clone, Default)]
pub struct Diagnostics {
    sinks: Arc<RwLock<BTreeMap<String, Arc<dyn ExportSink>>>>,
    analytics: crate::analytics::Analytics,
}

impl Diagnostics {
    pub fn analytics(&self) -> &crate::analytics::Analytics {
        &self.analytics
    }

    fn snapshot(&self) -> Vec<Arc<dyn ExportSink>> {
        self.sinks
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .values()
            .cloned()
            .collect()
    }

    pub fn active(&self) -> bool {
        !self
            .sinks
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .is_empty()
    }

    pub fn logger(&self, compatibility: impl Sink) -> Logger {
        Logger::new(Fanout {
            exports: self.clone(),
            compatibility,
        })
    }

    pub fn configure(&self, config: DiagnosticsConfig) -> Result<(), Error> {
        config.validate()?;
        let sinks = if config.enabled {
            config
                .destinations
                .iter()
                .map(|destination| {
                    let policy = destination.policy(&config.policy)?;
                    let sink: Arc<dyn ExportSink> = match destination {
                        DestinationConfig::Otlp {
                            endpoint, headers, ..
                        } => Arc::new(OtlpSink::new(
                            endpoint.clone(),
                            headers.clone().into_iter().collect(),
                            config.service_name.clone(),
                            policy,
                        )?),
                        DestinationConfig::Posthog {
                            api_key, endpoint, ..
                        } => {
                            #[cfg(feature = "posthog")]
                            {
                                Arc::new(crate::PostHogSink::new(
                                    api_key.clone(),
                                    endpoint.clone(),
                                    config.service_name.clone(),
                                    policy,
                                )?)
                            }
                            #[cfg(not(feature = "posthog"))]
                            {
                                let _ = (api_key, endpoint, policy);
                                return Err(Error::UnavailableTransport);
                            }
                        }
                    };
                    Ok((destination.name().to_owned(), sink))
                })
                .collect::<Result<BTreeMap<_, _>, Error>>()?
        } else {
            BTreeMap::new()
        };
        let previous = std::mem::replace(
            &mut *self
                .sinks
                .write()
                .unwrap_or_else(|error| error.into_inner()),
            sinks,
        );
        drain(previous.values().map(|sink| sink.shutdown()))
    }

    pub fn force_flush(&self) -> Result<(), Error> {
        drain(self.snapshot().iter().map(|sink| sink.force_flush()))
    }

    pub fn shutdown(&self) -> Result<(), Error> {
        let sinks = std::mem::take(
            &mut *self
                .sinks
                .write()
                .unwrap_or_else(|error| error.into_inner()),
        );
        drain(sinks.values().map(|sink| sink.shutdown()))
    }
}

fn drain(results: impl Iterator<Item = Result<(), Error>>) -> Result<(), Error> {
    let mut failure = None;
    for result in results {
        if let Err(error) = result {
            failure.get_or_insert(error);
        }
    }
    failure.map_or(Ok(()), Err)
}

impl Sink for Diagnostics {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        if metadata.target() == crate::analytics::TARGET {
            return self.analytics.enabled(metadata);
        }
        if self.analytics.enabled(metadata) {
            return true;
        }

        self.snapshot().iter().any(|sink| sink.enabled(metadata))
    }

    fn emit(&self, record: &Record) {
        self.analytics.emit(record);
        if record.metadata.target() == crate::analytics::TARGET {
            return;
        }
        for sink in self.snapshot() {
            sink.emit(record);
        }
    }
}

struct Fanout<S> {
    exports: Diagnostics,
    compatibility: S,
}

impl<S: Sink> Sink for Fanout<S> {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.exports.enabled(metadata)
            || (metadata.target() != crate::analytics::TARGET
                && self.compatibility.enabled(metadata))
    }

    fn emit(&self, record: &Record) {
        self.exports.emit(record);
        if record.metadata.target() != crate::analytics::TARGET
            && self.compatibility.enabled(record.metadata)
        {
            self.compatibility.emit(record);
        }
    }
}
