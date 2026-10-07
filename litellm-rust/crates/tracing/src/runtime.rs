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
    state: Arc<RwLock<State>>,
}

#[derive(Default)]
struct State {
    sinks: BTreeMap<String, Arc<dyn ExportSink>>,
    payload_shapes: bool,
}

impl Diagnostics {
    fn snapshot(&self) -> Vec<Arc<dyn ExportSink>> {
        self.state
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .sinks
            .values()
            .cloned()
            .collect()
    }

    pub fn active(&self) -> bool {
        !self
            .state
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .sinks
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
                .state
                .write()
                .unwrap_or_else(|error| error.into_inner()),
            State {
                sinks,
                payload_shapes: config.payload_shapes,
            },
        );
        drain(previous.sinks.values().map(|sink| sink.shutdown()))
    }

    pub fn force_flush(&self) -> Result<(), Error> {
        drain(self.snapshot().iter().map(|sink| sink.force_flush()))
    }

    pub fn shutdown(&self) -> Result<(), Error> {
        let sinks = std::mem::take(
            &mut *self
                .state
                .write()
                .unwrap_or_else(|error| error.into_inner()),
        );
        drain(sinks.sinks.values().map(|sink| sink.shutdown()))
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
        if metadata.target() == crate::payload::TARGET
            && !self
                .state
                .read()
                .unwrap_or_else(|error| error.into_inner())
                .payload_shapes
        {
            return false;
        }
        self.snapshot().iter().any(|sink| sink.enabled(metadata))
    }

    fn emit(&self, record: &Record) {
        if record.metadata.target() == crate::payload::TARGET && !self.enabled(record.metadata) {
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
            || (metadata.target() != crate::payload::TARGET && self.compatibility.enabled(metadata))
    }

    fn emit(&self, record: &Record) {
        self.exports.emit(record);
        if record.metadata.target() != crate::payload::TARGET
            && self.compatibility.enabled(record.metadata)
        {
            self.compatibility.emit(record);
        }
    }
}
