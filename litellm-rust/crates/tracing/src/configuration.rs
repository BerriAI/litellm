use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::{Error, ExportPolicy, Level};

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticPolicy {
    #[serde(
        serialize_with = "serialize_level",
        deserialize_with = "deserialize_level"
    )]
    pub minimum_level: Level,
    pub target_prefixes: Vec<String>,
    pub sample_rate: f64,
}

impl Default for DiagnosticPolicy {
    fn default() -> Self {
        Self {
            minimum_level: Level::INFO,
            target_prefixes: vec![],
            sample_rate: 1.0,
        }
    }
}

impl DiagnosticPolicy {
    pub(crate) fn build(&self) -> Result<ExportPolicy, Error> {
        ExportPolicy::new(
            self.minimum_level,
            self.target_prefixes.clone(),
            self.sample_rate,
        )
    }
}

fn serialize_level<S: Serializer>(level: &Level, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(level.as_str())
}

fn deserialize_level<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Level, D::Error> {
    let value = String::deserialize(deserializer)?;
    value
        .parse::<Level>()
        .ok()
        .filter(|level| level.as_str() == value)
        .ok_or_else(|| serde::de::Error::custom("expected TRACE, DEBUG, INFO, WARN, or ERROR"))
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "transport", rename_all = "snake_case", deny_unknown_fields)]
pub enum DestinationConfig {
    Otlp {
        name: String,
        endpoint: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
        #[serde(default)]
        policy: Option<DiagnosticPolicy>,
    },
    Posthog {
        name: String,
        api_key: String,
        #[serde(default = "posthog_endpoint")]
        endpoint: String,
        #[serde(default)]
        policy: Option<DiagnosticPolicy>,
    },
}

fn posthog_endpoint() -> String {
    "https://us.i.posthog.com".into()
}

impl DestinationConfig {
    pub fn name(&self) -> &str {
        match self {
            Self::Otlp { name, .. } | Self::Posthog { name, .. } => name,
        }
    }

    pub(crate) fn policy(&self, defaults: &DiagnosticPolicy) -> Result<ExportPolicy, Error> {
        match self {
            Self::Otlp { policy, .. } | Self::Posthog { policy, .. } => {
                policy.as_ref().unwrap_or(defaults).build()
            }
        }
    }

    fn resolve(self, lookup: &impl Fn(&str) -> Option<String>) -> Result<Self, Error> {
        match self {
            Self::Otlp {
                name,
                endpoint,
                headers,
                policy,
            } => Ok(Self::Otlp {
                name,
                endpoint: required(endpoint, "endpoint", lookup)?,
                headers: headers
                    .into_iter()
                    .map(|(name, value)| required(value, &name, lookup).map(|value| (name, value)))
                    .collect::<Result<_, _>>()?,
                policy,
            }),
            Self::Posthog {
                name,
                api_key,
                endpoint,
                policy,
            } => Ok(Self::Posthog {
                name,
                api_key: required(api_key, "api_key", lookup)?,
                endpoint: required(endpoint, "endpoint", lookup)?,
                policy,
            }),
        }
    }
}

fn required(
    value: String,
    field: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let resolved = match value.strip_prefix("os.environ/") {
        Some(name) => lookup(name).ok_or_else(|| Error::MissingValue(field.to_owned()))?,
        None => value,
    };
    if resolved.trim().is_empty() {
        return Err(Error::MissingValue(field.to_owned()));
    }
    Ok(resolved)
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticsConfig {
    pub enabled: bool,
    pub payload_shapes: bool,
    pub service_name: String,
    pub policy: DiagnosticPolicy,
    pub destinations: Vec<DestinationConfig>,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            payload_shapes: false,
            service_name: "litellm".into(),
            policy: DiagnosticPolicy::default(),
            destinations: vec![],
        }
    }
}

impl DiagnosticsConfig {
    pub fn from_sources(
        settings: Option<Value>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, Error> {
        let config: Self = match lookup("LITELLM_DIAGNOSTICS") {
            Some(value) => serde_json::from_str(&value)?,
            None => serde_json::from_value(settings.unwrap_or_else(|| serde_json::json!({})))?,
        };
        config.resolve(&lookup)
    }

    pub fn resolve(self, lookup: &impl Fn(&str) -> Option<String>) -> Result<Self, Error> {
        self.validate()?;
        if !self.enabled {
            return Ok(self);
        }
        Ok(Self {
            service_name: required(self.service_name, "service_name", lookup)?,
            destinations: self
                .destinations
                .into_iter()
                .map(|destination| destination.resolve(lookup))
                .collect::<Result<_, _>>()?,
            ..self
        })
    }

    pub fn validate(&self) -> Result<(), Error> {
        self.policy.build()?;
        let names: BTreeSet<_> = self
            .destinations
            .iter()
            .map(DestinationConfig::name)
            .collect();
        if names.len() != self.destinations.len() || names.iter().any(|name| name.trim().is_empty())
        {
            return Err(Error::DestinationName);
        }
        for destination in &self.destinations {
            destination.policy(&self.policy)?;
            if self.enabled
                && matches!(destination, DestinationConfig::Posthog { .. })
                && !cfg!(feature = "posthog")
            {
                return Err(Error::UnavailableTransport);
            }
            match destination {
                DestinationConfig::Otlp {
                    endpoint, headers, ..
                } => {
                    nonempty(endpoint, "endpoint")?;
                    for (name, value) in headers {
                        nonempty(value, name)?;
                    }
                }
                DestinationConfig::Posthog {
                    endpoint, api_key, ..
                } => {
                    nonempty(endpoint, "endpoint")?;
                    nonempty(api_key, "api_key")?;
                }
            }
        }
        nonempty(&self.service_name, "service_name")
    }
}

fn nonempty(value: &str, field: &str) -> Result<(), Error> {
    if value.trim().is_empty() {
        return Err(Error::MissingValue(field.to_owned()));
    }
    Ok(())
}
