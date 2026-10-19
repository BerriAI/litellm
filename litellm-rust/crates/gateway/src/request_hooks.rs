//! Built-in hooks a config turns on and configures under `litellm_settings.hooks`, run in the
//! order listed:
//!
//! ```yaml
//! litellm_settings:
//!   hooks:
//!     - forward_headers:
//!         headers: ["x-request-origin", "x-trace-*"]
//!     - spend_metadata:
//!         from_headers:
//!           x-litellm-spend-logs-metadata: json
//!           x-cost-center: cost_center
//!         from_key_metadata: [cost_center]
//!     - strip_headers:
//!         headers: ["x-litellm-spend-logs-metadata", "x-cost-center"]
//!     - provider_metadata:
//!         body:
//!           metadata.user_id: "{end_user|key.user_id}"
//!         headers:
//!           x-caller-team: "{key.team_alias}"
//! ```

use std::sync::Arc;

use futures_util::future::BoxFuture;
use http::{HeaderName, HeaderValue};
use litellm_config::Config;
use litellm_gateway_hooks::{
    Facts, HeaderPattern, Hook, Refusal, RequestEdit, RequestFacts, Route, Settle, Targets,
};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{Error, spend::SPEND_LOGS_METADATA};

/// Headers a hook never forwards: credentials, and those the HTTP client or a request signer
/// sets itself.
const PROTECTED: [&str; 13] = [
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-litellm-api-key",
    "x-goog-api-key",
    "cookie",
    "host",
    "content-length",
    "content-type",
    "transfer-encoding",
    "connection",
    "date",
];

fn protected(name: &str) -> bool {
    PROTECTED
        .iter()
        .any(|protected| protected.eq_ignore_ascii_case(name))
        || name
            .get(..6)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("x-amz-"))
}

/// The hooks `litellm_settings.hooks` lists, in its order.
pub fn from_config(config: &Config) -> Result<Vec<Arc<dyn Hook>>, Error> {
    let Some(entries) = config.litellm_settings.additional_fields.get("hooks") else {
        return Ok(Vec::new());
    };
    let litellm_config::Value::Sequence(entries) = entries else {
        return Err(Error::Hooks("must be a list".into()));
    };
    entries.iter().map(hook).collect()
}

fn hook(entry: &litellm_config::Value) -> Result<Arc<dyn Hook>, Error> {
    let (name, settings) = match entry {
        litellm_config::Value::Mapping(mapping) if mapping.len() == 1 => {
            let (name, settings) = mapping.iter().next().expect("one entry");
            (name.as_str().unwrap_or_default(), settings.clone())
        }
        litellm_config::Value::String(name) => (name.as_str(), litellm_config::Value::Null),
        _ => {
            return Err(Error::Hooks(
                "each entry is a hook name with its settings".into(),
            ));
        }
    };
    let built: Result<Arc<dyn Hook>, String> = match name {
        "forward_headers" => parse(settings)
            .and_then(patterns)
            .map(|patterns| Arc::new(ForwardHeaders { patterns }) as Arc<dyn Hook>),
        "strip_headers" => parse(settings)
            .and_then(patterns)
            .map(|patterns| Arc::new(StripHeaders { patterns }) as Arc<dyn Hook>),
        "spend_metadata" => parse(settings)
            .and_then(SpendMetadata::new)
            .map(|hook| Arc::new(hook) as Arc<dyn Hook>),
        "provider_metadata" => parse(settings)
            .and_then(ProviderMetadata::new)
            .map(|hook| Arc::new(hook) as Arc<dyn Hook>),
        other => return Err(Error::Hooks(format!("unknown hook {other}"))),
    };
    built.map_err(|error| Error::Hooks(format!("{name}: {error}")))
}

fn parse<T: serde::de::DeserializeOwned>(settings: litellm_config::Value) -> Result<T, String> {
    T::deserialize(settings).map_err(|error| error.to_string())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderList {
    headers: Vec<String>,
}

fn patterns(list: HeaderList) -> Result<Vec<HeaderPattern>, String> {
    list.headers
        .iter()
        .map(|pattern| {
            HeaderPattern::parse(pattern).map_err(|_| format!("{pattern} is not a header name"))
        })
        .collect()
}

/// `forward_headers`: copies the client headers it names onto the provider request.
struct ForwardHeaders {
    patterns: Vec<HeaderPattern>,
}

impl Hook for ForwardHeaders {
    fn name(&self) -> &'static str {
        "forward_headers"
    }

    fn reads(&self) -> Facts {
        Facts::CLIENT_HEADERS
    }

    fn writes(&self) -> Targets {
        Targets::UPSTREAM_HEADERS
    }

    fn run<'a>(
        &'a self,
        request: &'a Arc<RequestFacts>,
        edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
        for (name, value) in request.client_headers.iter() {
            if !protected(name.as_str())
                && self
                    .patterns
                    .iter()
                    .any(|pattern| pattern.matches(name.as_str()))
            {
                edit.set_upstream_header(name.clone(), value.clone());
            }
        }
        Box::pin(async { Ok(None) })
    }
}

/// `strip_headers`: removes the headers it names from the provider request, whether a hook set
/// them or the gateway forwards them by default.
struct StripHeaders {
    patterns: Vec<HeaderPattern>,
}

impl Hook for StripHeaders {
    fn name(&self) -> &'static str {
        "strip_headers"
    }

    fn reads(&self) -> Facts {
        Facts::NONE
    }

    fn writes(&self) -> Targets {
        Targets::UPSTREAM_HEADERS
    }

    fn run<'a>(
        &'a self,
        _request: &'a Arc<RequestFacts>,
        edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
        for pattern in &self.patterns {
            edit.remove_upstream_headers(pattern.clone());
        }
        Box::pin(async { Ok(None) })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpendMetadataSettings {
    /// Header name to `json`, merged as an object, or to the key it is stored under.
    #[serde(default)]
    from_headers: litellm_config::Value,
    /// Fields of the key's metadata copied under the same names.
    #[serde(default)]
    from_key_metadata: Vec<String>,
}

enum HeaderRule {
    Json,
    Field(String),
}

/// `spend_metadata`: adds caller-supplied values to the request's `spend_logs_metadata`
/// without changing the request.
struct SpendMetadata {
    headers: Vec<(HeaderName, HeaderRule)>,
    key_fields: Vec<String>,
}

impl SpendMetadata {
    fn new(settings: SpendMetadataSettings) -> Result<Self, String> {
        let headers = match settings.from_headers {
            litellm_config::Value::Null => Vec::new(),
            litellm_config::Value::Mapping(mapping) => mapping
                .iter()
                .map(|(name, rule)| {
                    let (Some(name), Some(rule)) = (name.as_str(), rule.as_str()) else {
                        return Err("from_headers maps header names to json or a key".to_owned());
                    };
                    let name = HeaderName::from_bytes(name.as_bytes())
                        .map_err(|_| format!("{name} is not a header name"))?;
                    let rule = match rule {
                        "json" => HeaderRule::Json,
                        field => HeaderRule::Field(field.to_owned()),
                    };
                    Ok((name, rule))
                })
                .collect::<Result<_, _>>()?,
            _ => return Err("from_headers maps header names to json or a key".into()),
        };
        Ok(Self {
            headers,
            key_fields: settings.from_key_metadata,
        })
    }
}

impl Hook for SpendMetadata {
    fn name(&self) -> &'static str {
        "spend_metadata"
    }

    fn reads(&self) -> Facts {
        Facts::CLIENT_HEADERS | Facts::KEY
    }

    fn writes(&self) -> Targets {
        Targets::ANNOTATIONS
    }

    fn run<'a>(
        &'a self,
        request: &'a Arc<RequestFacts>,
        edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
        let mut metadata = Map::new();
        for (name, rule) in &self.headers {
            let Some(value) = request
                .client_headers
                .get(name)
                .and_then(|value| value.to_str().ok())
            else {
                continue;
            };
            match rule {
                HeaderRule::Json => match serde_json::from_str::<Value>(value) {
                    Ok(Value::Object(fields)) => metadata.extend(fields),
                    _ => tracing::warn!(
                        header = name.as_str(),
                        "spend metadata header is not a JSON object; ignored"
                    ),
                },
                HeaderRule::Field(field) => {
                    metadata.insert(field.clone(), Value::from(value));
                }
            }
        }
        // The key's own values win over what the caller sent.
        if let Some(key) = request.key.as_deref() {
            for field in &self.key_fields {
                if let Some(value) = key.metadata.get(field) {
                    metadata.insert(field.clone(), value.clone());
                }
            }
        }
        if !metadata.is_empty() {
            edit.annotate(SPEND_LOGS_METADATA, Value::Object(metadata));
        }
        Box::pin(async { Ok(None) })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderMetadataSettings {
    /// Dotted body paths, such as `metadata.user_id`, to templates.
    #[serde(default)]
    body: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    headers: std::collections::BTreeMap<String, String>,
}

/// A fact a template can name.
#[derive(Clone, Debug, PartialEq)]
enum Fact {
    EndUser,
    Model,
    RequestId,
    KeyAlias,
    KeyUserId,
    KeyTeamId,
    KeyTeamAlias,
    KeyMetadata(String),
    TeamMetadata(String),
    Header(HeaderName),
}

impl Fact {
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "end_user" => Self::EndUser,
            "model" => Self::Model,
            "request_id" => Self::RequestId,
            "key.alias" => Self::KeyAlias,
            "key.user_id" => Self::KeyUserId,
            "key.team_id" => Self::KeyTeamId,
            "key.team_alias" => Self::KeyTeamAlias,
            other => {
                if let Some(field) = other.strip_prefix("key.metadata.") {
                    Self::KeyMetadata(field.to_owned())
                } else if let Some(field) = other.strip_prefix("team.metadata.") {
                    Self::TeamMetadata(field.to_owned())
                } else if let Some(header) = other.strip_prefix("header.") {
                    Self::Header(
                        HeaderName::from_bytes(header.as_bytes())
                            .map_err(|_| format!("{header} is not a header name"))?,
                    )
                } else {
                    return Err(format!("unknown fact {other}"));
                }
            }
        })
    }

    fn value(&self, request: &RequestFacts) -> Option<String> {
        let key = request.key.as_deref();
        let scalar = |value: &Value| match value {
            Value::String(text) => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            Value::Bool(flag) => Some(flag.to_string()),
            _ => None,
        };
        match self {
            Self::EndUser => request.end_user.as_deref().map(str::to_owned),
            Self::Model => Some(request.requested_model.to_string()),
            Self::RequestId => Some(request.request_id.to_string()),
            Self::KeyAlias => key?.alias.clone(),
            Self::KeyUserId => key?.user_id.clone(),
            Self::KeyTeamId => key?.team_id.clone(),
            Self::KeyTeamAlias => key?.team_alias.clone(),
            Self::KeyMetadata(field) => key?.metadata.get(field).and_then(scalar),
            Self::TeamMetadata(field) => key?.team_metadata.get(field).and_then(scalar),
            Self::Header(name) => request
                .client_headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
        }
        .filter(|value| !value.is_empty())
    }
}

#[derive(Debug, PartialEq)]
enum Part {
    Text(String),
    /// The first of these facts that is set.
    Facts(Vec<Fact>),
}

/// Text with `{fact|fallback}` placeholders, parsed once. A placeholder none of whose facts are
/// set leaves the whole value unset.
#[derive(Debug, PartialEq)]
struct Template(Vec<Part>);

impl Template {
    fn parse(source: &str) -> Result<Self, String> {
        let mut parts = Vec::new();
        let mut rest = source;
        while let Some(open) = rest.find('{') {
            if open > 0 {
                parts.push(Part::Text(rest[..open].to_owned()));
            }
            let close = rest[open..]
                .find('}')
                .ok_or_else(|| format!("unclosed placeholder in {source}"))?;
            let facts = rest[open + 1..open + close]
                .split('|')
                .map(|fact| Fact::parse(fact.trim()))
                .collect::<Result<_, _>>()?;
            parts.push(Part::Facts(facts));
            rest = &rest[open + close + 1..];
        }
        if !rest.is_empty() {
            parts.push(Part::Text(rest.to_owned()));
        }
        Ok(Self(parts))
    }

    fn reads(&self) -> Facts {
        let mut reads = Facts::NONE;
        for part in &self.0 {
            let Part::Facts(facts) = part else { continue };
            for fact in facts {
                reads |= match fact {
                    Fact::EndUser => Facts::END_USER,
                    Fact::Header(_) => Facts::CLIENT_HEADERS,
                    Fact::Model | Fact::RequestId => Facts::NONE,
                    _ => Facts::KEY,
                };
            }
        }
        reads
    }

    fn render(&self, request: &RequestFacts) -> Option<String> {
        let mut rendered = String::new();
        for part in &self.0 {
            match part {
                Part::Text(text) => rendered.push_str(text),
                Part::Facts(facts) => {
                    rendered.push_str(&facts.iter().find_map(|fact| fact.value(request))?)
                }
            }
        }
        Some(rendered)
    }
}

/// `provider_metadata`: sets body fields and headers on the provider request from facts about
/// the caller. Body fields are set on `/v1/messages` requests only.
struct ProviderMetadata {
    body: Vec<(Vec<String>, Template)>,
    headers: Vec<(HeaderName, Template)>,
    reads: Facts,
}

impl ProviderMetadata {
    fn new(settings: ProviderMetadataSettings) -> Result<Self, String> {
        let body = settings
            .body
            .iter()
            .map(|(path, template)| {
                let path: Vec<String> = path.split('.').map(str::to_owned).collect();
                if path.iter().any(String::is_empty) {
                    return Err(format!("{} is not a body path", path.join(".")));
                }
                Ok((path, Template::parse(template)?))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let headers = settings
            .headers
            .iter()
            .map(|(name, template)| {
                let header = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| format!("{name} is not a header name"))?;
                if protected(header.as_str()) {
                    return Err(format!("{name} cannot be set by a hook"));
                }
                Ok((header, Template::parse(template)?))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let reads = body
            .iter()
            .map(|(_, template)| template)
            .chain(headers.iter().map(|(_, template)| template))
            .fold(Facts::NONE, |reads, template| reads | template.reads());
        Ok(Self {
            body,
            headers,
            reads,
        })
    }
}

impl Hook for ProviderMetadata {
    fn name(&self) -> &'static str {
        "provider_metadata"
    }

    fn reads(&self) -> Facts {
        self.reads
    }

    fn writes(&self) -> Targets {
        Targets::UPSTREAM_BODY | Targets::UPSTREAM_HEADERS
    }

    fn run<'a>(
        &'a self,
        request: &'a Arc<RequestFacts>,
        edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
        // Body paths name Messages API fields; Chat Completions takes the headers only.
        let body = match request.route {
            Route::Messages => self.body.as_slice(),
            Route::ChatCompletions => &[],
        };
        for (path, template) in body {
            if let Some(value) = template.render(request) {
                edit.set_upstream_body_path(path.clone(), Value::from(value));
            }
        }
        for (name, template) in &self.headers {
            if let Some(value) = template
                .render(request)
                .and_then(|value| HeaderValue::from_str(&value).ok())
            {
                edit.set_upstream_header(name.clone(), value);
            }
        }
        Box::pin(async { Ok(None) })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn templates_name_facts_with_fallbacks() {
        assert_eq!(
            Template::parse("team-{key.team_alias|key.team_id}").unwrap(),
            Template(vec![
                Part::Text("team-".into()),
                Part::Facts(vec![Fact::KeyTeamAlias, Fact::KeyTeamId]),
            ])
        );
        assert!(Template::parse("{nope}").is_err());
        assert!(Template::parse("{end_user").is_err());
    }

    #[rstest]
    #[case::credential("Authorization", true)]
    #[case::signed("X-Amz-Date", true)]
    #[case::trace("x-trace-id", false)]
    fn credentials_are_never_forwarded(#[case] name: &str, #[case] expected: bool) {
        assert_eq!(protected(name), expected);
    }
}
