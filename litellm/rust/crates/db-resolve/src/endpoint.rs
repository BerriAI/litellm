use std::{iter, path::PathBuf};

use litellm_db::{ClientIdentity, Endpoint, PoolLimits, SslMode};
use litellm_python_compat::number;

use crate::{
    Error, Expected, Ignored,
    location::Location,
    params::{Params, SslModeText},
};

const PRISMA_APPLICATION_NAME: &str = "litellm-prisma";
const NATIVE_APPLICATION_NAME: &str = "litellm-native";
const DEFAULT_SCHEMA: &str = "public";
const DEFAULT_STATEMENT_CACHE_SIZE: usize = 100;
const STRICT: &str = "strict";
const ENABLED: &str = "true";

/// How a host's value meets what the url pins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule<T> {
    /// The host's value replaces the url's.
    OverUrl(T),
    /// The url's value wins, and the host's is the default.
    UnderUrl(T),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub max_connections: Rule<u32>,
    pub acquire_timeout: Rule<Option<f64>>,
    pub idle_timeout: Rule<Option<f64>>,
    pub socket_timeout: Rule<Option<f64>>,
}

/// What a host sets on top of the url it connects with.
#[derive(Clone, Debug, PartialEq)]
pub struct Overlay {
    pub limits: Limits,
    pub unnamed_statements: bool,
    pub session_timeouts: Vec<(&'static str, i64)>,
    pub client_identity: Option<ClientIdentity>,
}

struct Pinned<'a> {
    key: &'static str,
    text: Option<&'a str>,
}

macro_rules! pinned {
    ($params:expr, $field:ident) => {
        Pinned {
            key: stringify!($field),
            text: $params.$field.as_deref(),
        }
    };
}

struct Tls<'a> {
    mode: SslModeText,
    strict: bool,
    root_cert: Option<&'a str>,
}

impl<'a> Tls<'a> {
    fn read(params: &'a Params) -> Self {
        let mode = params.ssl_mode();
        let verifies = matches!(mode, SslModeText::VerifyCa | SslModeText::VerifyFull);
        let libpq = verifies || params.sslrootcert.is_some();
        Self {
            mode: if verifies { SslModeText::Require } else { mode },
            strict: params
                .sslaccept
                .as_deref()
                .map_or(libpq, |accept| accept == STRICT),
            root_cert: params.sslcert.as_deref().or(params.sslrootcert.as_deref()),
        }
    }

    fn needs_probe(&self) -> bool {
        self.mode == SslModeText::Prefer && self.strict
    }

    fn ssl(&self, tls_offered: bool) -> (SslMode, Option<PathBuf>) {
        if self.mode == SslModeText::Disable || (self.needs_probe() && !tls_offered) {
            return (SslMode::Disable, None);
        }
        if self.strict {
            return (SslMode::VerifyFull, self.root_cert.map(PathBuf::from));
        }
        match self.mode {
            SslModeText::Require => (SslMode::Require, None),
            SslModeText::Disable
            | SslModeText::Prefer
            | SslModeText::VerifyCa
            | SslModeText::VerifyFull
            | SslModeText::Unknown => (SslMode::Prefer, None),
        }
    }
}

pub(crate) fn needs_tls_probe(location: &Location) -> bool {
    Tls::read(&location.params).needs_probe() && location.params.host.is_none()
}

pub(crate) fn endpoint(
    location: &Location,
    overlay: &Overlay,
    tls_offered: bool,
) -> Result<(Endpoint, Vec<Ignored>), Error> {
    let params = &location.params;
    let limits = &overlay.limits;
    let (ssl_mode, ssl_root_cert) = Tls::read(params).ssl(tls_offered);
    let (options, unsent) = session_options(params, &overlay.session_timeouts);
    let connection_limit = pinned!(params, connection_limit);
    let max_connections = match (limits.max_connections, connection_limit.text) {
        (Rule::UnderUrl(_), Some(text)) => text
            .parse::<u32>()
            .ok()
            .filter(|limit| *limit > 0)
            .ok_or_else(|| invalid(location, connection_limit.key, Expected::PositiveInteger))?,
        (Rule::OverUrl(limit) | Rule::UnderUrl(limit), _) => limit,
    };
    let unnamed = overlay.unnamed_statements || params.pgbouncer.as_deref() == Some(ENABLED);
    let cache_size = pinned!(params, statement_cache_size);
    let statement_cache_capacity = match (unnamed, cache_size.text) {
        (true, _) => 0,
        (false, Some(text)) => text
            .parse::<usize>()
            .map_err(|_| invalid(location, cache_size.key, Expected::NonNegativeInteger))?,
        (false, None) => DEFAULT_STATEMENT_CACHE_SIZE,
    };
    let uses_tls = ssl_mode != SslMode::Disable;
    let ssl_client = overlay.client_identity.clone().filter(|_| uses_tls);
    let identity = pinned!(params, sslidentity);
    let unsent_identity = (identity.text.is_some() && uses_tls && ssl_client.is_none()).then(|| Ignored {
        setting: format!("{} ?{}", location.source, identity.key),
        consequence: "the native pool sends no client certificate, so a server that requires one refuses its connections",
    });
    let options_key = pinned!(params, options).key;
    let ignored = unsent
        .into_iter()
        .map(|token| Ignored {
            setting: format!("{} {options_key} {token}", location.source),
            consequence: "only -c name=value settings can be sent at startup",
        })
        .chain(unsent_identity)
        .collect();
    let endpoint = Endpoint {
        host: location.host.clone(),
        port: location.port,
        socket: params.host.as_deref().map(PathBuf::from),
        username: location.username.clone(),
        password: location.password.clone(),
        database: location.database.clone(),
        ssl_mode,
        ssl_root_cert,
        ssl_client,
        application_name: Some(match params.application_name.as_deref() {
            Some(name) if name != PRISMA_APPLICATION_NAME => name.to_owned(),
            _ => NATIVE_APPLICATION_NAME.to_owned(),
        }),
        options,
        statement_cache_capacity,
        pool: PoolLimits {
            max_connections,
            acquire_timeout_seconds: seconds(
                location,
                pinned!(params, pool_timeout),
                limits.acquire_timeout,
            )?,
            idle_timeout_seconds: seconds(
                location,
                pinned!(params, max_idle_connection_lifetime),
                limits.idle_timeout,
            )?,
            socket_timeout_seconds: seconds(
                location,
                pinned!(params, socket_timeout),
                limits.socket_timeout,
            )?,
        },
    };
    Ok((endpoint, ignored))
}

fn invalid(location: &Location, key: &str, expected: Expected) -> Error {
    Error::Invalid {
        setting: format!("{} ?{key}", location.source),
        expected,
    }
}

fn seconds(
    location: &Location,
    pinned: Pinned<'_>,
    rule: Rule<Option<f64>>,
) -> Result<Option<f64>, Error> {
    let read = || {
        pinned
            .text
            .map(|text| {
                number::float(text).ok_or_else(|| invalid(location, pinned.key, Expected::Seconds))
            })
            .transpose()
    };
    let value = match rule {
        Rule::OverUrl(Some(value)) => Some(value),
        Rule::OverUrl(None) => read()?,
        Rule::UnderUrl(default) => read()?.or(default),
    };
    Ok(value.filter(|seconds| *seconds != 0.0))
}

fn session_options(
    params: &Params,
    timeouts: &[(&'static str, i64)],
) -> (Vec<(String, String)>, Vec<String>) {
    let search_path = params
        .schema
        .as_deref()
        .filter(|schema| *schema != DEFAULT_SCHEMA)
        .map(|schema| ("search_path".to_owned(), format!("\"{schema}\"")));
    let pinned = params.options.as_deref().unwrap_or_default();
    let (settings, unsent): (Vec<(String, String)>, Vec<String>) = match shlex::split(pinned) {
        Some(tokens) => {
            let folded = fold(&tokens);
            (
                folded.iter().filter_map(|token| setting(token)).collect(),
                folded
                    .iter()
                    .filter(|token| setting(token).is_none())
                    .cloned()
                    .collect(),
            )
        }
        None => (Vec::new(), vec![pinned.to_owned()]),
    };
    let configured: Vec<(String, String)> = timeouts
        .iter()
        .filter(|timeout| !settings.iter().any(|(name, _)| name.as_str() == timeout.0))
        .map(|(name, milliseconds)| ((*name).to_owned(), milliseconds.to_string()))
        .collect();
    let options = search_path
        .into_iter()
        .chain(settings)
        .chain(configured)
        .collect();
    (options, unsent)
}

fn fold(tokens: &[String]) -> Vec<String> {
    let previous = iter::once(None).chain(tokens.iter().map(Some));
    let following = tokens.iter().skip(1).map(Some).chain(iter::once(None));
    tokens
        .iter()
        .zip(previous)
        .zip(following)
        .filter(|((_, previous), _)| previous.is_none_or(|token| token.as_str() != "-c"))
        .map(|((token, _), following)| match following {
            Some(next) if token.as_str() == "-c" => format!("-c{next}"),
            _ => token.clone(),
        })
        .collect()
}

fn setting(token: &str) -> Option<(String, String)> {
    let rest = token
        .strip_prefix("-c")
        .or_else(|| token.strip_prefix("--"))?;
    let (name, value) = rest.split_once('=')?;
    Some((name.to_owned(), value.to_owned()))
}
