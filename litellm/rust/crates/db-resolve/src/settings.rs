use litellm_config::{GeneralSettings, Spelled};
use litellm_env::Lookup;
use litellm_python_compat::{Value as PythonValue, number, pydantic, unicode::isspace};

use crate::{
    Error, Expected,
    endpoint::{Limits, Overlay, Rule},
};

const DEFAULT_CONNECTION_LIMIT: u64 = 10;
const DEFAULT_POOL_TIMEOUT: f64 = 60.0;
const DEFAULT_IDLE_LIFETIME: f64 = 60.0;
const DISABLE_PREPARED_STATEMENTS: &str = "DATABASE_DISABLE_PREPARED_STATEMENTS";

struct Configured<'a, T> {
    setting: &'static str,
    value: Option<&'a T>,
}

macro_rules! configured {
    ($general:expr, $field:ident) => {
        Configured {
            setting: concat!("general_settings.", stringify!($field)),
            value: $general.$field.as_ref(),
        }
    };
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PoolSettings {
    pub connection_limit: u32,
    pub pool_timeout: f64,
    pub socket_timeout: Option<f64>,
    pub idle_lifetime: f64,
    pub unnamed_statements: bool,
    pub session_timeouts: Vec<(&'static str, i64)>,
}

impl PoolSettings {
    pub(crate) fn read(general: &GeneralSettings, environment: &dyn Lookup) -> Result<Self, Error> {
        let connection_limit = configured!(general, database_connection_pool_limit);
        let pool_timeout = match general.database_connection_timeout {
            Some(seconds) => seconds,
            None => seconds(configured!(general, database_connection_pool_timeout))?
                .unwrap_or(DEFAULT_POOL_TIMEOUT),
        };
        let session_timeouts = [
            (
                "statement_timeout",
                configured!(general, database_statement_timeout),
            ),
            ("lock_timeout", configured!(general, database_lock_timeout)),
            (
                "idle_in_transaction_session_timeout",
                configured!(general, database_idle_in_transaction_session_timeout),
            ),
        ]
        .into_iter()
        .map(
            |(name, configured)| -> Result<Option<(&'static str, i64)>, Error> {
                Ok(seconds(configured)?.map(|value| (name, (value * 1000.0) as i64)))
            },
        )
        .filter_map(Result::transpose)
        .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            connection_limit: u32::try_from(
                connection_limit
                    .value
                    .copied()
                    .unwrap_or(DEFAULT_CONNECTION_LIMIT),
            )
            .ok()
            .filter(|limit| *limit > 0)
            .ok_or(Error::Invalid {
                setting: connection_limit.setting.to_owned(),
                expected: Expected::PositiveInteger,
            })?,
            pool_timeout,
            socket_timeout: general.database_socket_timeout,
            idle_lifetime: general
                .database_max_idle_connection_lifetime
                .unwrap_or(DEFAULT_IDLE_LIFETIME),
            unnamed_statements: general.database_disable_prepared_statements
                || flag(
                    DISABLE_PREPARED_STATEMENTS,
                    environment.get(DISABLE_PREPARED_STATEMENTS).as_deref(),
                )?,
            session_timeouts,
        })
    }
}

impl PoolSettings {
    pub(crate) fn writer(&self) -> Overlay {
        self.overlay(Limits {
            max_connections: Rule::OverUrl(self.connection_limit),
            acquire_timeout: Rule::OverUrl(Some(self.pool_timeout)),
            idle_timeout: Rule::UnderUrl(Some(self.idle_lifetime)),
            socket_timeout: Rule::OverUrl(self.socket_timeout),
        })
    }

    pub(crate) fn replica(&self) -> Overlay {
        self.overlay(Limits {
            max_connections: Rule::UnderUrl(self.connection_limit),
            acquire_timeout: Rule::UnderUrl(Some(self.pool_timeout)),
            idle_timeout: Rule::UnderUrl(Some(self.idle_lifetime)),
            socket_timeout: Rule::UnderUrl(self.socket_timeout),
        })
    }

    fn overlay(&self, limits: Limits) -> Overlay {
        Overlay {
            limits,
            unnamed_statements: self.unnamed_statements,
            session_timeouts: self.session_timeouts.clone(),
            client_identity: None,
        }
    }
}

pub(crate) fn flag(variable: &'static str, value: Option<&str>) -> Result<bool, Error> {
    let Some(text) = value
        .map(|text| text.trim_matches(isspace))
        .filter(|text| !text.is_empty())
    else {
        return Ok(false);
    };
    pydantic::lax_bool(&PythonValue::Str(text.to_owned())).ok_or(Error::Invalid {
        setting: variable.to_owned(),
        expected: Expected::Boolean,
    })
}

fn seconds(configured: Configured<'_, Spelled<f64>>) -> Result<Option<f64>, Error> {
    match configured.value {
        None => Ok(None),
        Some(Spelled::Value(seconds)) => Ok(Some(*seconds)),
        Some(Spelled::Text(text)) => number::float(text).map(Some).ok_or(Error::Invalid {
            setting: configured.setting.to_owned(),
            expected: Expected::Seconds,
        }),
    }
}
