use serde::Deserialize;
use url::Url;

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
#[strum(serialize_all = "kebab-case")]
pub(crate) enum SslModeText {
    Disable,
    Prefer,
    Require,
    VerifyCa,
    VerifyFull,
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub(crate) struct Params {
    pub host: Option<String>,
    pub schema: Option<String>,
    pub sslmode: Option<String>,
    pub sslaccept: Option<String>,
    pub sslcert: Option<String>,
    pub sslrootcert: Option<String>,
    pub sslidentity: Option<String>,
    pub connection_limit: Option<String>,
    pub pool_timeout: Option<String>,
    pub socket_timeout: Option<String>,
    pub max_idle_connection_lifetime: Option<String>,
    pub pgbouncer: Option<String>,
    pub statement_cache_size: Option<String>,
    pub application_name: Option<String>,
    pub options: Option<String>,
}

impl Params {
    pub(crate) fn from_url(source: &'static str, url: &Url) -> Result<Self, Error> {
        let query: serde_json::Map<String, serde_json::Value> = url
            .query_pairs()
            .filter(|(_, value)| !value.is_empty())
            .map(|(key, value)| {
                (
                    key.into_owned(),
                    serde_json::Value::String(value.into_owned()),
                )
            })
            .collect();
        serde_json::from_value(serde_json::Value::Object(query))
            .map_err(|_| Error::Url { setting: source })
    }

    pub(crate) fn ssl_mode(&self) -> SslModeText {
        self.sslmode.as_deref().map_or(SslModeText::Prefer, |text| {
            text.parse().unwrap_or(SslModeText::Unknown)
        })
    }
}
