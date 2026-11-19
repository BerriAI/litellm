use std::borrow::Cow;

use litellm_env::Lookup;
use litellm_secret_value::SecretValue;
use percent_encoding::percent_decode_str;
use url::{Host, Url};

use crate::{
    Error, Expected,
    params::{Params, SslModeText},
};

pub const WRITER_URL: &str = "DATABASE_URL";
pub const REPLICA_URL: &str = "DATABASE_URL_READ_REPLICA";
const PREFIX: &str = "DATABASE_";
const REPLICA_SUFFIX: &str = "_READ_REPLICA";
const DEFAULT_PORT: u16 = 5432;
const DEFAULT_DATABASE: &str = "postgres";

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Location {
    pub source: &'static str,
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: Option<SecretValue>,
    pub database: String,
    pub params: Params,
}

impl Location {
    pub(crate) fn from_url(source: &'static str, text: &str) -> Result<Self, Error> {
        let (text, hostless) = with_a_host(text);
        let url = Url::parse(&text).map_err(|_| Error::Url { setting: source })?;
        if !matches!(url.scheme(), "postgresql" | "postgres") {
            return Err(Error::Scheme {
                setting: source,
                scheme: url.scheme().to_owned(),
            });
        }
        let params = Params::from_url(source, &url)?;
        let host = match url.host() {
            Some(Host::Domain(name)) => name.to_lowercase(),
            Some(Host::Ipv4(address)) => address.to_string(),
            Some(Host::Ipv6(address)) => address.to_string(),
            None => String::new(),
        };
        let username = decoded(url.username());
        if ((hostless || host.is_empty()) && params.host.is_none()) || username.is_empty() {
            return Err(Error::Incomplete { setting: source });
        }
        let database = decoded(url.path().trim_start_matches('/'));
        Ok(Self {
            source,
            host: if host.is_empty() {
                "localhost".to_owned()
            } else {
                host
            },
            port: url.port().unwrap_or(DEFAULT_PORT),
            username,
            password: url
                .password()
                .map(decoded)
                .filter(|password| !password.is_empty())
                .map(SecretValue::new),
            database: if database.is_empty() {
                DEFAULT_DATABASE.to_owned()
            } else {
                database
            },
            params,
        })
    }
}

// the url crate refuses credentials without a host, which is how a Unix socket url is written
fn with_a_host(text: &str) -> (Cow<'_, str>, bool) {
    let Some((scheme, rest)) = text.split_once("://") else {
        return (Cow::Borrowed(text), false);
    };
    let (authority, tail) = rest.split_at(rest.find(['/', '?', '#']).unwrap_or(rest.len()));
    match authority.ends_with('@') {
        true => (
            Cow::Owned(format!("{scheme}://{authority}localhost{tail}")),
            true,
        ),
        false => (Cow::Borrowed(text), false),
    }
}

fn decoded(text: &str) -> String {
    percent_decode_str(text).decode_utf8_lossy().into_owned()
}

macro_rules! fields {
    ($($field:ident),* $(,)?) => {
        /// Each field is the `DATABASE_<FIELD>` variable, and `DATABASE_<FIELD>_READ_REPLICA` on the replica.
        struct Fields {
            $($field: Option<String>,)*
        }

        impl Fields {
            fn read(environment: &dyn Lookup, suffix: &str) -> Self {
                Self {
                    $($field: environment.truthy(&format!(
                        "{PREFIX}{}{suffix}",
                        stringify!($field).to_uppercase()
                    )),)*
                }
            }
        }
    };
}

fields!(
    url,
    host,
    port,
    user,
    username,
    password,
    name,
    schema,
    sslmode,
    sslrootcert
);

impl Fields {
    fn writer(environment: &dyn Lookup) -> Self {
        Self::read(environment, "")
    }

    fn replica(environment: &dyn Lookup) -> Self {
        let own = Self::read(environment, REPLICA_SUFFIX);
        let writer = Self::writer(environment);
        Self {
            url: own.url,
            host: own.host,
            port: own.port.or(writer.port),
            user: own
                .user
                .or(own.username)
                .or(writer.user)
                .or(writer.username),
            username: None,
            password: own.password.or(writer.password),
            name: own.name.or(writer.name),
            schema: own.schema.or(writer.schema),
            sslmode: writer.sslmode,
            sslrootcert: writer.sslrootcert,
        }
    }

    fn location(self, source: &'static str) -> Result<Option<Location>, Error> {
        let (Some(host), Some(username), Some(database)) =
            (self.host, self.user.or(self.username), self.name)
        else {
            return Ok(None);
        };
        let port = match self.port {
            Some(text) => text.trim().parse().map_err(|_| Error::Invalid {
                setting: format!("{source} port"),
                expected: Expected::Port,
            })?,
            None => DEFAULT_PORT,
        };
        let verified: &'static str = SslModeText::VerifyFull.into();
        let sslmode = self
            .sslmode
            .or_else(|| self.sslrootcert.as_ref().map(|_| verified.to_owned()));
        Ok(Some(Location {
            source,
            host: host.to_lowercase(),
            port,
            username,
            password: self.password.map(SecretValue::new),
            database,
            params: Params {
                schema: self.schema,
                sslmode,
                sslrootcert: self.sslrootcert,
                ..Params::default()
            },
        }))
    }
}

pub(crate) fn writer(
    configured: Option<&SecretValue>,
    environment: &dyn Lookup,
) -> Result<Option<Location>, Error> {
    let configured = configured
        .map(SecretValue::expose)
        .filter(|url| !url.is_empty());
    if let Some(url) = configured {
        return Location::from_url("general_settings.database_url", url).map(Some);
    }
    let fields = Fields::writer(environment);
    if let Some(url) = &fields.url {
        return Location::from_url(WRITER_URL, url).map(Some);
    }
    fields.location("DATABASE_HOST")
}

pub(crate) fn reader(environment: &dyn Lookup) -> Result<Option<Location>, Error> {
    let fields = Fields::replica(environment);
    if let Some(url) = &fields.url {
        return Location::from_url(REPLICA_URL, url).map(Some);
    }
    fields.location("DATABASE_HOST_READ_REPLICA")
}
