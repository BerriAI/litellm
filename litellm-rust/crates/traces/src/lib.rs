mod error;
mod insert;
mod otlp;
mod schema;
mod sql;

pub use error::{DecodeError, Error};
pub use insert::{InsertTable, encode_rows, insert_rows};
pub use otlp::{DecodedSpan, decode_otlp};
pub use schema::{ensure_schema, schema_statements};
pub use sql::{LensQuery, Parameter, ReadQuery, execute_named_read, execute_read};
use url::Url;

#[derive(Clone)]
pub struct Connection {
    url: Url,
}

impl Connection {
    pub fn parse(value: &str) -> Result<Self, Error> {
        let url = Url::parse(value).map_err(|_| Error::InvalidUrl)?;
        if !matches!(url.scheme(), "http" | "https") || url.host().is_none() {
            return Err(Error::InvalidUrl);
        }
        Ok(Self { url })
    }

    pub fn configured(
        url: &str,
        database: &str,
        user: &str,
        password: &str,
    ) -> Result<Self, Error> {
        let mut connection = Self::parse(url)?;
        connection
            .url
            .set_username(user)
            .map_err(|_| Error::InvalidUrl)?;
        connection
            .url
            .set_password(Some(password))
            .map_err(|_| Error::InvalidUrl)?;
        let pairs: Vec<_> = connection
            .url
            .query_pairs()
            .filter(|(key, _)| !matches!(key.as_ref(), "database" | "user" | "password"))
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect();
        connection
            .url
            .query_pairs_mut()
            .clear()
            .extend_pairs(pairs)
            .append_pair("database", database);
        Ok(connection)
    }

    pub fn writer(url: &str) -> Result<Self, Error> {
        let mut connection = Self::parse(url)?;
        let pairs: Vec<_> = connection
            .url
            .query_pairs()
            .filter(|(key, _)| !matches!(key.as_ref(), "database" | "readonly" | "query"))
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect();
        connection.url.query_pairs_mut().clear().extend_pairs(pairs);
        Ok(connection)
    }

    pub fn reader(url: &str, database: &str) -> Result<Self, Error> {
        let mut connection = Self::parse(url)?;
        let pairs: Vec<_> = connection
            .url
            .query_pairs()
            .filter(|(key, _)| key != "database")
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect();
        connection
            .url
            .query_pairs_mut()
            .clear()
            .extend_pairs(pairs)
            .append_pair("database", database);
        Ok(connection)
    }

    pub fn url(&self) -> &Url {
        &self.url
    }
}
