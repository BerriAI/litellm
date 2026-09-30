mod error;
mod sql;

pub use error::Error;
pub use sql::execute_admin_sql;
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

    pub fn url(&self) -> &Url {
        &self.url
    }
}

#[derive(Debug, Clone)]
pub struct ListQuery {
    pub start_ms: i64,
    pub end_ms: i64,
    pub service: Option<String>,
    pub status: Option<String>,
    pub search: Option<String>,
    pub cursor: Option<String>,
    pub limit: u8,
}

#[derive(Debug, Clone)]
pub enum Query {
    List(ListQuery),
    Trace { trace_id: String },
    Span { trace_id: String, span_id: String },
}

impl Query {
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            Self::List(query)
                if query.start_ms >= query.end_ms || !(1..=100).contains(&query.limit) =>
            {
                Err(Error::InvalidListQuery)
            }
            Self::Trace { trace_id }
            | Self::Span {
                trace_id,
                span_id: _,
            } if trace_id.is_empty() => Err(Error::InvalidIdentifier),
            Self::Span { span_id, .. } if span_id.is_empty() => Err(Error::InvalidIdentifier),
            _ => Ok(()),
        }
    }
}

pub async fn query(_connection: Connection, request: Query) -> Result<(), Error> {
    request.validate()?;
    Err(Error::SchemaPending)
}
