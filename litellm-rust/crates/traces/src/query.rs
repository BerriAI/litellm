pub mod guide;

#[derive(Clone, Copy, Debug, Eq, PartialEq, strum::EnumString, strum::Display, strum::AsRefStr)]
#[strum(serialize_all = "snake_case")]
pub enum ReadQuery {
    Availability,
    Agents,
    Sample,
    Content,
    Evidence,
}

impl ReadQuery {
    pub fn parse(value: &str) -> Result<Self, crate::InvalidQuery> {
        value.parse().map_err(|_| crate::InvalidQuery)
    }
}
