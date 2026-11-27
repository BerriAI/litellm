use std::fmt;

/// Why an auth object was not read from the database.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the auth object was not read from the database")]
    Database(#[from] litellm_db::Error),
    /// `column` is a column of the refused row itself, and `within` the field of the outer row
    /// that row is nested under, if any.
    #[error("the row's `{}` column is not what the auth cache holds for it", Path(*.within, .column))]
    Row {
        column: &'static str,
        within: Option<&'static str>,
    },
}

impl Error {
    pub(crate) fn row(column: &'static str) -> Self {
        Self::Row {
            column,
            within: None,
        }
    }
}

/// Names the field of the outer row that a refused nested row sits under, for `map_err` on the
/// nested row's shaper.
pub(crate) fn within(field: &'static str) -> impl FnOnce(Error) -> Error {
    move |error| match error {
        Error::Row { column, .. } => Error::Row {
            column,
            within: Some(field),
        },
        Error::Database(_) => error,
    }
}

struct Path<'a>(Option<&'a str>, &'a str);

impl fmt::Display for Path<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(within) => write!(formatter, "{within}.{}", self.1),
            None => formatter.write_str(self.1),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Error, within};

    #[rstest]
    #[case::a_column_of_the_row(Error::row("metadata"), "`metadata`")]
    #[case::a_column_of_a_nested_row(within("team_member")(Error::row("role")), "`team_member.role`")]
    fn a_refused_row_names_the_column_by_its_path(#[case] error: Error, #[case] path: &str) {
        assert_eq!(
            error.to_string(),
            format!("the row's {path} column is not what the auth cache holds for it")
        );
    }
}
