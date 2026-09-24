use crate::MAX_DEPTH;

/// The CPython exception class a failed conversion raises.
#[derive(Clone, Copy, Debug, Eq, PartialEq, strum::Display, strum::IntoStaticStr)]
pub enum PythonException {
    TypeError,
    ValueError,
    OverflowError,
}

/// A failure to read or write a Python format. Messages quote CPython's own wording where
/// the Python side raises (`TypeError`, `ValueError`), so callers can log them as is.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("malformed Python literal at byte {0}")]
    InvalidLiteral(usize),
    #[error("unhashable type: '{0}'")]
    Unhashable(&'static str),
    #[error("value nests deeper than {MAX_DEPTH} levels")]
    TooDeep,
    #[error("invalid pickle: {0}")]
    InvalidPickle(String),
    #[error("Object of type {0} is not JSON serializable")]
    NotJsonSerializable(&'static str),
    #[error("keys must be str, int, float, bool or None, not {0}")]
    InvalidJsonKey(&'static str),
    #[error("Out of range float values are not JSON compliant")]
    NonFiniteFloat,
    #[error("integer does not fit in the JSON number range")]
    IntegerOutOfRange,
    #[error("Object of type {0} cannot be pickled as plain data")]
    NotPicklable(&'static str),
    #[error("{exception}: {message}")]
    Conversion {
        exception: PythonException,
        message: String,
    },
}

impl Error {
    pub(crate) fn conversion(exception: PythonException, message: impl Into<String>) -> Self {
        Self::Conversion {
            exception,
            message: message.into(),
        }
    }
}
