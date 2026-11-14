use std::fmt::{self, Write};

use litellm_redis_store::{Error, ErrorKind, rate_limits::Counter};

const NO_REDIS: &str = "the rate limiter has no Redis to run its scripts on";

#[derive(Debug)]
pub enum Failure {
    Store(Error),
    Raised {
        exception: &'static str,
        message: String,
    },
}

impl Failure {
    pub fn no_redis() -> Self {
        Self::raised("RuntimeError", NO_REDIS)
    }

    pub fn raised(exception: &'static str, message: impl Into<String>) -> Self {
        Self::Raised {
            exception,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> Option<ErrorKind> {
        match self {
            Self::Store(error) => Some(error.kind()),
            Self::Raised { .. } => None,
        }
    }

    pub fn exception(&self) -> &'static str {
        match self {
            Self::Store(_) => "NativeRedisStoreError",
            Self::Raised { exception, .. } => exception,
        }
    }
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self::Store(error)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => {
                let kind: &'static str = error.kind().into();
                let report = litellm_error::Report::new(error).to_string();
                write!(
                    formatter,
                    "({}, {})",
                    python_repr(kind),
                    python_repr(&report)
                )
            }
            Self::Raised { message, .. } => formatter.write_str(message),
        }
    }
}

macro_rules! redis_failure {
    ($failure:expr, $message:expr $(,)?) => {{
        let failure: &$crate::failure::Failure = $failure;
        match failure.kind() {
            Some(litellm_redis_store::ErrorKind::BreakerOpen) => {
                tracing::debug!(redis.failure = "breaker_open", "{}: {}", $message, failure)
            }
            kind => tracing::warn!(
                redis.failure = kind.map(<&'static str>::from),
                "{}: {}",
                $message,
                failure
            ),
        }
    }};
}

pub(crate) use redis_failure;

pub fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut repr = String::with_capacity(text.len() + 2);
    repr.push(quote);
    for character in text.chars() {
        match character {
            '\\' => repr.push_str("\\\\"),
            '\n' => repr.push_str("\\n"),
            '\r' => repr.push_str("\\r"),
            '\t' => repr.push_str("\\t"),
            _ if character == quote => {
                repr.push('\\');
                repr.push(character);
            }
            '\0'..='\x1f' | '\x7f'..='\u{a0}' => {
                let _ = write!(repr, "\\x{:02x}", u32::from(character));
            }
            _ => repr.push(character),
        }
    }
    repr.push(quote);
    repr
}

pub fn python_counter(counter: &Counter) -> String {
    let kind: &'static str = counter.kind.into();
    format!(
        "RateLimitCounter(scope=RateLimitScope(descriptor_key={}, descriptor_value={}), kind={})",
        python_repr(&counter.scope.descriptor),
        python_repr(&counter.scope.value),
        python_repr(kind)
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::plain("down", "'down'")]
    #[case::apostrophe("it's down", "\"it's down\"")]
    #[case::both_quotes("it's \"down\"", "'it\\'s \"down\"'")]
    #[case::controls("a\nb\tc\\d\x01", "'a\\nb\\tc\\\\d\\x01'")]
    #[case::unicode("caf\u{e9}", "'caf\u{e9}'")]
    fn python_repr_quotes_like_python(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(python_repr(text), expected);
    }

    #[test]
    fn a_missing_redis_reads_as_the_runtime_error_python_raises() {
        let failure = Failure::no_redis();
        assert_eq!(failure.exception(), "RuntimeError");
        assert_eq!(failure.to_string(), NO_REDIS);
    }
}
