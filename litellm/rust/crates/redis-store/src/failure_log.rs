/// `log_redis_failure`: an open breaker is DEBUG, anything else WARNING. The event's `redis.failure`
/// field names the class, so the host can rate-limit timeout lines with the throttle redis-py shares.
/// A macro so the event carries its caller's target, file and line, as `stacklevel=2` does in Python.
macro_rules! redis_failure {
    ($error:expr, $message:expr $(,)?) => {{
        let error = $error;
        let kind = $crate::ErrorKind::from(error);
        let failure: &'static str = kind.into();
        match kind {
            $crate::ErrorKind::BreakerOpen => {
                tracing::debug!(redis.failure = failure, "{}: {}", $message, error)
            }
            _ => tracing::warn!(redis.failure = failure, "{}: {}", $message, error),
        }
    }};
}

pub(crate) use redis_failure;
