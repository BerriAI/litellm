use redis::{ErrorKind, RedisError, ServerErrorKind};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Redis did not answer, so the command may or may not have applied")]
    Transport(#[source] RedisError),
    #[error("Redis rejected the command")]
    Server(#[source] RedisError),
    #[error("Redis replied with a value of a different type")]
    Decode(#[source] RedisError),
    #[error("the Redis circuit breaker is open, so the command was not sent")]
    BreakerOpen,
}

#[derive(Debug, thiserror::Error)]
#[error("connecting to Redis")]
pub struct ConnectError(#[from] RedisError);

impl From<RedisError> for Error {
    fn from(error: RedisError) -> Self {
        match error.kind() {
            ErrorKind::Server(
                ServerErrorKind::BusyLoading
                | ServerErrorKind::ClusterDown
                | ServerErrorKind::MasterDown
                | ServerErrorKind::TryAgain
                | ServerErrorKind::ReadOnly,
            ) => Self::Transport(error),
            ErrorKind::Server(_) | ErrorKind::Extension => Self::Server(error),
            ErrorKind::Parse | ErrorKind::UnexpectedReturnType => Self::Decode(error),
            _ => Self::Transport(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use redis::{RedisError, Value};
    use rstest::rstest;

    use super::Error;

    fn reply(resp: &str) -> RedisError {
        match redis::parse_redis_value(resp.as_bytes()).unwrap() {
            Value::ServerError(error) => error.into(),
            other => panic!("{resp:?} parsed as {other:?}"),
        }
    }

    #[rstest]
    #[case::wrong_type("-WRONGTYPE Operation against a key holding the wrong kind of value\r\n")]
    #[case::no_script("-NOSCRIPT No matching script\r\n")]
    #[case::cross_slot("-CROSSSLOT Keys in request don't hash to the same slot\r\n")]
    #[case::generic("-ERR value is not an integer or out of range\r\n")]
    #[case::no_perm("-NOPERM this user has no permissions to access one of the keys\r\n")]
    fn rejections_are_server_errors(#[case] resp: &str) {
        assert!(matches!(Error::from(reply(resp)), Error::Server(_)));
    }

    #[rstest]
    #[case::loading("-LOADING Redis is loading the dataset in memory\r\n")]
    #[case::cluster_down("-CLUSTERDOWN The cluster is down\r\n")]
    #[case::master_down("-MASTERDOWN Link with MASTER is down\r\n")]
    #[case::try_again("-TRYAGAIN Multiple keys request during rehashing of slot\r\n")]
    #[case::read_only("-READONLY You can't write against a read only replica.\r\n")]
    fn unavailable_nodes_are_transport_errors(#[case] resp: &str) {
        assert!(matches!(Error::from(reply(resp)), Error::Transport(_)));
    }
}
