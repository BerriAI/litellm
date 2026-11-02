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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Deployment {
    Standalone,
    Cluster,
}

// redis-py 5.3.1 redis/_parsers/base.py EXCEPTION_CLASSES["ERR"] entries that subclass ConnectionError
const CONNECTION_ERR_MESSAGES: &[&str] = &[
    "max number of clients reached",
    "invalid password",
    "Client sent AUTH, but no password is set",
    "AUTH <password> called without any password configured for the default user. Are you sure your configuration is correct?",
];

impl Error {
    pub(crate) fn classify(error: RedisError, deployment: Deployment) -> Self {
        match (error.kind(), deployment) {
            (ErrorKind::Parse | ErrorKind::UnexpectedReturnType, _) => Self::Decode(error),
            (ErrorKind::Server(kind), _) if is_unavailable(kind, deployment) => {
                Self::Transport(error)
            }
            (ErrorKind::Server(ServerErrorKind::ResponseError), _)
                if error
                    .detail()
                    .is_some_and(|detail| CONNECTION_ERR_MESSAGES.contains(&detail)) =>
            {
                Self::Transport(error)
            }
            (ErrorKind::Extension, _) if matches!(error.code(), Some("NOAUTH" | "WRONGPASS")) => {
                Self::Transport(error)
            }
            (ErrorKind::Server(_) | ErrorKind::Extension | ErrorKind::InvalidClientConfig, _) => {
                Self::Server(error)
            }
            _ => Self::Transport(error),
        }
    }
}

fn is_unavailable(kind: ServerErrorKind, deployment: Deployment) -> bool {
    matches!(
        (kind, deployment),
        (ServerErrorKind::BusyLoading, _)
            | (
                ServerErrorKind::ClusterDown | ServerErrorKind::MasterDown,
                Deployment::Cluster
            )
    )
}

#[cfg(test)]
mod tests {
    use redis::{RedisError, Value};
    use rstest::rstest;

    use super::{Deployment, Error};

    fn reply(resp: &str) -> RedisError {
        match redis::parse_redis_value(resp.as_bytes()).unwrap() {
            Value::ServerError(error) => error.into(),
            other => panic!("{resp:?} parsed as {other:?}"),
        }
    }

    fn is_transport(resp: &str, deployment: Deployment) -> bool {
        match Error::classify(reply(resp), deployment) {
            Error::Transport(_) => true,
            Error::Server(_) => false,
            other => panic!("{resp:?} classified as {other:?}"),
        }
    }

    #[rstest]
    #[case::wrong_type("-WRONGTYPE Operation against a key holding the wrong kind of value\r\n")]
    #[case::no_script("-NOSCRIPT No matching script\r\n")]
    #[case::cross_slot("-CROSSSLOT Keys in request don't hash to the same slot\r\n")]
    #[case::generic("-ERR value is not an integer or out of range\r\n")]
    #[case::no_perm("-NOPERM this user has no permissions to access one of the keys\r\n")]
    #[case::try_again("-TRYAGAIN Multiple keys request during rehashing of slot\r\n")]
    #[case::out_of_memory("-OOM command not allowed when used memory > 'maxmemory'.\r\n")]
    #[case::auth_arity("-ERR wrong number of arguments for 'auth' command\r\n")]
    #[case::max_clients_with_suffix("-ERR max number of clients reached, try later\r\n")]
    fn rejections_are_server_errors_on_every_deployment(
        #[case] resp: &str,
        #[values(Deployment::Standalone, Deployment::Cluster)] deployment: Deployment,
    ) {
        assert!(!is_transport(resp, deployment));
    }

    #[rstest]
    #[case::loading("-LOADING Redis is loading the dataset in memory\r\n")]
    #[case::no_auth("-NOAUTH Authentication required.\r\n")]
    #[case::wrong_pass("-WRONGPASS invalid username-password pair or user is disabled.\r\n")]
    #[case::max_clients("-ERR max number of clients reached\r\n")]
    #[case::invalid_password("-ERR invalid password\r\n")]
    #[case::no_password_set("-ERR Client sent AUTH, but no password is set\r\n")]
    #[case::no_default_password(
        "-ERR AUTH <password> called without any password configured for the default user. Are you sure your configuration is correct?\r\n"
    )]
    fn unavailable_nodes_are_transport_errors_on_every_deployment(
        #[case] resp: &str,
        #[values(Deployment::Standalone, Deployment::Cluster)] deployment: Deployment,
    ) {
        assert!(is_transport(resp, deployment));
    }

    #[rstest]
    #[case::master_down_on_standalone(
        "-MASTERDOWN Link with MASTER is down\r\n",
        Deployment::Standalone,
        false
    )]
    #[case::master_down_on_cluster(
        "-MASTERDOWN Link with MASTER is down\r\n",
        Deployment::Cluster,
        true
    )]
    #[case::cluster_down_on_standalone(
        "-CLUSTERDOWN The cluster is down\r\n",
        Deployment::Standalone,
        false
    )]
    #[case::cluster_down_on_cluster(
        "-CLUSTERDOWN The cluster is down\r\n",
        Deployment::Cluster,
        true
    )]
    #[case::read_only_on_standalone(
        "-READONLY You can't write against a read only replica.\r\n",
        Deployment::Standalone,
        false
    )]
    #[case::read_only_on_cluster(
        "-READONLY You can't write against a read only replica.\r\n",
        Deployment::Cluster,
        false
    )]
    fn topology_decides_like_redis_py(
        #[case] resp: &str,
        #[case] deployment: Deployment,
        #[case] transport: bool,
    ) {
        assert_eq!(is_transport(resp, deployment), transport);
    }
}
