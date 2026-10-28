use redis::{Cmd, ScriptInvocation};

#[derive(Clone)]
pub struct Script(redis::Script);

impl Script {
    pub fn new(source: &str) -> Self {
        Self(redis::Script::new(source))
    }

    pub(crate) fn evalsha(&self, keys: &[String], args: &[Vec<u8>]) -> Cmd {
        redis::cmd("EVALSHA")
            .arg(self.0.get_hash())
            .arg(keys.len())
            .arg(keys)
            .arg(args)
            .to_owned()
    }

    pub(crate) fn invocation(&self, keys: &[String], args: &[Vec<u8>]) -> ScriptInvocation<'_> {
        let mut invocation = self.0.prepare_invoke();
        invocation.key(keys).arg(args);
        invocation
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;

    use super::Script;
    use crate::{
        DEFAULT_TIMEOUT, Error,
        test_support::{Server, Topology, flush_scripts},
    };

    #[rstest]
    #[tokio::test]
    async fn script_keys_are_namespaced(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("a", DEFAULT_TIMEOUT).await;
        let bare = server.client("", DEFAULT_TIMEOUT).await;
        let read = Script::new("return redis.call('GET', KEYS[1])");
        bare.set("a:x", "1", Duration::from_secs(60)).await.unwrap();

        let value: Option<String> = client
            .eval(&read, &["x"], Vec::<String>::new())
            .await
            .unwrap();

        assert_eq!(value.as_deref(), Some("1"));
    }

    #[rstest]
    #[tokio::test]
    async fn a_failing_script_is_a_server_error_and_keeps_the_connection(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;
        let incr = Script::new("return redis.call('INCRBY', KEYS[1], ARGV[1])");
        client.rpush("list", "item").await.unwrap();

        let failed = client.eval::<i64>(&incr, &["list"], 1).await;

        assert!(matches!(failed, Err(Error::Server(_))), "{failed:?}");
        assert_eq!(client.eval::<i64>(&incr, &["counter"], 3).await.unwrap(), 3);
    }

    #[rstest]
    #[tokio::test]
    async fn a_flushed_script_is_loaded_again(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;
        let incr = Script::new("return redis.call('INCRBY', KEYS[1], ARGV[1])");
        client.eval::<i64>(&incr, &["counter"], 1).await.unwrap();

        flush_scripts(&client).await;

        assert_eq!(client.eval::<i64>(&incr, &["counter"], 1).await.unwrap(), 2);
    }
}
