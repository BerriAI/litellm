use std::sync::Arc;

use redis::{Cmd, ScriptInvocation};

/// A Lua script, cheap to clone into every call that runs it.
#[derive(Clone)]
pub struct Script(Arc<redis::Script>);

impl Script {
    pub fn new(source: &str) -> Self {
        Self(Arc::new(redis::Script::new(source)))
    }

    /// The `EVALSHA` a pipeline sends, like redis-py. A `NOSCRIPT` reply is settled by
    /// [`Script::invocation`] alone, which loads the script and sends it again.
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

/// A script call in a pipeline, kept apart from plain commands because it is the one entry a
/// pipeline may send again.
pub(crate) struct ScriptCall {
    pub(crate) script: Script,
    pub(crate) keys: Vec<String>,
    pub(crate) args: Vec<Vec<u8>>,
}
