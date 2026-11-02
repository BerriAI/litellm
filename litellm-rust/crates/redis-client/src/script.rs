use std::sync::Arc;

use redis::{Cmd, ScriptInvocation};

#[derive(Clone)]
pub struct Script {
    cached: redis::Script,
    source: Arc<str>,
}

impl Script {
    pub fn new(source: &str) -> Self {
        Self {
            cached: redis::Script::new(source),
            source: source.into(),
        }
    }

    pub(crate) fn eval(&self, keys: &[String], args: &[Vec<u8>]) -> Cmd {
        redis::cmd("EVAL")
            .arg(&*self.source)
            .arg(keys.len())
            .arg(keys)
            .arg(args)
            .to_owned()
    }

    pub(crate) fn invocation(&self, keys: &[String], args: &[Vec<u8>]) -> ScriptInvocation<'_> {
        let mut invocation = self.cached.prepare_invoke();
        invocation.key(keys).arg(args);
        invocation
    }
}
