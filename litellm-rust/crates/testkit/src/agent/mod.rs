mod claude_code;
mod codex;
mod opencode;

use std::future::Future;
use std::path::Path;

use crate::install::release::Release;
use crate::{Error, Fetch, Gateway, LaunchSpec, Target};

pub trait Agent: Sync {
    fn binary(&self) -> &'static str;

    fn release(
        &self,
        fetch: &impl Fetch,
        version: &str,
        target: Target,
    ) -> impl Future<Output = Result<Release, Error>> + Send;

    fn launch_spec(&self, gateway: &Gateway, home: &Path) -> LaunchSpec;
}

pub use claude_code::ClaudeCode;
pub use codex::Codex;
pub use opencode::Opencode;
