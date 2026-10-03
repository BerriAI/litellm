#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "a dev-only installer tool that never talks to providers"
)]

mod agent;
mod error;
mod install;
mod session;
mod target;

pub use agent::{
    Agent, ClaudeCode, Codex, Configure, Drive, Install, LaunchSpec, Opencode, Outcome, Prompt,
    Settings, Usage, Wire,
};
pub use error::Error;
pub use install::{Fetch, HttpFetch, Installed, Installer, Packaging, Release};
pub use semver::Version;
pub use session::Session;
pub use target::{Arch, Os, Target};
