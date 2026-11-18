mod claude;
mod codex;
mod configure;
mod drive;
mod install;
mod opencode;

pub use claude::ClaudeCode;
pub use codex::Codex;
pub use configure::{Configure, LaunchSpec, Settings, Wire};
pub use drive::{Drive, Outcome, Prompt, Usage};
pub use install::Install;
pub use opencode::Opencode;

pub(crate) use configure::{env, path_string, quoted, v1};
pub(crate) use drive::json_lines;

pub trait Agent: Install + Configure + Drive {}

impl<T: Install + Configure + Drive> Agent for T {}
