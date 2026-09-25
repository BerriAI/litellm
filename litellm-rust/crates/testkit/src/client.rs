#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Client {
    ClaudeCode,
    Codex,
    Opencode,
}

impl Client {
    pub const fn binary(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude",
            Self::Codex => "codex",
            Self::Opencode => "opencode",
        }
    }
}
