/// The LiteLLM entrypoint an attempt calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Completion,
    Responses,
    AnthropicMessages,
    /// Any other `litellm.<name>` the router forwards through its generic path.
    Generic(String),
}

/// How an operation is carried out. Every operation is an `Invoke` of the host today; a
/// native implementation slots in here without changing the execute loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Native,
    Invoke,
}

impl Operation {
    pub fn from_name(name: &str) -> Self {
        match name {
            "completion" | "acompletion" => Self::Completion,
            "responses" | "aresponses" => Self::Responses,
            "anthropic_messages" | "aanthropic_messages" => Self::AnthropicMessages,
            _ => Self::Generic(name.to_owned()),
        }
    }

    pub fn dispatch(&self) -> Dispatch {
        Dispatch::Invoke
    }

    /// Whether the router treats it as a generic call, whose 404s never cool a deployment
    /// down from the fallback path.
    pub fn is_generic(&self) -> bool {
        !matches!(self, Self::Completion)
    }
}
