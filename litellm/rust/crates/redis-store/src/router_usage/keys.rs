/// Who counts under a key. The Rust router core's counters carry a `global_router:` prefix, and
/// the counters usage-based-routing-v2 and the model rate limit check share carry none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::EnumString, strum::IntoStaticStr)]
pub enum Scope {
    #[strum(serialize = "global_router")]
    GlobalRouter,
    #[strum(serialize = "deployment")]
    Deployment,
}

/// A counter's key segment, which is also how the Python router names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::EnumString, strum::IntoStaticStr)]
pub enum Kind {
    #[strum(serialize = "tpm")]
    Tpm,
    #[strum(serialize = "rpm")]
    Rpm,
    /// Input tokens per minute, which the io-token rate limit check reads.
    #[strum(serialize = "itpm")]
    Itpm,
    /// Output tokens per minute.
    #[strum(serialize = "otpm")]
    Otpm,
    /// Tokens per UTC day, whose window is `YYYY-MM-DD`.
    #[strum(serialize = "tpd")]
    Tpd,
    /// Requests per UTC day.
    #[strum(serialize = "rpd")]
    Rpd,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Counter {
    pub scope: Scope,
    pub deployment_id: String,
    pub model: Option<String>,
    /// `HH-MM` for a per-minute kind, `YYYY-MM-DD` for a per-day one.
    pub window: String,
    pub kind: Kind,
}

impl Counter {
    /// `global_router:{id}:{model}:{kind}:{window}` or `{id}:{model}:{kind}:{window}`, where Python
    /// formats a missing model as `None`. Public only for the hosts that hand a counter's key back
    /// to their callers.
    pub fn key(&self) -> String {
        let kind: &str = self.kind.into();
        let model = self.model.as_deref().unwrap_or("None");
        let id = &self.deployment_id;
        let window = &self.window;
        match self.scope {
            Scope::GlobalRouter => format!("global_router:{id}:{model}:{kind}:{window}"),
            Scope::Deployment => format!("{id}:{model}:{kind}:{window}"),
        }
    }
}
