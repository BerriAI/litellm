/// A counter's key segment, which is also how the Python router names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
pub enum Kind {
    #[strum(serialize = "tpm")]
    Tpm,
    #[strum(serialize = "rpm")]
    Rpm,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counter {
    pub deployment_id: String,
    pub model: Option<String>,
    pub minute: String,
    pub kind: Kind,
}

impl Counter {
    /// `global_router:{id}:{model}:{tpm|rpm}:{HH-MM}`, where Python formats a missing model as `None`.
    pub(super) fn key(&self) -> String {
        let kind: &str = self.kind.into();
        let model = self.model.as_deref().unwrap_or("None");
        format!(
            "global_router:{}:{model}:{kind}:{}",
            self.deployment_id, self.minute
        )
    }
}
