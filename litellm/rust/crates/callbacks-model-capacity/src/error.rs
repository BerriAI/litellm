use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RaisedKind {
    Http,
    RegistryUnavailable,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("{kind:?}: {message}")]
pub struct Raised {
    pub token: u64,
    pub kind: RaisedKind,
    pub message: String,
}

#[derive(Debug, thiserror::Error)]
pub enum CapacityError {
    #[error("the host raised {0}")]
    Raised(Raised),
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    Type(String),
    #[error("the host answered {method} with an unexpected shape: {detail}")]
    Shape {
        method: &'static str,
        detail: String,
    },
}

impl CapacityError {
    pub fn raised(&self) -> Option<&Raised> {
        match self {
            Self::Raised(raised) => Some(raised),
            _ => None,
        }
    }

    pub fn is_http(&self) -> bool {
        self.raised()
            .is_some_and(|raised| raised.kind == RaisedKind::Http)
    }

    pub fn is_type_error(&self) -> bool {
        matches!(self, Self::Type(_))
    }

    pub fn python_message(&self) -> String {
        match self {
            Self::Raised(raised) => raised.message.clone(),
            _ => self.to_string(),
        }
    }

    pub fn is_registry_unavailable(&self) -> bool {
        self.raised()
            .is_some_and(|raised| raised.kind == RaisedKind::RegistryUnavailable)
    }
}
