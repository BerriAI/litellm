use super::public::StatusClass;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalClass {
    ValueError,
    FileNotFound,
    OsError,
}

/// A route failure in the shape Python's `exception_type` receives it, before any public
/// class is chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum OriginalException {
    Http {
        status: u16,
        body: String,
        headers: Vec<(String, String)>,
    },
    Connection {
        message: String,
    },
    Timeout {
        timeout_seconds: Option<f64>,
        elapsed_seconds: Option<f64>,
    },
    Response {
        message: String,
    },
    Local {
        class: LocalClass,
        message: String,
    },
    /// A failure Python raises as a public LiteLLM exception itself, which `exception_type`
    /// hands back unchanged.
    Public {
        class: StatusClass,
        message: String,
    },
}

/// Which of the provider-specific mappers in `exception_type` a route's provider uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExceptionFamily {
    OpenAiCompatible,
    VertexAi,
    Cohere,
    #[default]
    Other,
}
