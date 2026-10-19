pub const GLOBAL_API_BASE: &str = "https://aiplatform.googleapis.com";
pub const GLOBAL_LOCATION: &str = "global";
pub const DEFAULT_LOCATION: &str = "us-central1";

pub const RAW_PREDICT: &str = "rawPredict";
pub const STREAM_RAW_PREDICT: &str = "streamRawPredict";
pub const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

pub const GENERATE_CONTENT: &str = "generateContent";
pub const STREAM_GENERATE_CONTENT: &str = "streamGenerateContent";
/// The Gemini API host Google AI Studio keys address.
pub const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com";
/// Files uploaded through the Gemini Files API, passed by reference and never fetched.
pub const GEMINI_FILES_API_URI_PREFIX: &str =
    "https://generativelanguage.googleapis.com/v1beta/files/";
/// The header a Gemini API key travels in, never the `key=` query parameter.
pub const GEMINI_API_KEY_HEADER: &str = "x-goog-api-key";
