pub mod error;
pub mod ocr;
pub mod providers;
pub mod realtime;

pub use error::{CoreError, CoreResult};
pub use providers::LlmProvider;
