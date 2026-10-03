//! The pagination contract shared by every LiteLLM read: an opaque signed cursor codec, the
//! common page envelope with traversal metadata, and stable failure codes.

mod cursor;
mod error;
mod page;

pub use cursor::{Binding, Cursor, KeyRing};
pub use error::{Error, Failure, FailureCode};
pub use page::{Page, Traversal, rfc3339, rfc3339_ms};
