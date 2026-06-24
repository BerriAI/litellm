//! Pure guardrail building blocks: the content-filter scanner, in-process PII
//! detection/masking, and the shared verdict and config types. Network-backed
//! guardrail providers live in the `ai-gateway` host crate; this module never
//! touches the network, filesystem, or environment.

pub mod config;
pub mod pii;
pub mod scanner;
pub mod verdict;

pub use config::{LocalPiiConfig, PiiAction};
pub use pii::LocalPiiEngine;
pub use scanner::{CompileError, LiteralTerm, RegexTerm, ScanMatch, Scanner};
pub use verdict::{Detection, Verdict};
