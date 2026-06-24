//! Pure guardrail building blocks: the content-filter scanner, in-process PII
//! detection/masking, and the shared verdict, input, error, and config types.
//! Network-backed guardrail providers live in the `ai-gateway` host crate; this
//! module never touches the network, filesystem, or environment.

pub mod config;
pub mod error;
pub mod input;
pub mod pii;
pub mod scanner;
pub mod verdict;

pub use config::{
    AzurePromptShieldConfig, AzureTextModerationConfig, BedrockConfig, GenericApiConfig,
    LakeraV2Config, LocalPiiConfig, OnFlagged, OpenaiModerationConfig, PiiAction, PresidioConfig,
    ProviderConfig, UnreachableFallback,
};
pub use error::ProviderError;
pub use input::{GuardrailInput, InputType, Message, MessageContent, RequestContext};
pub use pii::LocalPiiEngine;
pub use scanner::{CompileError, LiteralTerm, RegexTerm, ScanMatch, Scanner};
pub use verdict::{Detection, GuardrailOutcome, GuardrailStatus, Verdict};
