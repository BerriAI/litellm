use serde::{Deserialize, Serialize};

use crate::field::{Field, FieldSet};

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Hook {
    PromptResolve = 0,
    PreRoute = 1,
    FilterDeployments = 2,
    PreCall = 3,
    DuringCall = 4,
    PreDeploymentCall = 5,
    PostDeploymentSuccess = 6,
    PostDeploymentFailure = 7,
    PostCallSuccess = 8,
    PostCallFailure = 9,
    ResponseHeaders = 10,
    StreamChunk = 11,
    ListModels = 12,
    McpToolResult = 13,
    LogPrepare = 14,
    ProviderRequestSent = 15,
    ProviderResponseReceived = 16,
    AttemptEnded = 17,
    CallEnded = 18,
    RouterEvent = 19,
    AuditEvent = 20,
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    Abort,
    Respond,
    Reroute,
}

const NONE: &[OutcomeKind] = &[];
const ABORT: &[OutcomeKind] = &[OutcomeKind::Abort];
const ABORT_OR_RESPOND: &[OutcomeKind] = &[OutcomeKind::Abort, OutcomeKind::Respond];
const ANY: &[OutcomeKind] = &[
    OutcomeKind::Abort,
    OutcomeKind::Respond,
    OutcomeKind::Reroute,
];

impl Hook {
    pub const fn bit(self) -> u8 {
        self as u8
    }

    pub const fn request_path(self) -> bool {
        (self as u8) < (Self::LogPrepare as u8)
    }

    pub const fn subscribable(self) -> bool {
        !matches!(self, Self::StreamChunk)
    }

    pub const fn writable(self) -> FieldSet {
        match self {
            Self::PromptResolve => {
                FieldSet::of(&[Field::Messages, Field::OptionalParams, Field::Model])
            }
            Self::PreRoute => FieldSet::of(&[Field::Model, Field::Messages]),
            Self::FilterDeployments => FieldSet::of(&[Field::CandidateDeployments]),
            Self::PreCall => FieldSet::of(&[
                Field::Messages,
                Field::OptionalParams,
                Field::Metadata,
                Field::RequestHeaders,
                Field::GuardrailInfo,
                Field::Tools,
                Field::Model,
                Field::Input,
                Field::BudgetReservation,
                Field::RateLimitStatuses,
            ]),
            Self::PreDeploymentCall => {
                FieldSet::of(&[Field::ProviderRequestBody, Field::RequestHeaders])
            }
            Self::PostDeploymentSuccess => FieldSet::of(&[Field::Response]),
            Self::PostDeploymentFailure | Self::PostCallFailure => FieldSet::of(&[Field::Error]),
            Self::PostCallSuccess => FieldSet::of(&[Field::Response, Field::GuardrailInfo]),
            Self::ResponseHeaders => FieldSet::of(&[Field::ResponseHeaders]),
            Self::StreamChunk => FieldSet::of(&[Field::StreamChunk]),
            Self::ListModels => FieldSet::of(&[Field::ModelList]),
            Self::McpToolResult => FieldSet::of(&[Field::ToolResult]),
            Self::LogPrepare => FieldSet::of(&[Field::StandardLoggingPayload]),
            Self::DuringCall
            | Self::ProviderRequestSent
            | Self::ProviderResponseReceived
            | Self::AttemptEnded
            | Self::CallEnded
            | Self::RouterEvent
            | Self::AuditEvent => FieldSet::EMPTY,
        }
    }

    pub const fn outcomes(self) -> &'static [OutcomeKind] {
        match self {
            Self::PromptResolve
            | Self::PreRoute
            | Self::DuringCall
            | Self::PreDeploymentCall
            | Self::PostDeploymentSuccess
            | Self::PostCallSuccess
            | Self::McpToolResult => ABORT_OR_RESPOND,
            Self::FilterDeployments
            | Self::PostDeploymentFailure
            | Self::PostCallFailure
            | Self::StreamChunk
            | Self::ListModels => ABORT,
            Self::PreCall => ANY,
            Self::ResponseHeaders
            | Self::LogPrepare
            | Self::ProviderRequestSent
            | Self::ProviderResponseReceived
            | Self::AttemptEnded
            | Self::CallEnded
            | Self::RouterEvent
            | Self::AuditEvent => NONE,
        }
    }
}
