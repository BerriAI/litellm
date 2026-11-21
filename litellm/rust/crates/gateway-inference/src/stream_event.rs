//! What one event of a stream says, in the words the shared stream code asks in. Each streaming
//! route reads the events of its own format into a [`Reading`], and names that reader to
//! `stream_outcome`, for the tokens and the failure it tells the router, and to `stream_hold`, for
//! whether the stream has begun to answer.

use litellm_core_utils::core_helpers::usage_from_parts;
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;

/// The tokens an event's `usage` counts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Counts {
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) cache_creation_input_tokens: Option<u64>,
    pub(crate) cache_read_input_tokens: Option<u64>,
}

impl Counts {
    /// These counts over `earlier` ones, a count this event does not carry kept from them.
    pub(crate) fn over(self, earlier: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.or(earlier.input_tokens),
            output_tokens: self.output_tokens.or(earlier.output_tokens),
            cache_creation_input_tokens: self
                .cache_creation_input_tokens
                .or(earlier.cache_creation_input_tokens),
            cache_read_input_tokens: self
                .cache_read_input_tokens
                .or(earlier.cache_read_input_tokens),
        }
    }

    /// The counts as Python's litellm `Usage` holds a Messages answer's: the cache tokens folded
    /// into `prompt_tokens`, the reads also named as `cached_tokens`.
    pub(crate) fn messages_usage(self) -> ChatCompletionsUsage {
        usage_from_parts(
            self.input_tokens.unwrap_or(0),
            self.output_tokens.unwrap_or(0),
            self.cache_read_input_tokens.unwrap_or(0),
            self.cache_creation_input_tokens.unwrap_or(0),
        )
    }

    /// The input plus the output.
    pub(crate) fn total(self) -> u64 {
        self.input_tokens.unwrap_or(0) + self.output_tokens.unwrap_or(0)
    }
}

/// What an event says about how far its stream has come.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Nothing a client would miss if another attempt answered instead.
    Lifecycle,
    /// Generated output: the stream is the answer from here on.
    Content,
    /// The provider's error. `status` and `answer` are what the provider answers a request with
    /// when it fails the same way before any stream.
    Failed { status: u16, answer: String },
}

/// What one event of a stream says.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub kind: Kind,
    pub usage: Option<Counts>,
}

impl Reading {
    pub(crate) fn lifecycle(usage: Option<impl Into<Counts>>) -> Self {
        Self {
            kind: Kind::Lifecycle,
            usage: usage.map(Into::into),
        }
    }

    pub(crate) fn of(kind: Kind) -> Self {
        Self { kind, usage: None }
    }
}

/// Reads the data of one event of a format's stream.
pub type Read = fn(&str) -> Reading;
