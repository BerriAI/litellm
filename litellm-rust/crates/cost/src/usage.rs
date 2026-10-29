use std::collections::BTreeMap;
use std::time::Duration;

use crate::price::{CacheReadMode, CacheTtl, ImageSize, Resolution, SearchContextSize, TokenKind};

/// What one call consumed. Python fills it from the provider response, `calculate` never reads the catalog for it
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Prompt size as the provider reported it. Only picks `_above_Nk_tokens` bands, never billed.
    /// Filled even when the call bills characters, `0` when the provider reports none
    pub prompt_tokens: u64,
    /// Billed token quantities by meter
    pub tokens: Tokens,
    /// Billed character quantities. Used only when the row's `billing.text_unit` is `Character`
    pub characters: Characters,
    /// Billed time per second meter, never a bare number of seconds or milliseconds
    pub durations: BTreeMap<DurationMeter, Duration>,
    /// Images sent and returned
    pub images: Images,
    /// Search queries, web searches and grounding requests
    pub search: Search,
    /// Realtime text messages, xAI bills them per message
    pub text_messages: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tokens {
    /// Uncached prompt tokens by modality
    pub input: BTreeMap<TokenKind, u64>,
    /// Prompt tokens served from cache, by modality and implicit or explicit cache
    pub cache_read: BTreeMap<(TokenKind, CacheReadMode), u64>,
    /// Prompt tokens written to cache, by modality and cache lifetime
    pub cache_write: BTreeMap<(TokenKind, CacheTtl), u64>,
    /// Completion tokens by modality, reasoning excluded
    pub output: BTreeMap<TokenKind, u64>,
    /// Reasoning tokens, billed at the reasoning rate or the text output rate
    pub reasoning: u64,
    /// Perplexity citation tokens, prompt side
    pub citation: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Characters {
    /// Characters of the formatted prompt
    pub prompt: u64,
    /// Characters of the response text
    pub completion: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DurationMeter {
    /// Response time of the call, for rows priced by wall clock
    WallClock,
    /// Length of the input audio, a transcription file or a realtime transcription
    InputAudio,
    /// Realtime voice session time
    LiveVoice,
    /// Length of generated video, already multiplied by the video count. `resolution` picks `output_cost_per_second_<resolution>`
    OutputVideo { resolution: Option<Resolution> },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Images {
    /// Images sent in the request
    pub input: u64,
    /// Pixels sent in the request, for rows priced per input pixel
    pub input_pixels: u64,
    /// One entry per returned image, since price can depend on each image's size
    pub output: Vec<ImageOut>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageOut {
    /// Size the request asked for, picks a size keyed price like fal's `output_cost_per_image_<size>`
    pub size: Option<ImageSize>,
    /// Pixel width the response reported, for per-pixel and nearest-size pricing
    pub width: Option<u32>,
    /// Pixel height the response reported, for per-pixel and nearest-size pricing
    pub height: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Search {
    /// Search API queries, for `input_cost_per_query` rows
    pub queries: u64,
    /// Results the query asked for, picks the `max_results_range` tier
    pub max_results: Option<u64>,
    /// Web searches a model ran as a tool
    pub web_search: Option<WebSearch>,
    /// Google Maps grounding requests
    pub maps_grounding: u64,
    /// Groq `browser.open` calls
    pub browser_opens: u64,
    /// Results past the ones the query price includes. Python counts them, since the included 10 is a request default
    pub additional_results: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WebSearch {
    /// Web search requests the model made
    pub requests: u64,
    /// Context size the request asked for, picks `search_context_size_<size>`
    pub context: SearchContextSize,
}
