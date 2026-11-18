//! Built-in tool fees, a port of `StandardBuiltInToolCostTracking` (`llm_cost_calc/tool_call_cost_tracking.py`) and
//! the web search and Maps grounding calculators it dispatches to (`llms/__init__.py` and each provider's
//! `cost_per_web_search_request`). The caller reads which tools a response used and how often off the response, and
//! looks up the rows main's calculators read. This module picks each tool's price, its fallbacks and the order main
//! adds the fees in

use crate::image::{ImageCalculator, ImageQuery};
use crate::money::{UnitPrice, Usd};
use crate::price::{
    Channel, Feature, InputUnit, Meter, OutputUnit, SearchContextSize, TokenKind, WebSearchUnit,
};
use crate::priced::Quantity;
use crate::query::{Header, PriceSource};
use crate::receipt::{PriceOutcome, Receipt, Unpriced};
use crate::row::{LoweredRow, RowRef, RowSource};
use crate::units::{self, Billed};

/// Gemini's grounding price per request for a row that lists none (`gemini/cost_calculator.py`)
const GEMINI_WEB_SEARCH: f64 = 35e-3;
/// xAI's web search price per call for a row that lists none (`xai/cost_calculator.py`)
const XAI_WEB_SEARCH: f64 = 5.0 / 1000.0;
/// Groq's `browser.open` price per action (`GROQ_BROWSER_VISIT_WEBSITE_COST_PER_CALL`)
const GROQ_BROWSER_OPEN: f64 = 1.0 / 1000.0;
/// Google's Maps grounding list prices, per query and per grounded prompt (`gemini/cost_calculator.py`)
const MAPS_PER_QUERY: f64 = 14e-3;
const MAPS_PER_PROMPT: f64 = 25e-3;

const SIZES: [SearchContextSize; 3] = [
    SearchContextSize::Low,
    SearchContextSize::Medium,
    SearchContextSize::High,
];

/// Built-in tool fees to price
#[derive(Clone, Debug, PartialEq)]
pub struct ToolsQuery {
    pub header: Header,
    pub tools: Tools,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(
        rename_all = "snake_case",
        bound(deserialize = "R: serde::Deserialize<'de>")
    )
)]
pub enum Tools<R = RowRef> {
    /// Every tool fee a finished call bills, `get_cost_for_built_in_tools`
    Call(Box<ToolUse<R>>),
    /// One tool's calculator alone. A calculator main returned `None` from answers `Unpriced`
    One(Tool<R>),
}

/// The built-in tools a finished call used. Main bills Maps grounding and the Responses image generation tool on
/// every call, then the web search, else the file search, else the Azure assistant tools. `R` names a row, a
/// `RowRef` once the caller resolved it
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(default, bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct ToolUse<R = RowRef> {
    /// The call's provider, which gates the Azure assistant tools
    pub provider: Option<Box<str>>,
    pub maps_grounding: Option<MapsGrounding<R>>,
    pub image_generation: Option<ImageGeneration<R>>,
    /// Present when the response shows a web search
    pub web_search: Option<WebSearch<R>>,
    /// Present when the response shows a file search
    pub file_search: Option<FileSearch<R>>,
    pub vector_store: Option<VectorStore<R>>,
    pub computer_use: Option<ComputerUse<R>>,
    pub code_interpreter: Option<CodeInterpreter<R>>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(
        tag = "tool",
        rename_all = "snake_case",
        bound(deserialize = "R: serde::Deserialize<'de>")
    )
)]
pub enum Tool<R = RowRef> {
    WebSearch(WebSearch<R>),
    MapsGrounding(MapsGrounding<R>),
    FileSearch(FileSearch<R>),
    VectorStore(VectorStore<R>),
    ComputerUse(ComputerUse<R>),
    CodeInterpreter(CodeInterpreter<R>),
}

/// A web search: the provider calculator main dispatched to, and the price per call it fell back to
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct WebSearch<R = RowRef> {
    /// `None` where main had no row, usage or provider to dispatch with
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub dispatch: Option<Dispatch<R>>,
    /// `None` where main's caller had no fallback and kept the calculator's `None`
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub per_call: Option<PerCall<R>>,
}

/// `get_cost_for_web_search_request`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct Dispatch<R = RowRef> {
    pub provider: Box<str>,
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub row: Option<R>,
    /// The row of the model the response names, whose search prices an Anthropic, Vertex AI Claude or Azure AI row
    /// without any borrows (`_with_served_model_search_pricing`)
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub served_row: Option<R>,
    /// The row's `key`, which picks the Anthropic calculator for a Vertex AI or Azure AI Claude model
    #[cfg_attr(feature = "serde", serde(default))]
    pub model_key: Box<str>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub response_model: Option<Box<str>>,
    pub counts: SearchCounts,
}

/// The search counts a usage reports
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct SearchCounts {
    /// `prompt_tokens_details.web_search_requests`
    pub prompt_requests: Option<u64>,
    /// `server_tool_use.web_search_requests`
    pub server_requests: Option<u64>,
    /// `server_tool_use.browser_open_requests`
    pub browser_opens: Option<u64>,
    /// `server_side_tool_usage_details.web_search_calls`
    pub search_calls: u64,
    /// The usage states a cost of its own, which xAI's total already holds
    pub stated_cost: bool,
}

/// `get_cost_for_web_search` times the billable calls
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct PerCall<R = RowRef> {
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub row: Option<R>,
    /// `web_search_options.search_context_size`, `None` for any size main does not name
    #[cfg_attr(feature = "serde", serde(default))]
    pub context: Option<SearchContextSize>,
    pub calls: u64,
}

/// `cost_per_google_maps_grounding_request`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct MapsGrounding<R = RowRef> {
    /// The provider main dispatched with, `None` when the caller called the Gemini calculator itself
    #[cfg_attr(feature = "serde", serde(default))]
    pub provider: Option<Box<str>>,
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub row: Option<R>,
    pub requests: u64,
}

/// The Responses API image generation tool. `items` are the completed calls, each the row and pixels
/// `default_image_cost_calculator` read for it, `None` where it raised
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct ImageGeneration<R = RowRef> {
    /// The tool model's row, `None` where `get_model_info` raised
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub row: Option<R>,
    /// `tool_usage.image_gen` when it reports tokens
    #[cfg_attr(feature = "serde", serde(default))]
    pub tokens: Option<ImageTokens>,
    pub items: Vec<Option<ImageItem<R>>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct ImageTokens {
    pub text_input: u64,
    pub image_input: u64,
    pub image_output: u64,
    pub text_output: u64,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct ImageItem<R = RowRef> {
    pub row: R,
    pub pixels: u64,
}

/// `get_cost_for_file_search`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct FileSearch<R = RowRef> {
    /// The request named a file search tool, without which main bills nothing
    pub requested: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    pub provider: Option<Box<str>>,
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub row: Option<R>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub storage_gb: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub days: Option<f64>,
    /// `OPENAI_FILE_SEARCH_COST_PER_1K_CALLS`, billed whole per call
    pub per_call: UnitPrice,
    /// `AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY`
    pub azure_gigabyte_day: UnitPrice,
}

/// `get_cost_for_vector_store`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct VectorStore<R = RowRef> {
    #[cfg_attr(feature = "serde", serde(default))]
    pub provider: Option<Box<str>>,
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub row: Option<R>,
    pub storage_gb: f64,
    pub days: f64,
    /// `AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY`
    pub azure_gigabyte_day: UnitPrice,
}

/// `get_cost_for_computer_use`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct ComputerUse<R = RowRef> {
    #[cfg_attr(feature = "serde", serde(default))]
    pub provider: Option<Box<str>>,
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub row: Option<R>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// `AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS` and `AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS`
    pub azure_input: UnitPrice,
    pub azure_output: UnitPrice,
}

/// `get_cost_for_code_interpreter`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(bound(deserialize = "R: serde::Deserialize<'de>"))
)]
pub struct CodeInterpreter<R = RowRef> {
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub row: Option<R>,
    /// The `<provider>/container` row main falls back to
    #[cfg_attr(feature = "serde", serde(default = "none"))]
    pub container_row: Option<R>,
    pub sessions: u64,
}

#[cfg(feature = "serde")]
fn none<T>() -> Option<T> {
    None
}

impl<R> Default for ToolUse<R> {
    fn default() -> Self {
        Self {
            provider: None,
            maps_grounding: None,
            image_generation: None,
            web_search: None,
            file_search: None,
            vector_store: None,
            computer_use: None,
            code_interpreter: None,
        }
    }
}

impl<R> Tools<R> {
    /// The same tools with each row named by `at`, a row `at` cannot name left out
    pub fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> Tools<T> {
        match self {
            Self::Call(used) => Tools::Call(Box::new(used.rows(at))),
            Self::One(tool) => Tools::One(tool.rows(at)),
        }
    }
}

impl<R> ToolUse<R> {
    /// The same tools with each row named by `at`, a row `at` cannot name left out
    pub fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> ToolUse<T> {
        ToolUse {
            provider: self.provider,
            maps_grounding: self.maps_grounding.map(|maps| maps.rows(at)),
            image_generation: self.image_generation.map(|image| image.rows(at)),
            web_search: self.web_search.map(|search| search.rows(at)),
            file_search: self.file_search.map(|files| files.rows(at)),
            vector_store: self.vector_store.map(|store| store.rows(at)),
            computer_use: self.computer_use.map(|computer| computer.rows(at)),
            code_interpreter: self.code_interpreter.map(|code| code.rows(at)),
        }
    }
}

impl<R> Tool<R> {
    fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> Tool<T> {
        match self {
            Self::WebSearch(search) => Tool::WebSearch(search.rows(at)),
            Self::MapsGrounding(maps) => Tool::MapsGrounding(maps.rows(at)),
            Self::FileSearch(files) => Tool::FileSearch(files.rows(at)),
            Self::VectorStore(store) => Tool::VectorStore(store.rows(at)),
            Self::ComputerUse(computer) => Tool::ComputerUse(computer.rows(at)),
            Self::CodeInterpreter(code) => Tool::CodeInterpreter(code.rows(at)),
        }
    }
}

impl<R> WebSearch<R> {
    fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> WebSearch<T> {
        WebSearch {
            dispatch: self.dispatch.map(|dispatch| Dispatch {
                provider: dispatch.provider,
                row: dispatch.row.and_then(at),
                served_row: dispatch.served_row.and_then(at),
                model_key: dispatch.model_key,
                response_model: dispatch.response_model,
                counts: dispatch.counts,
            }),
            per_call: self.per_call.map(|per_call| PerCall {
                row: per_call.row.and_then(at),
                context: per_call.context,
                calls: per_call.calls,
            }),
        }
    }
}

impl<R> MapsGrounding<R> {
    fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> MapsGrounding<T> {
        MapsGrounding {
            provider: self.provider,
            row: self.row.and_then(at),
            requests: self.requests,
        }
    }
}

impl<R> ImageGeneration<R> {
    fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> ImageGeneration<T> {
        ImageGeneration {
            row: self.row.and_then(at),
            tokens: self.tokens,
            items: self
                .items
                .into_iter()
                .map(|item| {
                    let item = item?;
                    Some(ImageItem {
                        row: at(item.row)?,
                        pixels: item.pixels,
                    })
                })
                .collect(),
        }
    }
}

impl<R> FileSearch<R> {
    fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> FileSearch<T> {
        FileSearch {
            requested: self.requested,
            provider: self.provider,
            row: self.row.and_then(at),
            storage_gb: self.storage_gb,
            days: self.days,
            per_call: self.per_call,
            azure_gigabyte_day: self.azure_gigabyte_day,
        }
    }
}

impl<R> VectorStore<R> {
    fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> VectorStore<T> {
        VectorStore {
            provider: self.provider,
            row: self.row.and_then(at),
            storage_gb: self.storage_gb,
            days: self.days,
            azure_gigabyte_day: self.azure_gigabyte_day,
        }
    }
}

impl<R> ComputerUse<R> {
    fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> ComputerUse<T> {
        ComputerUse {
            provider: self.provider,
            row: self.row.and_then(at),
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            azure_input: self.azure_input,
            azure_output: self.azure_output,
        }
    }
}

impl<R> CodeInterpreter<R> {
    fn rows<T>(self, at: &impl Fn(R) -> Option<T>) -> CodeInterpreter<T> {
        CodeInterpreter {
            row: self.row.and_then(at),
            container_row: self.container_row.and_then(at),
            sessions: self.sessions,
        }
    }
}

pub(crate) fn price<S: RowSource>(source: &S, query: &ToolsQuery) -> PriceOutcome {
    let fee = match &query.tools {
        Tools::Call(used) => Some(fees(source, used, &query.header)),
        Tools::One(tool) => one(source, tool),
    };
    let Some(fee) = fee else {
        return Unpriced::NoRow {
            tried: query.header.candidates.clone(),
        }
        .into();
    };
    let receipt = units::receipt(
        source,
        &query.header,
        None,
        Billed::of(Vec::new(), Vec::new()),
    );
    Receipt {
        tools: fee,
        ..receipt
    }
    .settled(source, &query.header.adjustments, None)
    .into()
}

/// What `get_cost_for_built_in_tools` bills, added in main's order
pub(crate) fn fees<S: RowSource>(source: &S, used: &ToolUse, header: &Header) -> Usd {
    let maps = used
        .maps_grounding
        .as_ref()
        .and_then(|maps| maps_grounding(source, maps))
        .unwrap_or(Usd::ZERO);
    let image = used
        .image_generation
        .as_ref()
        .map_or(Usd::ZERO, |image| image_generation(source, image, header));
    let rest = match (&used.web_search, &used.file_search) {
        (None, None) if used.provider.as_deref() == Some("azure") => [
            used.vector_store
                .as_ref()
                .map(|store| vector_store(source, store)),
            used.computer_use
                .as_ref()
                .map(|computer| computer_use(source, computer)),
            used.code_interpreter
                .as_ref()
                .map(|code| code_interpreter(source, code)),
        ]
        .into_iter()
        .flatten()
        .sum(),
        (None, None) => Usd::ZERO,
        (search, files) => {
            search
                .as_ref()
                .and_then(|search| web_search(source, search))
                .unwrap_or(Usd::ZERO)
                + files
                    .as_ref()
                    .map_or(Usd::ZERO, |files| file_search(source, files))
        }
    };
    maps + image + rest
}

fn one<S: RowSource>(source: &S, tool: &Tool) -> Option<Usd> {
    match tool {
        Tool::WebSearch(search) => web_search(source, search),
        Tool::MapsGrounding(maps) => maps_grounding(source, maps),
        Tool::FileSearch(files) => Some(file_search(source, files)),
        Tool::VectorStore(store) => Some(vector_store(source, store)),
        Tool::ComputerUse(computer) => Some(computer_use(source, computer)),
        Tool::CodeInterpreter(code) => Some(code_interpreter(source, code)),
    }
}

fn rate<S: RowSource>(source: &S, row: Option<&RowRef>, meter: &Meter) -> Option<UnitPrice> {
    units::listed(source.row(row?)?, meter, Channel::Online)
}

fn search_rate<S: RowSource>(
    source: &S,
    row: Option<&RowRef>,
    size: SearchContextSize,
) -> Option<UnitPrice> {
    rate(source, row, &Meter::WebSearch(size))
}

fn per_query<S: RowSource>(source: &S, row: Option<&RowRef>) -> bool {
    row.and_then(|at| source.row(at))
        .is_some_and(|row| row.prices().flat().billing().web_search == WebSearchUnit::PerQuery)
}

fn times(rate: UnitPrice, count: u64) -> Usd {
    rate * Quantity::Count(count)
}

fn fixed(usd: f64) -> UnitPrice {
    UnitPrice::new(usd).unwrap_or(UnitPrice::ZERO)
}

/// `_handle_web_search_cost`: the provider calculator's fee, else the price per call times the billable calls
pub(crate) fn web_search<S: RowSource>(source: &S, search: &WebSearch) -> Option<Usd> {
    search
        .dispatch
        .as_ref()
        .and_then(|dispatch| dispatched(source, dispatch))
        .or_else(|| {
            let per_call = search.per_call.as_ref()?;
            Some(times(
                call_rate(source, per_call.row.as_ref(), per_call.context),
                per_call.calls,
            ))
        })
}

/// `get_cost_for_web_search`: the requested context size's price, the medium one when the request names none
pub(crate) fn call_rate<S: RowSource>(
    source: &S,
    row: Option<&RowRef>,
    context: Option<SearchContextSize>,
) -> UnitPrice {
    search_rate(source, row, context.unwrap_or(SearchContextSize::Medium))
        .unwrap_or(UnitPrice::ZERO)
}

/// `get_cost_for_web_search_request`, `None` for a provider without a calculator of its own
fn dispatched<S: RowSource>(source: &S, dispatch: &Dispatch) -> Option<Usd> {
    let counts = &dispatch.counts;
    let row = dispatch.row.as_ref();
    let names_claude = |name: &str| name.to_lowercase().contains("claude");
    let anthropic = || {
        let declares = SIZES
            .iter()
            .any(|size| search_rate(source, row, *size).is_some());
        let priced_by = match (declares, &dispatch.served_row) {
            (false, Some(served)) => Some(served),
            _ => row,
        };
        anthropic_fee(
            search_rate(source, priced_by, SearchContextSize::Medium),
            counts.server_requests,
        )
    };
    match &*dispatch.provider {
        "gemini" => Some(gemini(source, row, counts)),
        "anthropic" => Some(anthropic()),
        vertex if vertex.starts_with("vertex_ai") => {
            let named = dispatch
                .response_model
                .as_deref()
                .filter(|model| !model.is_empty())
                .unwrap_or(&dispatch.model_key);
            Some(match names_claude(named) {
                true => anthropic(),
                false => gemini(source, row, counts),
            })
        }
        "azure_ai" if !names_claude(&dispatch.model_key) && counts.server_requests.is_none() => {
            None
        }
        "azure_ai" => Some(anthropic()),
        "perplexity" => Some(Usd::ZERO),
        "xai" => Some(xai(source, row, counts)),
        "groq" => Some(groq(source, row, counts)),
        _ => None,
    }
}

/// `get_cost_for_anthropic_web_search`: each reported request at the medium context price
pub(crate) fn anthropic_fee(medium: Option<UnitPrice>, requests: Option<u64>) -> Usd {
    match (medium, requests) {
        (Some(rate), Some(requests)) if rate.get() != 0.0 => times(rate, requests),
        _ => Usd::ZERO,
    }
}

/// Gemini's and Vertex AI's grounding fee: per query, or once per grounded prompt
fn gemini<S: RowSource>(source: &S, row: Option<&RowRef>, counts: &SearchCounts) -> Usd {
    let medium =
        search_rate(source, row, SearchContextSize::Medium).unwrap_or(fixed(GEMINI_WEB_SEARCH));
    let requests = [counts.prompt_requests, counts.server_requests]
        .into_iter()
        .flatten()
        .find(|requests| *requests > 0)
        .unwrap_or(0);
    let billed = match per_query(source, row) {
        false if requests > 0 => 1,
        _ => requests,
    };
    times(medium, billed)
}

/// xAI's fee per web search call, nothing when the usage states the call's cost
fn xai<S: RowSource>(source: &S, row: Option<&RowRef>, counts: &SearchCounts) -> Usd {
    if counts.stated_cost || counts.search_calls == 0 {
        return Usd::ZERO;
    }
    let per_call = [
        SearchContextSize::Medium,
        SearchContextSize::Low,
        SearchContextSize::High,
    ]
    .into_iter()
    .filter_map(|size| search_rate(source, row, size))
    .find(|rate| rate.get() > 0.0)
    .unwrap_or(fixed(XAI_WEB_SEARCH));
    times(per_call, counts.search_calls)
}

/// Groq's browser tool: each search at the medium context price and each page visit at the fixed visit price
fn groq<S: RowSource>(source: &S, row: Option<&RowRef>, counts: &SearchCounts) -> Usd {
    let medium = search_rate(source, row, SearchContextSize::Medium).unwrap_or(UnitPrice::ZERO);
    times(medium, counts.server_requests.unwrap_or(0))
        + times(fixed(GROQ_BROWSER_OPEN), counts.browser_opens.unwrap_or(0))
}

/// `get_cost_for_google_maps_grounding_request`, `None` for a provider other than Gemini or Vertex AI
fn maps_grounding<S: RowSource>(source: &S, maps: &MapsGrounding) -> Option<Usd> {
    if let Some(provider) = &maps.provider
        && provider.as_ref() != "gemini"
        && !provider.starts_with("vertex_ai")
    {
        return None;
    }
    if maps.requests == 0 {
        return Some(Usd::ZERO);
    }
    let row = maps.row.as_ref();
    let per_query = per_query(source, row);
    let listed = rate(
        source,
        row,
        &Meter::Feature(Feature::GoogleMapsGroundingQuery),
    );
    let price = listed.unwrap_or(fixed(match per_query {
        true => MAPS_PER_QUERY,
        false => MAPS_PER_PROMPT,
    }));
    Some(times(price, if per_query { maps.requests } else { 1 }))
}

/// `_handle_image_generation_cost`: the tool's reported tokens at the tool model's rates when they bill anything,
/// else each completed call through `default_image_cost_calculator`
fn image_generation<S: RowSource>(source: &S, image: &ImageGeneration, header: &Header) -> Usd {
    let row = image.row.as_ref();
    let token_cost = image.tokens.and_then(|tokens| {
        row?;
        let by = |meter: Meter, count: u64| {
            times(rate(source, row, &meter).unwrap_or(UnitPrice::ZERO), count)
        };
        let cost = by(
            Meter::Input(InputUnit::Token(TokenKind::Text)),
            tokens.text_input,
        ) + by(
            Meter::Input(InputUnit::Token(TokenKind::Image)),
            tokens.image_input,
        ) + by(
            Meter::Output(OutputUnit::Token(TokenKind::Image)),
            tokens.image_output,
        ) + by(
            Meter::Output(OutputUnit::Token(TokenKind::Text)),
            tokens.text_output,
        );
        (cost.get() > 0.0).then_some(cost)
    });
    token_cost.unwrap_or_else(|| {
        image
            .items
            .iter()
            .map(|item| {
                item.as_ref()
                    .map_or(Usd::ZERO, |item| image_item(source, item, header))
            })
            .sum()
    })
}

fn image_item<S: RowSource>(source: &S, item: &ImageItem, header: &Header) -> Usd {
    let query = ImageQuery {
        header: Header {
            candidates: Vec::new(),
            source: PriceSource::Rows {
                rows: vec![item.row.clone()],
            },
            basis: header.basis.clone(),
            adjustments: crate::query::AdjustmentPolicy::None,
            reported: None,
        },
        calculator: ImageCalculator::Default,
        images: 1,
        pixels: item.pixels,
        input_images: 0,
    };
    match crate::image::price(source, &query) {
        PriceOutcome::Priced(receipt) => Usd::new(receipt.total).unwrap_or(Usd::ZERO),
        PriceOutcome::Unpriced(_) => Usd::ZERO,
    }
}

fn nonzero(value: Option<f64>) -> Option<f64> {
    value.filter(|value| *value != 0.0)
}

/// `get_cost_for_file_search`. The row's price per thousand calls is billed whole, as main bills it
fn file_search<S: RowSource>(source: &S, files: &FileSearch) -> Usd {
    if !files.requested {
        return Usd::ZERO;
    }
    let row = files.row.as_ref();
    let azure = files.provider.as_deref() == Some("azure");
    let stored = nonzero(files.storage_gb).zip(nonzero(files.days));
    let by_storage = |price: UnitPrice| {
        stored.map_or(Usd::ZERO, |(gb, days)| {
            price * Quantity::Measured(gb * days)
        })
    };
    let storage_rate = rate(source, row, &Meter::Feature(Feature::FileSearchGigabyteDay));
    let call_rate = rate(
        source,
        row,
        &Meter::Feature(Feature::FileSearchThousandCalls),
    );
    match (storage_rate.filter(|_| azure), call_rate) {
        (Some(price), _) => by_storage(price),
        (None, Some(price)) => times(price, 1),
        (None, None) if azure => by_storage(files.azure_gigabyte_day),
        (None, None) => times(files.per_call, 1),
    }
}

/// `get_cost_for_vector_store`
fn vector_store<S: RowSource>(source: &S, store: &VectorStore) -> Usd {
    let stored = Quantity::Measured(store.storage_gb * store.days);
    match rate(
        source,
        store.row.as_ref(),
        &Meter::Feature(Feature::VectorStoreGigabyteDay),
    ) {
        Some(price) => price * stored,
        None if store.provider.as_deref() == Some("azure") => store.azure_gigabyte_day * stored,
        None => Usd::ZERO,
    }
}

/// `get_cost_for_computer_use`: Azure bills thousands of tokens at the row's prices, else at the configured ones
fn computer_use<S: RowSource>(source: &S, computer: &ComputerUse) -> Usd {
    let used = computer.input_tokens > 0 || computer.output_tokens > 0;
    if computer.provider.as_deref() != Some("azure") || !used {
        return Usd::ZERO;
    }
    let row = computer.row.as_ref();
    let listed = |feature| rate(source, row, &Meter::Feature(feature)).unwrap_or(UnitPrice::ZERO);
    let row_rates = (
        listed(Feature::ComputerUseInputThousandTokens),
        listed(Feature::ComputerUseOutputThousandTokens),
    );
    let (input, output) =
        match row.is_some() && (row_rates.0.get() != 0.0 || row_rates.1.get() != 0.0) {
            true => row_rates,
            false => (computer.azure_input, computer.azure_output),
        };
    let thousands = |tokens: u64| Quantity::Measured(tokens as f64 / 1000.0);
    let input_cost = match computer.input_tokens {
        0 => Usd::ZERO,
        tokens => input * thousands(tokens),
    };
    match computer.output_tokens {
        0 => input_cost,
        tokens => input_cost + output * thousands(tokens),
    }
}

/// `get_cost_for_code_interpreter`: the sessions at the row's price, else at the provider's container row's
fn code_interpreter<S: RowSource>(source: &S, code: &CodeInterpreter) -> Usd {
    let meter = Meter::Feature(Feature::CodeInterpreterSession);
    if code.sessions == 0 {
        return Usd::ZERO;
    }
    rate(source, code.row.as_ref(), &meter)
        .or_else(|| rate(source, code.container_row.as_ref(), &meter))
        .map_or(Usd::ZERO, |price| times(price, code.sessions))
}

#[cfg(test)]
mod parity {
    //! Parity with main's built-in tool calculators. `fixtures/tools_parity.json` was captured once from main's Python
    //! (532df6ef85, `LITELLM_LOCAL_MODEL_COST_MAP=True`): each case holds the rows the calculator read, the counts the
    //! response reported and main's fee, `null` where main's calculator returned `None`. Every fee must match to the
    //! last bit

    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::Value;

    use super::*;
    use crate::price::{Billing, Price, Prices, When};
    use crate::query::AdjustmentPolicy;
    use crate::testing::{TestRow, TestSource, basis, fixture_case, row_ref, units_row};

    const FIXTURE: &str = include_str!("../fixtures/tools_parity.json");

    fn tools_row(fields: &Value) -> TestRow {
        let features = [
            (
                "google_maps_grounding_cost_per_query",
                Feature::GoogleMapsGroundingQuery,
            ),
            (
                "code_interpreter_cost_per_session",
                Feature::CodeInterpreterSession,
            ),
            (
                "file_search_cost_per_1k_calls",
                Feature::FileSearchThousandCalls,
            ),
            (
                "file_search_cost_per_gb_per_day",
                Feature::FileSearchGigabyteDay,
            ),
            (
                "vector_store_cost_per_gb_per_day",
                Feature::VectorStoreGigabyteDay,
            ),
            (
                "computer_use_input_cost_per_1k_tokens",
                Feature::ComputerUseInputThousandTokens,
            ),
            (
                "computer_use_output_cost_per_1k_tokens",
                Feature::ComputerUseOutputThousandTokens,
            ),
        ]
        .map(|(key, feature)| (key, Meter::Feature(feature)));
        let image_tokens = [
            (
                "input_cost_per_image_token",
                Meter::Input(InputUnit::Token(TokenKind::Image)),
            ),
            (
                "output_cost_per_image_token",
                Meter::Output(OutputUnit::Token(TokenKind::Image)),
            ),
        ];
        let sizes = [
            ("search_context_size_low", SearchContextSize::Low),
            ("search_context_size_medium", SearchContextSize::Medium),
            ("search_context_size_high", SearchContextSize::High),
        ];
        let rate = |value: &Value| UnitPrice::new(value.as_f64()?);
        let flat = features
            .into_iter()
            .chain(image_tokens)
            .filter_map(|(key, meter)| Some((meter, rate(fields.get(key)?)?)));
        let search = sizes.into_iter().filter_map(|(key, size)| {
            let listed = fields.get("search_context_cost_per_query")?.get(key)?;
            Some((Meter::WebSearch(size), rate(listed)?))
        });
        let rates: BTreeMap<Price, UnitPrice> = units_row(fields)
            .prices
            .iter()
            .map(|(price, rate)| (price.clone(), rate))
            .chain(
                flat.chain(search)
                    .map(|(meter, rate)| (Price::online(meter, When::default()), rate)),
            )
            .collect();
        let billing = Billing {
            web_search: match fields
                .get("web_search_billing_unit")
                .and_then(Value::as_str)
            {
                Some("per_query") => WebSearchUnit::PerQuery,
                _ => WebSearchUnit::PerPrompt,
            },
            ..Billing::default()
        };
        TestRow {
            prices: Prices::new(rates, None, billing),
            ..TestRow::default()
        }
    }

    fn source(case: &Value) -> TestSource {
        case["rows"]
            .as_object()
            .unwrap()
            .iter()
            .fold(TestSource::default(), |source, (name, fields)| {
                source.and_row(name, tools_row(fields))
            })
    }

    fn flagged(facts: &Value, flag: &str, name: &str) -> Option<RowRef> {
        facts[flag]
            .as_bool()
            .unwrap_or(false)
            .then(|| row_ref(name))
    }

    fn text(value: &Value) -> Option<Box<str>> {
        value.as_str().map(Into::into)
    }

    fn count(value: &Value) -> u64 {
        value.as_u64().unwrap_or(0)
    }

    fn context(value: &Value) -> Option<SearchContextSize> {
        match value.as_str()? {
            "low" => Some(SearchContextSize::Low),
            "medium" => Some(SearchContextSize::Medium),
            "high" => Some(SearchContextSize::High),
            _ => None,
        }
    }

    fn configured(per_unit: f64) -> UnitPrice {
        UnitPrice::new(per_unit).unwrap()
    }

    fn dispatch(case: &Value, dispatch: &Value) -> Dispatch {
        let counts = &dispatch["counts"];
        Dispatch {
            provider: text(&dispatch["provider"]).unwrap(),
            row: case["rows"].get("row").map(|_| row_ref("row")),
            served_row: flagged(dispatch, "served", "served"),
            model_key: text(&dispatch["model_key"]).unwrap(),
            response_model: text(&dispatch["response_model"]),
            counts: SearchCounts {
                prompt_requests: counts["prompt_requests"].as_u64(),
                server_requests: counts["server_requests"].as_u64(),
                browser_opens: counts["browser_opens"].as_u64(),
                search_calls: count(&counts["search_calls"]),
                stated_cost: counts["stated_cost"].as_bool().unwrap(),
            },
        }
    }

    fn tool(case: &Value) -> Tool {
        let facts = &case["facts"];
        let given_row = || case["rows"].get("row").map(|_| row_ref("row"));
        match case["tool"].as_str().unwrap() {
            "web_search" => Tool::WebSearch(WebSearch {
                dispatch: (!facts["dispatch"].is_null())
                    .then(|| dispatch(case, &facts["dispatch"])),
                per_call: (!facts["per_call"].is_null()).then(|| PerCall {
                    row: flagged(&facts["per_call"], "row", "row"),
                    context: context(&facts["per_call"]["context"]),
                    calls: count(&facts["per_call"]["calls"]),
                }),
            }),
            "maps_grounding" => Tool::MapsGrounding(MapsGrounding {
                provider: text(&facts["provider"]),
                row: given_row(),
                requests: count(&facts["requests"]),
            }),
            "file_search" => Tool::FileSearch(FileSearch {
                requested: true,
                provider: text(&facts["provider"]),
                row: flagged(facts, "row", "row"),
                storage_gb: facts["storage_gb"].as_f64(),
                days: facts["days"].as_f64(),
                per_call: configured(2.5 / 1000.0),
                azure_gigabyte_day: configured(0.1),
            }),
            "vector_store" => Tool::VectorStore(VectorStore {
                provider: text(&facts["provider"]),
                row: flagged(facts, "row", "row"),
                storage_gb: facts["storage_gb"].as_f64().unwrap(),
                days: facts["days"].as_f64().unwrap(),
                azure_gigabyte_day: configured(0.1),
            }),
            "computer_use" => Tool::ComputerUse(ComputerUse {
                provider: text(&facts["provider"]),
                row: flagged(facts, "row", "row"),
                input_tokens: count(&facts["input_tokens"]),
                output_tokens: count(&facts["output_tokens"]),
                azure_input: configured(3.0),
                azure_output: configured(12.0),
            }),
            "code_interpreter" => Tool::CodeInterpreter(CodeInterpreter {
                row: flagged(facts, "row", "row"),
                container_row: flagged(facts, "container", "container"),
                sessions: count(&facts["sessions"]),
            }),
            other => panic!("no single tool {other}"),
        }
    }

    fn fee(case: &Value, tools: Tools) -> Option<f64> {
        let header = Header {
            candidates: Vec::new(),
            source: PriceSource::Rows { rows: Vec::new() },
            basis: basis(),
            adjustments: AdjustmentPolicy::None,
            reported: None,
        };
        match price(&source(case), &ToolsQuery { header, tools }) {
            PriceOutcome::Priced(receipt) => Some(receipt.tools.get()),
            PriceOutcome::Unpriced(_) => None,
        }
    }

    #[rstest]
    #[case::gemini_per_prompt("dispatch_gemini_per_prompt")]
    #[case::gemini_per_query("dispatch_gemini_per_query")]
    #[case::gemini_server_requests("dispatch_gemini_server_requests_when_prompt_zero")]
    #[case::gemini_default_rate("dispatch_gemini_default_rate")]
    #[case::gemini_no_requests("dispatch_gemini_no_requests")]
    #[case::vertex_gemini("dispatch_vertex_gemini")]
    #[case::vertex_claude_served_row("dispatch_vertex_claude_served_row")]
    #[case::vertex_claude_by_key("dispatch_vertex_claude_by_key")]
    #[case::anthropic("dispatch_anthropic")]
    #[case::anthropic_served_row("dispatch_anthropic_served_row")]
    #[case::anthropic_no_requests("dispatch_anthropic_no_requests")]
    #[case::azure_ai_claude("dispatch_azure_ai_claude")]
    #[case::azure_ai_other_without_requests("dispatch_azure_ai_other_without_requests")]
    #[case::azure_ai_other_with_requests("dispatch_azure_ai_other_with_requests")]
    #[case::perplexity("dispatch_perplexity")]
    #[case::xai_first_positive_rate("dispatch_xai_first_positive_rate")]
    #[case::xai_default_rate("dispatch_xai_default_rate")]
    #[case::xai_stated_cost("dispatch_xai_stated_cost")]
    #[case::groq("dispatch_groq")]
    #[case::groq_opens_only("dispatch_groq_opens_only")]
    #[case::openai_has_no_calculator("dispatch_openai_has_no_calculator")]
    #[case::per_call_medium_default("per_call_medium_default")]
    #[case::per_call_low("per_call_low")]
    #[case::per_call_high("per_call_high")]
    #[case::per_call_unknown_size("per_call_unknown_size_is_medium")]
    #[case::per_call_no_row("per_call_no_row")]
    #[case::per_call_row_without_search("per_call_row_without_search")]
    #[case::maps_gemini("maps_gemini_per_prompt_listed")]
    #[case::maps_vertex_per_query("maps_vertex_per_query_listed")]
    #[case::maps_per_query_default("maps_per_query_default")]
    #[case::maps_per_prompt_default("maps_per_prompt_default")]
    #[case::maps_zero_requests("maps_zero_requests")]
    #[case::maps_other_provider("maps_other_provider")]
    #[case::maps_without_provider("maps_without_provider")]
    #[case::file_search_openai_default("file_search_openai_default")]
    #[case::file_search_azure_storage("file_search_azure_storage")]
    #[case::file_search_azure_without_storage("file_search_azure_without_storage")]
    #[case::file_search_azure_row_storage("file_search_azure_row_storage")]
    #[case::file_search_row_storage_without_days("file_search_azure_row_storage_without_days")]
    #[case::file_search_per_thousand_calls("file_search_row_per_thousand_calls")]
    #[case::file_search_storage_off_azure("file_search_storage_row_off_azure_reads_calls")]
    #[case::file_search_no_provider("file_search_no_provider")]
    #[case::vector_store_azure_default("vector_store_azure_default")]
    #[case::vector_store_row_rate("vector_store_row_rate_any_provider")]
    #[case::vector_store_openai_free("vector_store_openai_free")]
    #[case::computer_use_azure_default("computer_use_azure_default")]
    #[case::computer_use_row_rates("computer_use_azure_row_rates")]
    #[case::computer_use_row_output_only("computer_use_azure_row_output_only")]
    #[case::computer_use_zero_rates("computer_use_azure_row_zero_rates_use_default")]
    #[case::computer_use_input_only("computer_use_azure_input_only")]
    #[case::computer_use_openai_free("computer_use_openai_free")]
    #[case::computer_use_no_tokens("computer_use_azure_no_tokens")]
    #[case::code_interpreter_row_rate("code_interpreter_row_rate")]
    #[case::code_interpreter_azure_container("code_interpreter_azure_container")]
    #[case::code_interpreter_openai_container("code_interpreter_openai_container")]
    #[case::code_interpreter_no_sessions("code_interpreter_no_sessions")]
    #[case::code_interpreter_no_provider("code_interpreter_no_provider")]
    fn one_tool_matches_main(#[case] name: &str) {
        let case = fixture_case(FIXTURE, name);

        assert_eq!(
            fee(&case, Tools::One(tool(&case))),
            case["expected"].as_f64()
        );
    }

    #[rstest]
    #[case::tokens("image_generation_tokens")]
    #[case::items("image_generation_items")]
    #[case::zero_tokens_fall_back_to_items("image_generation_zero_tokens_fall_back_to_items")]
    #[case::unpriced_item("image_generation_unpriced_item_bills_nothing")]
    fn the_image_generation_tool_matches_main(#[case] name: &str) {
        let case = fixture_case(FIXTURE, name);
        let facts = &case["facts"];
        let tokens = facts["tokens"].as_object().map(|tokens| ImageTokens {
            text_input: count(&tokens["text_input"]),
            image_input: count(&tokens["image_input"]),
            image_output: count(&tokens["image_output"]),
            text_output: count(&tokens["text_output"]),
        });
        let items = facts["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                Some(ImageItem {
                    row: row_ref(item.get("row")?.as_str()?),
                    pixels: count(&item["pixels"]),
                })
            })
            .collect();
        let used = ToolUse {
            image_generation: Some(ImageGeneration {
                row: Some(row_ref("tool")),
                tokens,
                items,
            }),
            ..ToolUse::default()
        };

        assert_eq!(
            fee(&case, Tools::Call(Box::new(used))),
            case["expected"].as_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::{Price, Prices, When};
    use crate::query::AdjustmentPolicy;
    use crate::testing::{TestRow, TestSource, basis, row_ref};

    fn rated(meter: Meter, rate: f64) -> TestRow {
        TestRow {
            prices: Prices::new(
                [(
                    Price::online(meter, When::default()),
                    UnitPrice::new(rate).unwrap(),
                )]
                .into(),
                None,
                crate::price::Billing::default(),
            ),
            ..TestRow::default()
        }
    }

    fn header() -> Header {
        Header {
            candidates: Vec::new(),
            source: PriceSource::Rows { rows: Vec::new() },
            basis: basis(),
            adjustments: AdjustmentPolicy::None,
            reported: None,
        }
    }

    fn per_call(calls: u64) -> WebSearch {
        WebSearch {
            dispatch: None,
            per_call: Some(PerCall {
                row: Some(row_ref("search")),
                context: Some(SearchContextSize::Low),
                calls,
            }),
        }
    }

    fn files() -> FileSearch {
        FileSearch {
            requested: true,
            provider: Some("openai".into()),
            row: None,
            storage_gb: None,
            days: None,
            per_call: UnitPrice::new(4.0).unwrap(),
            azure_gigabyte_day: UnitPrice::ZERO,
        }
    }

    fn maps() -> MapsGrounding {
        MapsGrounding {
            provider: Some("gemini".into()),
            row: Some(row_ref("maps")),
            requests: 3,
        }
    }

    fn source() -> TestSource {
        TestSource::with_row(
            "search",
            rated(Meter::WebSearch(SearchContextSize::Low), 2.0),
        )
        .and_row(
            "maps",
            rated(Meter::Feature(Feature::GoogleMapsGroundingQuery), 0.5),
        )
    }

    #[rstest]
    #[case::web_and_file_search(Some(per_call(3)), Some(files()), 10.0)]
    #[case::web_search_alone(Some(per_call(3)), None, 6.0)]
    #[case::file_search_alone(None, Some(files()), 4.0)]
    #[case::neither(None, None, 0.0)]
    fn a_call_bills_maps_grounding_beside_its_web_and_file_search(
        #[case] web_search: Option<WebSearch>,
        #[case] file_search: Option<FileSearch>,
        #[case] beside_maps: f64,
    ) {
        let used = ToolUse {
            maps_grounding: Some(maps()),
            web_search,
            file_search,
            ..ToolUse::default()
        };

        assert_eq!(fees(&source(), &used, &header()).get(), 0.5 + beside_maps);
    }

    #[rstest]
    #[case::azure(Some("azure"), 3.0)]
    #[case::openai(Some("openai"), 0.0)]
    fn the_assistant_tools_bill_only_an_azure_call(
        #[case] provider: Option<&str>,
        #[case] expected: f64,
    ) {
        let used = ToolUse {
            provider: provider.map(Into::into),
            code_interpreter: Some(CodeInterpreter {
                row: Some(row_ref("code")),
                container_row: None,
                sessions: 2,
            }),
            ..ToolUse::default()
        };
        let source = TestSource::with_row(
            "code",
            rated(Meter::Feature(Feature::CodeInterpreterSession), 1.5),
        );

        assert_eq!(fees(&source, &used, &header()).get(), expected);
    }

    #[rstest]
    fn a_tool_without_a_calculator_is_unpriced_not_free() {
        let query = ToolsQuery {
            header: header(),
            tools: Tools::One(Tool::MapsGrounding(MapsGrounding {
                provider: Some("openai".into()),
                ..maps()
            })),
        };

        assert!(matches!(
            price(&source(), &query),
            PriceOutcome::Unpriced(_)
        ));
    }
}
