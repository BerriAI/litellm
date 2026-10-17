//! `litellm/llms/azure/chat/gpt_transformation.py`, `o_series_transformation.py` and
//! `gpt_5_transformation.py` over an OpenAI-shaped body: Python's `get_optional_params` sorts
//! the request's OpenAI params from the rest, refuses or drops the ones the deployment's model
//! family does not take, maps the rest by that family's rules, and passes every other key to
//! Azure through `extra_body`. The response reaches the caller as Azure sent it.

use serde_json::{Map, Value};

use crate::{
    Error, ErrorDetail,
    base_llm::{
        chat::params::{
            OPENAI_PARAMS, UNVALIDATED_PARAMS, ordered, request_drop_params, string_list, truthy,
        },
        model_info::{self, ModelInfo},
    },
    openai::chat::transformation::{is_litellm_owned, take_extra_body},
};

/// `AzureOpenAIConfig.get_supported_openai_params`.
const AZURE_GPT_PARAMS: &[&str] = &[
    "temperature",
    "n",
    "stream",
    "stream_options",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "function_call",
    "functions",
    "top_p",
    "logprobs",
    "top_logprobs",
    "response_format",
    "seed",
    "extra_headers",
    "parallel_tool_calls",
    "prediction",
    "modalities",
    "audio",
    "web_search_options",
    "prompt_cache_key",
    "store",
];

/// The `base_params` of `OpenAIGPTConfig.get_supported_openai_params`, which the o-series and
/// gpt-5 configs start from.
const OPENAI_GPT_PARAMS: &[&str] = &[
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "modalities",
    "prediction",
    "n",
    "presence_penalty",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "function_call",
    "functions",
    "max_retries",
    "extra_headers",
    "parallel_tool_calls",
    "audio",
    "web_search_options",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "store",
];

/// `AzureOpenAIO1Config`'s `non_supported_params`.
const O_SERIES_UNSUPPORTED: &[&str] = &[
    "logprobs",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "top_logprobs",
];

/// `OpenAIGPT5Config`'s `non_supported_params`.
const GPT_5_UNSUPPORTED: &[&str] = &[
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "logit_bias",
    "modalities",
    "prediction",
    "audio",
    "web_search_options",
];

/// The params of a gpt-5 search model.
const GPT_5_SEARCH_PARAMS: &[&str] = &[
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "web_search_options",
    "service_tier",
    "safety_identifier",
    "response_format",
    "user",
    "store",
    "verbosity",
    "max_retries",
    "extra_headers",
];

const GPT_5_SAMPLING_PARAMS: &[&str] = &["logprobs", "top_logprobs", "top_p"];
const REASONING_SUMMARY_ALIASES: &[&str] = &["reasoningSummary", "reasoning_summary"];
const O_SERIES_ROUTE: &str = "o_series/";
const GPT_5_SERIES_ROUTE: &str = "gpt5_series/";
const RESPONSES_ROUTE: &str = "responses/";
/// `API_VERSION_YEAR_SUPPORTED_RESPONSE_FORMAT` and its month.
const RESPONSE_FORMAT_SINCE: (u32, u32) = (2024, 8);
const DROP_HINT: &str = "Set `drop_params: true` under litellm_settings to drop it. ";

/// Which of Python's Azure configs serves a deployment, picked from `base_model` when the
/// deployment sets one and from the deployment name otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzureModelFamily {
    /// `AzureOpenAIConfig`.
    Gpt,
    /// `AzureOpenAIO1Config`.
    OSeries,
    /// `AzureOpenAIGPT5Config`.
    Gpt5,
}

impl AzureModelFamily {
    pub fn detect(model: &str) -> Self {
        if is_o_series_model(model) {
            Self::OSeries
        } else if is_gpt_5_model(model) {
            Self::Gpt5
        } else {
            Self::Gpt
        }
    }
}

/// `AzureOpenAIO1Config.is_o_series_model`.
fn is_o_series_model(model: &str) -> bool {
    ["o1", "o3", "o4", O_SERIES_ROUTE]
        .iter()
        .any(|marker| model.contains(marker))
}

/// `is_gpt_reasoning_series_name`: a gpt-5 or gpt-6 name, except the gpt-5-chat models.
fn is_gpt_reasoning_series_name(model: &str) -> bool {
    let name = model.rsplit('/').next().unwrap_or(model);
    (model.contains("gpt-5") || model.contains("gpt-6")) && !name.contains("gpt-5-chat")
}

/// `AzureOpenAIGPT5Config.is_model_gpt_5_model`.
fn is_gpt_5_model(model: &str) -> bool {
    is_gpt_reasoning_series_name(model) || model.contains("gpt5_series")
}

fn is_gpt_5_search_model(model: &str) -> bool {
    model.contains("gpt-5") && model.contains("search")
}

/// `is_model_gpt_5_2_model`: gpt-5.2 and gpt-5.4, pro included.
fn is_gpt_5_2_model(model: &str) -> bool {
    let name = model.rsplit('/').next().unwrap_or(model);
    name.starts_with("gpt-5.2") || name.starts_with("gpt-5.4")
}

/// `_gpt_series_version`: the `(major, minor)` of a `gpt-N[.M]` name, as
/// `^gpt-(\d+)(?:\.(\d+))?(?=[.-]|$)` reads it.
fn gpt_series_version(model: &str) -> Option<(u32, u32)> {
    let name = model.rsplit('/').next().unwrap_or(model);
    let rest = name.strip_prefix("gpt-")?;
    let digits = |text: &str| {
        text.find(|c: char| !c.is_ascii_digit())
            .unwrap_or(text.len())
    };
    let boundary = |text: &str| text.is_empty() || text.starts_with(['.', '-']);
    let major_len = digits(rest);
    let major = rest[..major_len].parse().ok()?;
    let after_major = &rest[major_len..];
    if let Some(minor) = after_major.strip_prefix('.') {
        let minor_len = digits(minor);
        if minor_len > 0 && boundary(&minor[minor_len..]) {
            return Some((major, minor[..minor_len].parse().ok()?));
        }
    }
    boundary(after_major).then_some((major, 0))
}

/// `is_model_gpt_5_4_plus_model`.
fn is_gpt_5_4_plus_model(model: &str) -> bool {
    gpt_series_version(model).is_some_and(|version| version >= (5, 4))
}

/// `AzureOpenAIGPT5Config._model_map_lookup_name`: the cost map key of a gpt-5 deployment.
fn gpt_5_lookup_name(model: &str) -> String {
    match model.strip_prefix(GPT_5_SERIES_ROUTE) {
        Some(rest) => format!("azure/{rest}"),
        None if model.starts_with("azure/") => model.to_string(),
        None => format!("azure/{model}"),
    }
}

/// `get_effort_level`: a `reasoning_effort` string, or the `effort` of its object form.
fn effort_level(value: Option<&Value>) -> Option<&str> {
    match value? {
        Value::String(effort) => Some(effort),
        Value::Object(object) => object.get("effort").and_then(Value::as_str),
        _ => None,
    }
}

/// What a request to one Azure deployment is mapped with.
pub struct AzureChatRequest<'a> {
    /// The deployment, without its `azure/` prefix.
    pub deployment: &'a str,
    /// The deployment's `base_model`, which picks the config and the cost map entry when the
    /// deployment name says nothing about the model.
    pub base_model: Option<&'a str>,
    pub api_version: &'a str,
    /// `drop_params` from the deployment or the proxy settings. The request's own
    /// `drop_params` is honoured too.
    pub drop_params: bool,
    pub model_info: Option<&'a dyn ModelInfo>,
}

/// The body to send and the deployment the SDK puts in its URL.
#[derive(Debug, PartialEq)]
pub struct AzureChatBody {
    pub deployment: String,
    pub body: Map<String, Value>,
}

impl AzureChatRequest<'_> {
    /// `base_model or model`, the name Python picks the config from.
    fn detection_model(&self) -> &str {
        self.base_model
            .filter(|model| !model.is_empty())
            .unwrap_or(self.deployment)
    }

    pub fn family(&self) -> AzureModelFamily {
        AzureModelFamily::detect(self.detection_model())
    }

    fn info(&self) -> Option<&dyn ModelInfo> {
        self.model_info
    }

    pub fn transform(&self, body: Map<String, Value>) -> Result<AzureChatBody, Error> {
        let mut body = body;
        body.remove("model");
        let messages = body.remove("messages").unwrap_or(Value::Array(Vec::new()));
        let mut extra_body = take_extra_body(&mut body)?;
        // `main.py` takes a token from here as the call's credential; it never reaches Azure.
        extra_body.remove("azure_ad_token");
        extra_body.remove("model");
        let drop_params = self.drop_params || request_drop_params(body.remove("drop_params"));
        let allowed = string_list(body.remove("allowed_openai_params"));
        let additional_drop = string_list(body.remove("additional_drop_params"));
        let family = self.family();
        if family == AzureModelFamily::Gpt5 {
            for alias in REASONING_SUMMARY_ALIASES {
                extra_body.remove(*alias);
            }
        }
        if let Some(reason) = self.responses_bridge(&body, &extra_body) {
            return Err(Error::Unsupported(reason));
        }

        let mut non_default = Map::new();
        for (name, value) in body {
            if value.is_null()
                || is_litellm_owned(&name)
                || additional_drop.contains(&name)
                || (family == AzureModelFamily::Gpt5
                    && REASONING_SUMMARY_ALIASES.contains(&name.as_str()))
            {
                continue;
            }
            match OPENAI_PARAMS.contains(&name.as_str()) {
                true => {
                    non_default.insert(name, value);
                }
                false => {
                    extra_body.insert(name, value);
                }
            }
        }
        self.check_supported(family, &mut non_default, &allowed, drop_params)?;

        let model = self.detection_model();
        let mut optional = Map::new();
        match family {
            AzureModelFamily::Gpt => self.map_gpt(&non_default, &mut optional, drop_params)?,
            AzureModelFamily::OSeries => {
                self.map_o_series(&mut non_default, &mut optional, model, drop_params)?
            }
            AzureModelFamily::Gpt5 => {
                self.map_gpt_5(&mut non_default, &mut optional, model, drop_params)?
            }
        }
        // `apply_openai_param_overrides`: what the caller allowed goes as sent.
        for name in &allowed {
            if !optional.contains_key(name)
                && let Some(value) = non_default.remove(name)
            {
                optional.insert(name.clone(), value);
            }
        }

        let (deployment, messages) = match family {
            AzureModelFamily::OSeries => {
                let deployment = self.deployment.replace(O_SERIES_ROUTE, "");
                let messages = self.o_series_messages(&deployment, messages);
                (deployment, messages)
            }
            AzureModelFamily::Gpt5 => (
                self.deployment.replace(GPT_5_SERIES_ROUTE, ""),
                azure_messages(messages),
            ),
            AzureModelFamily::Gpt => (self.deployment.to_string(), azure_messages(messages)),
        };
        let keeps_tool_choice = family == AzureModelFamily::OSeries
            || ["tools", "functions"]
                .iter()
                .any(|name| optional.get(*name).is_some_and(truthy));
        let mut sent = Map::new();
        sent.insert("model".into(), Value::String(deployment.clone()));
        sent.insert("messages".into(), messages);
        sent.extend(
            optional
                .into_iter()
                .filter(|(name, _)| name != "tool_choice" || keeps_tool_choice),
        );
        // The SDK merges `extra_body` over the body it builds.
        sent.extend(extra_body);
        Ok(AzureChatBody {
            deployment,
            body: sent,
        })
    }

    /// `responses_api_bridge_check`: the requests Python sends to the Responses API instead,
    /// which this passthrough does not do.
    fn responses_bridge(
        &self,
        body: &Map<String, Value>,
        extra_body: &Map<String, Value>,
    ) -> Option<&'static str> {
        let model = self.deployment;
        if model.starts_with(RESPONSES_ROUTE)
            || model_info::field(self.info(), &format!("azure/{model}"), "mode")
                .and_then(Value::as_str)
                == Some("responses")
        {
            return Some("Azure deployments served through the Responses API");
        }
        if !is_gpt_reasoning_series_name(model) || is_gpt_5_search_model(model) {
            return None;
        }
        let effort = body
            .get("reasoning_effort")
            .filter(|value| !value.is_null());
        let summary = REASONING_SUMMARY_ALIASES
            .iter()
            .any(|alias| body.contains_key(*alias) || extra_body.contains_key(*alias));
        let reasoning_active = match effort {
            Some(Value::Object(effort)) => {
                effort.get("effort").and_then(Value::as_str) != Some("none")
                    || effort.get("summary").is_some_and(|value| !value.is_null())
            }
            other => effort_level(other) != Some("none"),
        };
        let tools = body
            .get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let tool_type = |tool: &Value| tool.get("type").and_then(Value::as_str).map(str::to_owned);
        let has_function_tool = tools.iter().any(|tool| {
            tool_type(tool).as_deref() == Some("function")
                && (tool.get("function").is_some_and(Value::is_object)
                    || tool.get("name").is_some())
        });
        let has_tool_search = tools
            .iter()
            .any(|tool| tool_type(tool).as_deref() == Some("tool_search"));
        let rejects_function_tools =
            has_function_tool && reasoning_active && is_gpt_5_4_plus_model(model);
        ((effort.is_some() && summary) || rejects_function_tools || has_tool_search).then_some(
            "gpt-5 Azure requests Python bridges to the Responses API (function tools with reasoning on gpt-5.4+, tool_search, or a reasoning summary)",
        )
    }

    /// `_check_valid_arg`: an OpenAI param the family does not take is dropped under
    /// `drop_params` and refused otherwise. It reads the deployment name, as Python asks the
    /// config for its params with the deployment rather than the base model.
    fn check_supported(
        &self,
        family: AzureModelFamily,
        non_default: &mut Map<String, Value>,
        allowed: &[String],
        drop_params: bool,
    ) -> Result<(), Error> {
        let supported = supported_params(family, self.deployment, self.info());
        let unsupported: Vec<String> = non_default
            .iter()
            .filter(|(name, value)| {
                !supported.contains(&name.as_str())
                    && !allowed.iter().any(|allowed| allowed == *name)
                    && !UNVALIDATED_PARAMS.contains(&name.as_str())
                    && !(name.as_str() == "n" && value.as_f64() == Some(1.0))
            })
            .map(|(name, _)| name.clone())
            .collect();
        if unsupported.is_empty() {
            return Ok(());
        }
        if !drop_params {
            return Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams {
                provider: "azure",
                params: unsupported,
                model: self.deployment.to_string(),
            }));
        }
        for name in unsupported {
            non_default.remove(&name);
        }
        Ok(())
    }

    /// `AzureOpenAIConfig.map_openai_params`.
    fn map_gpt(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        drop_params: bool,
    ) -> Result<(), Error> {
        let model = self.detection_model();
        let renames_max_tokens =
            model.contains("gpt-5") || model.contains("gpt-6") || model.contains("gpt5_series");
        let parts: Vec<&str> = self.api_version.split('-').collect();
        let date = (parts.len() >= 3).then(|| (parts[0], parts[1], parts[2]));
        for (name, value) in ordered(non_default) {
            match name {
                "max_tokens" if renames_max_tokens => {
                    optional
                        .entry("max_completion_tokens")
                        .or_insert_with(|| value.clone());
                }
                "tool_choice" => match date {
                    Some((year, month, day))
                        if year < "2023"
                            || (year == "2023" && month < "12")
                            || (year == "2023" && month == "12" && day < "01") =>
                    {
                        if !drop_params {
                            return Err(self.unsupported_value(
                                "tool_choice",
                                value,
                                "tool_choice needs api_version 2023-12-01-preview or later. ",
                            ));
                        }
                    }
                    Some((year, month, _))
                        if value.as_str() == Some("required")
                            && year == "2024"
                            && month <= "05" =>
                    {
                        if !drop_params {
                            return Err(self.unsupported_value(
                                "tool_choice",
                                value,
                                "tool_choice='required' needs an api_version after 2024-05. ",
                            ));
                        }
                    }
                    _ => {
                        optional.insert(name.into(), value.clone());
                    }
                },
                "response_format" if value.is_object() => {
                    let supported = response_format_model(model)
                        && date.is_none_or(|(year, month, _)| {
                            match (year.parse::<u32>(), month.parse::<u32>()) {
                                (Ok(year), Ok(month)) => (year, month) >= RESPONSE_FORMAT_SINCE,
                                _ => true,
                            }
                        });
                    let schema = value
                        .get("response_schema")
                        .or_else(|| {
                            value
                                .get("json_schema")
                                .and_then(|schema| schema.get("schema"))
                        })
                        .filter(|schema| truthy(schema));
                    match (supported, schema) {
                        (true, _) => {
                            optional.insert(name.into(), value.clone());
                        }
                        (false, Some(_)) => {
                            return Err(Error::Unsupported(
                                "a JSON schema response_format on an Azure model or api_version without it, which Python answers through a forced tool call",
                            ));
                        }
                        // Python leaves a schemaless format out when Azure cannot take it.
                        (false, None) => {}
                    }
                }
                "tools" if value.is_array() => {
                    let tools = optional
                        .entry("tools")
                        .or_insert_with(|| Value::Array(Vec::new()));
                    if let (Value::Array(tools), Value::Array(added)) = (tools, value) {
                        tools.extend(added.iter().cloned());
                    }
                }
                _ if AZURE_GPT_PARAMS.contains(&name) => {
                    optional.insert(name.into(), value.clone());
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// `OpenAIOSeriesConfig.map_openai_params` with the Azure o-series params.
    fn map_o_series(
        &self,
        non_default: &mut Map<String, Value>,
        optional: &mut Map<String, Value>,
        model: &str,
        drop_params: bool,
    ) -> Result<(), Error> {
        if let Some(max_tokens) = non_default.remove("max_tokens") {
            optional.insert("max_completion_tokens".into(), max_tokens);
        }
        if let Some(temperature) = non_default.remove("temperature") {
            if temperature.as_f64() == Some(1.0) {
                optional.insert("temperature".into(), temperature);
            } else if !drop_params {
                return Err(self.unsupported_value(
                    "temperature",
                    &temperature,
                    "O-series models only support temperature=1. ",
                ));
            }
        }
        copy_supported(
            non_default,
            optional,
            &supported_params(AzureModelFamily::OSeries, model, self.info()),
        );
        Ok(())
    }

    /// `AzureOpenAIGPT5Config.map_openai_params` over `OpenAIGPT5Config`'s.
    fn map_gpt_5(
        &self,
        non_default: &mut Map<String, Value>,
        optional: &mut Map<String, Value>,
        model: &str,
        drop_params: bool,
    ) -> Result<(), Error> {
        let info = self.info();
        let lookup = gpt_5_lookup_name(model);
        let supports_none = model_info::supports(info, &lookup, "supports_none_reasoning_effort");
        if effort_level(non_default.get("reasoning_effort")) == Some("none") && !supports_none {
            if !drop_params {
                return Err(self.unsupported_value(
                    "reasoning_effort",
                    &Value::from("none"),
                    "Azure OpenAI takes reasoning_effort='none' only on models that support it. ",
                ));
            }
            non_default.remove("reasoning_effort");
        }
        let supported = supported_params(AzureModelFamily::Gpt5, model, info);
        if is_gpt_5_search_model(model) {
            if let Some(max_tokens) = non_default.remove("max_tokens") {
                optional.insert("max_completion_tokens".into(), max_tokens);
            }
            copy_supported(non_default, optional, &supported);
            return Ok(());
        }
        let effective = effort_level(non_default.get("reasoning_effort")).map(str::to_owned);
        if let Some(Value::Object(effort)) = non_default.get("reasoning_effort")
            && let Some(level) = effort.get("effort").cloned()
        {
            non_default.insert("reasoning_effort".into(), level);
        }
        let refused_effort = match effective.as_deref() {
            Some("xhigh") => {
                !model_info::supports(info, &lookup, "supports_xhigh_reasoning_effort")
            }
            Some(level @ ("minimal" | "low")) => model_info::explicitly_disabled(
                info,
                &lookup,
                &format!("supports_{level}_reasoning_effort"),
            ),
            _ => false,
        };
        if refused_effort {
            if !drop_params {
                return Err(self.unsupported_value(
                    "reasoning_effort",
                    &Value::from(effective.clone().unwrap_or_default()),
                    "",
                ));
            }
            non_default.remove("reasoning_effort");
        }
        if let Some(max_tokens) = non_default.remove("max_tokens") {
            optional.insert("max_completion_tokens".into(), max_tokens);
        }
        let resolves_to_none = self.effort_resolves_to_none(&lookup, effective.as_deref());
        if supports_none
            && GPT_5_SAMPLING_PARAMS
                .iter()
                .any(|name| non_default.contains_key(*name))
            && !resolves_to_none
        {
            if !drop_params {
                let name = GPT_5_SAMPLING_PARAMS
                    .iter()
                    .find(|name| non_default.contains_key(**name))
                    .copied()
                    .unwrap_or_default();
                return Err(self.unsupported_value(
                    name,
                    &non_default[name],
                    "logprobs, top_p and top_logprobs need reasoning_effort to resolve to 'none'. ",
                ));
            }
            for name in GPT_5_SAMPLING_PARAMS {
                non_default.remove(*name);
            }
        }
        if let Some(temperature) = non_default.remove("temperature") {
            if (supports_none && resolves_to_none) || temperature.as_f64() == Some(1.0) {
                optional.insert("temperature".into(), temperature);
            } else if !drop_params {
                return Err(self.unsupported_value(
                    "temperature",
                    &temperature,
                    "Only temperature=1 is supported while reasoning is active. ",
                ));
            }
        }
        copy_supported(non_default, optional, &supported);
        if effort_level(optional.get("reasoning_effort")) == Some("none") && !supports_none {
            optional.remove("reasoning_effort");
        }
        Ok(())
    }

    /// `OpenAIGPT5Config.effort_resolves_to_none`.
    fn effort_resolves_to_none(&self, lookup: &str, effective: Option<&str>) -> bool {
        if let Some(effort) = effective {
            return effort == "none";
        }
        let info = self.info();
        if let Some(declared) = model_info::field(info, lookup, "default_reasoning_effort") {
            return declared.as_str() == Some("none");
        }
        if !info.is_some_and(|info| info.declares("default_reasoning_effort")) {
            return model_info::supports(info, lookup, "supports_none_reasoning_effort");
        }
        false
    }

    /// `OpenAIOSeriesConfig._transform_messages`: a system message becomes a user message
    /// unless the cost map says the model takes system messages.
    fn o_series_messages(&self, model: &str, messages: Value) -> Value {
        if model_info::supports(self.info(), model, "supports_system_messages") {
            return messages;
        }
        match messages {
            Value::Array(messages) => Value::Array(
                messages
                    .into_iter()
                    .map(
                        |message| match message.get("role").and_then(Value::as_str) {
                            Some("system") => {
                                let content =
                                    message.get("content").cloned().unwrap_or(Value::Null);
                                serde_json::json!({"content": content, "role": "user"})
                            }
                            _ => message,
                        },
                    )
                    .collect(),
            ),
            other => other,
        }
    }

    fn unsupported_value(&self, param: &str, value: &Value, hint: &str) -> Error {
        Error::InvalidRequest(ErrorDetail::UnsupportedParameter {
            model: format!("azure/{}", self.detection_model()),
            param: param.into(),
            value: match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            },
            hint: format!("{hint}{DROP_HINT}"),
        })
    }
}

/// The OpenAI params a family takes, as its config's `get_supported_openai_params(model)`
/// lists them.
fn supported_params(
    family: AzureModelFamily,
    model: &str,
    info: Option<&dyn ModelInfo>,
) -> Vec<&'static str> {
    match family {
        AzureModelFamily::Gpt => AZURE_GPT_PARAMS.to_vec(),
        AzureModelFamily::OSeries => {
            let mut params = openai_gpt_params(model, info);
            // A model the cost map knows takes `reasoning_effort` only if it reasons; an
            // unknown name, often a custom deployment name, is assumed to.
            if !model_info::known(info, model)
                || model_info::supports(info, model, "supports_reasoning")
            {
                params.push("reasoning_effort");
            }
            params.retain(|name| !O_SERIES_UNSUPPORTED.contains(name));
            params
        }
        AzureModelFamily::Gpt5 => {
            let lookup = gpt_5_lookup_name(model);
            let supports_none =
                model_info::supports(info, &lookup, "supports_none_reasoning_effort");
            let mut params = match is_gpt_5_search_model(model) {
                true => GPT_5_SEARCH_PARAMS.to_vec(),
                false => {
                    let mut params = openai_gpt_params(model, info);
                    params.extend(["reasoning_effort", "verbosity"]);
                    params.retain(|name| {
                        !GPT_5_UNSUPPORTED.contains(name)
                            && (supports_none || !GPT_5_SAMPLING_PARAMS.contains(name))
                    });
                    params
                }
            };
            if !params.contains(&"tool_choice") {
                params.push("tool_choice");
            }
            if is_gpt_5_2_model(model) {
                params.extend(["logprobs", "top_logprobs"]);
            } else if supports_none {
                params.retain(|name| !matches!(*name, "logprobs" | "top_logprobs"));
            }
            params
        }
    }
}

/// `OpenAIGPTConfig.get_supported_openai_params`.
fn openai_gpt_params(model: &str, info: Option<&dyn ModelInfo>) -> Vec<&'static str> {
    let mut params = OPENAI_GPT_PARAMS.to_vec();
    if model != "gpt-3.5-turbo-16k" && model != "gpt-4" {
        params.push("response_format");
    }
    // `is_openai_catalog_model`: `user` only for a model OpenAI's own catalog lists.
    let catalog_name = model
        .split_once(RESPONSES_ROUTE)
        .map_or(model, |(_, rest)| rest);
    if matches!(
        model_info::field(info, catalog_name, "litellm_provider").and_then(Value::as_str),
        Some("openai" | "text-completion-openai")
    ) {
        params.push("user");
    }
    params
}

/// `_is_response_format_supported_model`: everything but gpt-3.5.
fn response_format_model(model: &str) -> bool {
    !(dotted_versions(model).contains("gpt-3.5") || model.contains("gpt-35"))
}

/// `re.sub(r"(\d)-(\d)", r"\1.\2", model)`: `gpt-4-1` reads as `gpt-4.1`.
fn dotted_versions(model: &str) -> String {
    let chars: Vec<char> = model.chars().collect();
    let mut dotted = String::with_capacity(model.len());
    let mut index = 0;
    while index < chars.len() {
        if index + 2 < chars.len()
            && chars[index].is_ascii_digit()
            && chars[index + 1] == '-'
            && chars[index + 2].is_ascii_digit()
        {
            dotted.extend([chars[index], '.', chars[index + 2]]);
            index += 3;
        } else {
            dotted.push(chars[index]);
            index += 1;
        }
    }
    dotted
}

/// `convert_to_azure_openai_messages`: Azure wants a function call's `arguments` as a string,
/// an `image_url` as an object without `format`, and a `file` without `format`.
fn azure_messages(messages: Value) -> Value {
    let Value::Array(mut messages) = messages else {
        return messages;
    };
    for message in &mut messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        match message.get("role").and_then(Value::as_str) {
            Some("assistant") => {
                if let Some(Value::Object(call)) = message.get_mut("function_call")
                    && call.get("arguments").is_none_or(Value::is_null)
                {
                    call.insert("arguments".into(), Value::from(""));
                }
            }
            Some("user") => {
                let Some(Value::Array(parts)) = message.get_mut("content") else {
                    continue;
                };
                for part in parts.iter_mut().filter_map(Value::as_object_mut) {
                    match part.get("type").and_then(Value::as_str) {
                        Some("image_url") => match part.get_mut("image_url") {
                            Some(Value::String(url)) => {
                                let url = std::mem::take(url);
                                part.insert("image_url".into(), serde_json::json!({"url": url}));
                            }
                            Some(Value::Object(image)) => {
                                image.remove("format");
                            }
                            _ => {}
                        },
                        Some("file") => {
                            if let Some(Value::Object(file)) = part.get_mut("file") {
                                file.remove("format");
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    Value::Array(messages)
}

/// `_map_openai_params`: every supported param still unmapped goes as sent.
fn copy_supported(
    non_default: &mut Map<String, Value>,
    optional: &mut Map<String, Value>,
    supported: &[&str],
) {
    for name in OPENAI_PARAMS {
        if supported.contains(name)
            && let Some(value) = non_default.get(*name)
        {
            optional.insert((*name).into(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rstest::rstest;
    use serde_json::json;

    use super::*;

    /// A cost map of a few entries.
    struct Entries(HashMap<String, Map<String, Value>>);

    impl Entries {
        fn new(entries: Value) -> Self {
            let Value::Object(entries) = entries else {
                panic!("entries are an object");
            };
            Self(
                entries
                    .into_iter()
                    .map(|(name, entry)| (name, object(entry)))
                    .collect(),
            )
        }
    }

    impl ModelInfo for Entries {
        fn entry(&self, model: &str) -> Option<&Map<String, Value>> {
            self.0.get(model)
        }

        fn declares(&self, key: &str) -> bool {
            self.0.values().any(|entry| entry.contains_key(key))
        }
    }

    fn cost_map() -> Entries {
        Entries::new(json!({
            "azure/gpt-5": {"supports_reasoning": true},
            "azure/gpt-5.1": {"supports_none_reasoning_effort": true, "default_reasoning_effort": "none"},
            "azure/gpt-5.2": {"supports_reasoning": true},
            "azure/gpt-5.5": {"supports_none_reasoning_effort": true, "supports_xhigh_reasoning_effort": true, "supports_minimal_reasoning_effort": false},
            "o3-mini": {"litellm_provider": "openai", "supports_reasoning": true},
            "o1-preview": {"litellm_provider": "openai", "supports_system_messages": true},
            "gpt-4o-mini": {"litellm_provider": "openai"},
        }))
    }

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("expected an object, got {value}");
        };
        map
    }

    fn request<'a>(
        deployment: &'a str,
        base_model: Option<&'a str>,
        api_version: &'a str,
        info: Option<&'a dyn ModelInfo>,
    ) -> AzureChatRequest<'a> {
        AzureChatRequest {
            deployment,
            base_model,
            api_version,
            drop_params: false,
            model_info: info,
        }
    }

    fn transform(request: &AzureChatRequest<'_>, body: Value) -> Result<Map<String, Value>, Error> {
        request.transform(object(body)).map(|sent| sent.body)
    }

    fn user(text: &str) -> Value {
        json!([{"role": "user", "content": text}])
    }

    fn is_unsupported_param(result: &Result<Map<String, Value>, Error>) -> bool {
        matches!(
            result,
            Err(Error::InvalidRequest(
                ErrorDetail::UnsupportedParams { .. } | ErrorDetail::UnsupportedParameter { .. }
            ))
        )
    }

    #[rstest]
    #[case::gpt_4o("gpt-4o", None, AzureModelFamily::Gpt)]
    #[case::gpt_5_chat("gpt-5-chat", None, AzureModelFamily::Gpt)]
    #[case::o3_mini("o3-mini", None, AzureModelFamily::OSeries)]
    #[case::o4("my-o4-mini", None, AzureModelFamily::OSeries)]
    #[case::o_series_route("o_series/reasoner", None, AzureModelFamily::OSeries)]
    #[case::gpt_5("gpt-5.1", None, AzureModelFamily::Gpt5)]
    #[case::gpt_6("gpt-6", None, AzureModelFamily::Gpt5)]
    #[case::gpt_5_route("gpt5_series/prod", None, AzureModelFamily::Gpt5)]
    #[case::base_model_names_the_model("prod-chat", Some("azure/gpt-5.1"), AzureModelFamily::Gpt5)]
    #[case::base_model_o_series("prod-chat", Some("azure/o3"), AzureModelFamily::OSeries)]
    fn the_family_follows_the_base_model_then_the_deployment(
        #[case] deployment: &str,
        #[case] base_model: Option<&str>,
        #[case] family: AzureModelFamily,
    ) {
        assert_eq!(
            request(deployment, base_model, "2024-10-21", None).family(),
            family
        );
    }

    #[rstest]
    fn a_gpt_body_passes_through_with_unknown_params_and_without_litellm_keys() {
        let sent = transform(
            &request("gpt-4o", None, "2024-10-21", None),
            json!({
                "model": "public",
                "messages": user("hi"),
                "temperature": 0.2,
                "stream": true,
                "stream_options": {"include_usage": true},
                "top_k": 5,
                "seed": null,
                "metadata": {"k": "v"},
                "extra_body": {"data_sources": [{"type": "x"}], "azure_ad_token": "t"},
            }),
        )
        .unwrap();
        assert_eq!(
            sent,
            object(json!({
                "model": "gpt-4o",
                "messages": user("hi"),
                "temperature": 0.2,
                "stream": true,
                "stream_options": {"include_usage": true},
                "top_k": 5,
                "data_sources": [{"type": "x"}],
            }))
        );
    }

    #[rstest]
    #[case::refused(false, json!({}), None)]
    #[case::deployment_drops(true, json!({}), Some(json!({"model": "gpt-4o", "messages": user("hi")})))]
    #[case::request_drops(false, json!({"drop_params": true}), Some(json!({"model": "gpt-4o", "messages": user("hi")})))]
    #[case::allowed(false, json!({"allowed_openai_params": ["reasoning_effort"]}), Some(json!({"model": "gpt-4o", "messages": user("hi"), "reasoning_effort": "low"})))]
    fn an_openai_param_the_family_lacks_is_refused_unless_dropped_or_allowed(
        #[case] drop_params: bool,
        #[case] extra: Value,
        #[case] expected: Option<Value>,
    ) {
        let mut body =
            object(json!({"model": "m", "messages": user("hi"), "reasoning_effort": "low"}));
        body.extend(object(extra));
        let request = AzureChatRequest {
            drop_params,
            ..request("gpt-4o", None, "2024-10-21", None)
        };
        let result = request.transform(body).map(|sent| sent.body);
        match expected {
            Some(expected) => assert_eq!(result.unwrap(), object(expected)),
            None => assert!(
                matches!(
                    &result,
                    Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams { params, .. }))
                        if params == &["reasoning_effort"]
                ),
                "{result:?}"
            ),
        }
    }

    #[rstest]
    #[case::gpt_4o("gpt-4o", json!({"max_tokens": 10}))]
    #[case::gpt_5_chat("gpt-5-chat", json!({"max_completion_tokens": 10}))]
    fn max_tokens_is_renamed_for_the_gpt_5_names(
        #[case] deployment: &str,
        #[case] expected: Value,
    ) {
        let sent = transform(
            &request(deployment, None, "2024-10-21", None),
            json!({"messages": user("hi"), "max_tokens": 10}),
        )
        .unwrap();
        let mut body = object(json!({"model": deployment, "messages": user("hi")}));
        body.extend(object(expected));
        assert_eq!(sent, body);
    }

    #[rstest]
    #[case::before_tools("2023-07-01-preview", "auto", None)]
    #[case::required_too_early("2024-05-01-preview", "required", None)]
    #[case::auto_on_may("2024-05-01-preview", "auto", Some("auto"))]
    #[case::required_later("2024-10-21", "required", Some("required"))]
    #[case::v1("v1", "required", Some("required"))]
    fn tool_choice_follows_the_api_version(
        #[case] api_version: &str,
        #[case] tool_choice: &str,
        #[case] expected: Option<&str>,
    ) {
        let tools = json!([{"type": "function", "function": {"name": "f"}}]);
        let result = transform(
            &request("gpt-4o", None, api_version, None),
            json!({"messages": user("hi"), "tools": tools, "tool_choice": tool_choice}),
        );
        match expected {
            Some(expected) => assert_eq!(result.unwrap()["tool_choice"], json!(expected)),
            None => assert!(is_unsupported_param(&result), "{result:?}"),
        }
    }

    #[rstest]
    fn tool_choice_without_tools_is_left_out() {
        let sent = transform(
            &request("gpt-4o", None, "2024-10-21", None),
            json!({"messages": user("hi"), "tool_choice": "auto"}),
        )
        .unwrap();
        assert!(!sent.contains_key("tool_choice"));
    }

    #[rstest]
    #[case::supported("gpt-4o", "2024-10-21", json!({"type": "json_object"}), Some(true))]
    #[case::old_version_drops_json_object("gpt-4o", "2024-06-01", json!({"type": "json_object"}), Some(false))]
    #[case::gpt_35_drops_json_object("gpt-35-turbo", "2024-10-21", json!({"type": "json_object"}), Some(false))]
    #[case::gpt_35_schema_is_refused(
        "gpt-35-turbo",
        "2024-10-21",
        json!({"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}}}),
        None
    )]
    fn response_format_follows_the_model_and_api_version(
        #[case] deployment: &str,
        #[case] api_version: &str,
        #[case] format: Value,
        #[case] kept: Option<bool>,
    ) {
        let result = transform(
            &request(deployment, None, api_version, None),
            json!({"messages": user("hi"), "response_format": format}),
        );
        match kept {
            Some(kept) => assert_eq!(result.unwrap().contains_key("response_format"), kept),
            None => assert!(matches!(result, Err(Error::Unsupported(_))), "{result:?}"),
        }
    }

    #[rstest]
    fn gpt_messages_are_shaped_for_azure() {
        let sent = transform(
            &request("gpt-4o", None, "2024-10-21", None),
            json!({"messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": "https://x/i.png"},
                    {"type": "image_url", "image_url": {"url": "https://x/j.png", "format": "image/png"}},
                    {"type": "file", "file": {"file_id": "f", "format": "application/pdf"}},
                ]},
                {"role": "assistant", "content": null, "function_call": {"name": "f", "arguments": null}},
            ]}),
        )
        .unwrap();
        assert_eq!(
            sent["messages"],
            json!([
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/i.png"}},
                    {"type": "image_url", "image_url": {"url": "https://x/j.png"}},
                    {"type": "file", "file": {"file_id": "f"}},
                ]},
                {"role": "assistant", "content": null, "function_call": {"name": "f", "arguments": ""}},
            ])
        );
    }

    #[rstest]
    fn o_series_maps_max_tokens_and_keeps_reasoning_effort_for_custom_names() {
        let sent = transform(
            &request("o_series/reasoner", None, "2024-12-01-preview", None),
            json!({"messages": [{"role": "system", "content": "be brief", "name": "s"}, {"role": "user", "content": "hi"}],
                "max_tokens": 100, "reasoning_effort": "high", "temperature": 1}),
        )
        .unwrap();
        assert_eq!(
            sent,
            object(json!({
                "model": "reasoner",
                "messages": [{"content": "be brief", "role": "user"}, {"role": "user", "content": "hi"}],
                "max_completion_tokens": 100,
                "reasoning_effort": "high",
                "temperature": 1,
            }))
        );
    }

    #[rstest]
    #[case::temperature(json!({"temperature": 0.5}))]
    #[case::top_p(json!({"top_p": 0.5}))]
    #[case::presence_penalty(json!({"presence_penalty": 1}))]
    fn o_series_refuses_sampling_params(#[case] params: Value) {
        let mut body = object(json!({"messages": user("hi")}));
        body.extend(object(params));
        let result = request("o3-mini", None, "2024-12-01-preview", None)
            .transform(body)
            .map(|sent| sent.body);
        assert!(is_unsupported_param(&result), "{result:?}");
    }

    #[rstest]
    #[case::unknown_model_has_no_system_messages("o3-mini", false)]
    #[case::the_cost_map_says_so("o1-preview", true)]
    fn o_series_system_messages_follow_the_cost_map(#[case] deployment: &str, #[case] kept: bool) {
        let map = cost_map();
        let sent = transform(
            &request(deployment, None, "2024-12-01-preview", Some(&map)),
            json!({"messages": [{"role": "system", "content": "s"}]}),
        )
        .unwrap();
        assert_eq!(
            sent["messages"][0]["role"],
            json!(if kept { "system" } else { "user" })
        );
    }

    #[rstest]
    fn o_series_user_needs_an_openai_catalog_model() {
        let map = cost_map();
        let known = transform(
            &request("o3-mini", None, "2024-12-01-preview", Some(&map)),
            json!({"messages": user("hi"), "user": "u"}),
        )
        .unwrap();
        assert_eq!(known["user"], json!("u"));
        let custom = transform(
            &request("reasoner-o3", None, "2024-12-01-preview", Some(&map)),
            json!({"messages": user("hi"), "user": "u"}),
        )
        .unwrap();
        assert!(!custom.contains_key("user"));
    }

    #[rstest]
    fn gpt_5_renames_max_tokens_and_strips_its_route() {
        let sent = transform(
            &request("gpt5_series/prod", None, "2025-04-01-preview", None),
            json!({"messages": user("hi"), "max_tokens": 64, "reasoning_effort": "low", "verbosity": "low"}),
        )
        .unwrap();
        assert_eq!(
            sent,
            object(json!({
                "model": "prod",
                "messages": user("hi"),
                "max_completion_tokens": 64,
                "reasoning_effort": "low",
                "verbosity": "low",
            }))
        );
    }

    #[rstest]
    #[case::gpt_5_refuses_a_low_temperature("gpt-5", None, json!({"temperature": 0}), None)]
    #[case::gpt_5_takes_temperature_one("gpt-5", None, json!({"temperature": 1}), Some(json!({"temperature": 1})))]
    #[case::gpt_5_1_defaults_to_no_reasoning("gpt-5.1", None, json!({"temperature": 0}), Some(json!({"temperature": 0})))]
    #[case::gpt_5_1_with_reasoning_refuses_it("gpt-5.1", None, json!({"temperature": 0, "reasoning_effort": "high"}), None)]
    #[case::base_model_decides("prod", Some("azure/gpt-5.1"), json!({"temperature": 0}), Some(json!({"temperature": 0})))]
    #[case::gpt_5_refuses_effort_none("gpt-5", None, json!({"reasoning_effort": "none"}), None)]
    #[case::gpt_5_1_takes_effort_none("gpt-5.1", None, json!({"reasoning_effort": "none"}), Some(json!({"reasoning_effort": "none"})))]
    #[case::effort_object_is_flattened("gpt-5.1", None, json!({"reasoning_effort": {"effort": "high"}}), Some(json!({"reasoning_effort": "high"})))]
    #[case::xhigh_needs_support("gpt-5", None, json!({"reasoning_effort": "xhigh"}), None)]
    #[case::xhigh_supported("gpt-5.5", None, json!({"reasoning_effort": "xhigh"}), Some(json!({"reasoning_effort": "xhigh"})))]
    #[case::minimal_disabled("gpt-5.5", None, json!({"reasoning_effort": "minimal"}), None)]
    #[case::gpt_5_refuses_top_p("gpt-5", None, json!({"top_p": 0.5}), None)]
    #[case::gpt_5_refuses_stop("gpt-5", None, json!({"stop": ["x"]}), None)]
    #[case::gpt_5_2_takes_logprobs("gpt-5.2", None, json!({"logprobs": true}), Some(json!({"logprobs": true})))]
    fn gpt_5_params_follow_the_cost_map(
        #[case] deployment: &str,
        #[case] base_model: Option<&str>,
        #[case] params: Value,
        #[case] expected: Option<Value>,
    ) {
        let map = cost_map();
        let mut body = object(json!({"messages": user("hi")}));
        body.extend(object(params));
        let result = request(deployment, base_model, "2025-04-01-preview", Some(&map))
            .transform(body)
            .map(|sent| sent.body);
        match expected {
            Some(expected) => {
                let mut body = object(json!({"model": deployment, "messages": user("hi")}));
                body.extend(object(expected));
                assert_eq!(result.unwrap(), body);
            }
            None => assert!(is_unsupported_param(&result), "{result:?}"),
        }
    }

    #[rstest]
    fn drop_params_drops_what_gpt_5_refuses() {
        let map = cost_map();
        let request = AzureChatRequest {
            drop_params: true,
            ..request("gpt-5", None, "2025-04-01-preview", Some(&map))
        };
        let sent = request
            .transform(object(json!({"messages": user("hi"), "temperature": 0, "top_p": 0.5, "reasoning_effort": "none"})))
            .unwrap()
            .body;
        assert_eq!(
            sent,
            object(json!({"model": "gpt-5", "messages": user("hi")}))
        );
    }

    #[rstest]
    #[case::responses_route("responses/prod", json!({}))]
    #[case::function_tools_with_reasoning("gpt-5.4", json!({"tools": [{"type": "function", "function": {"name": "f"}}]}))]
    #[case::reasoning_summary("gpt-5.1", json!({"reasoning_effort": "high", "reasoning_summary": "auto"}))]
    fn requests_python_bridges_to_responses_are_refused(
        #[case] deployment: &str,
        #[case] params: Value,
    ) {
        let mut body = object(json!({"messages": user("hi")}));
        body.extend(object(params));
        assert!(matches!(
            request(deployment, None, "2025-04-01-preview", None).transform(body),
            Err(Error::Unsupported(_))
        ));
    }

    #[rstest]
    fn function_tools_without_reasoning_stay_on_chat() {
        let sent = transform(
            &request("gpt-5.4", None, "2025-04-01-preview", None),
            json!({"messages": user("hi"), "reasoning_effort": "none",
                "tools": [{"type": "function", "function": {"name": "f"}}]}),
        );
        assert!(!matches!(sent, Err(Error::Unsupported(_))), "{sent:?}");
    }

    #[rstest]
    #[case::bare("gpt-5", Some((5, 0)))]
    #[case::minor("gpt-5.4", Some((5, 4)))]
    #[case::suffix("gpt-5.4-pro", Some((5, 4)))]
    #[case::routed("azure/gpt-6", Some((6, 0)))]
    #[case::dash_suffix("gpt-5-mini", Some((5, 0)))]
    #[case::glued("gpt-5x", None)]
    #[case::other("o3", None)]
    fn the_gpt_version_is_read_from_the_name(
        #[case] model: &str,
        #[case] expected: Option<(u32, u32)>,
    ) {
        assert_eq!(gpt_series_version(model), expected);
    }

    #[rstest]
    #[case::dashed("gpt-3-5-turbo", "gpt-3.5-turbo")]
    #[case::non_overlapping("gpt-4-1-2", "gpt-4.1-2")]
    fn dashed_versions_read_as_dotted(#[case] model: &str, #[case] expected: &str) {
        assert_eq!(dotted_versions(model), expected);
    }
}
