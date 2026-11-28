//! The daily entity, model and guardrail usage projections of an accounting delivery.

use std::collections::{BTreeMap, BTreeSet};

use litellm_db::Error;
use litellm_python_compat::{Value as PythonValue, json};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{Payload, parse_timestamp};

const MAX_COUNTER: i64 = 1 << 40;

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Fields {
    #[serde(rename = "user", with = "serde_with::rust::double_option")]
    user_id: Option<Option<String>>,
    request_id: Option<String>,
    call_type: Option<String>,
    #[serde(with = "serde_with::rust::double_option")]
    api_key: Option<Option<String>>,
    #[serde(with = "serde_with::rust::double_option")]
    model: Option<Option<String>>,
    #[serde(with = "serde_with::rust::double_option")]
    model_group: Option<Option<String>>,
    #[serde(with = "serde_with::rust::double_option")]
    custom_llm_provider: Option<Option<String>>,
    #[serde(with = "serde_with::rust::double_option")]
    mcp_namespaced_tool_name: Option<Option<String>>,
    team_id: Option<String>,
    end_user: Option<String>,
    agent_id: Option<String>,
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
    request_duration_ms: Option<i64>,
    status: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Metadata {
    status: Option<String>,
    internal_call_origin: Option<String>,
    billed_retry_attempt: bool,
    route_call_type: Option<String>,
    usage_object: Option<Usage>,
    accounting_savings: Option<Savings>,
    accounting_guardrail_spend: Option<f64>,
    accounting_customer_margin_spend: Option<f64>,
    cache_cost_breakdown: Option<CacheCosts>,
    compression_savings: Option<Compression>,
    guardrail_information: Option<Guardrails>,
    standard_logging_guardrail_information: Option<Guardrails>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Usage {
    cache_read_input_tokens: Option<f64>,
    cache_creation_input_tokens: Option<f64>,
    prompt_tokens_details: Option<PromptTokens>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct PromptTokens {
    cached_tokens: Option<f64>,
    cache_write_tokens: Option<f64>,
    cache_creation_tokens: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Savings {
    compression: Option<f64>,
    prompt_caching: Option<f64>,
    gateway_injected_caching: Option<f64>,
    autorouter: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct CacheCosts {
    cache_read_cost: Option<f64>,
    cache_creation_cost: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Compression {
    tokens_saved: Option<f64>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Guardrails {
    List(Vec<Guardrail>),
    One(Guardrail),
}

impl Guardrails {
    fn entries(&self) -> &[Guardrail] {
        match self {
            Self::List(items) => items,
            Self::One(item) => std::slice::from_ref(item),
        }
    }
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Guardrail {
    guardrail_id: Option<String>,
    guardrail_name: Option<String>,
    guardrail_provider: Option<String>,
    guardrail_response: Option<Compression>,
    guardrail_usage: Option<BTreeMap<String, serde_json::Value>>,
    guardrail_cost_in_spend: Option<bool>,
    guardrail_cost_by_unit: Option<BTreeMap<String, Option<f64>>>,
}

macro_rules! measures {
    ($name:ident { $($field:ident : $kind:ty),* $(,)? }) => {
        #[derive(Default, Serialize)]
        struct $name { $($field: $kind),* }

        impl std::ops::AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                $(self.$field += other.$field;)*
            }
        }
    };
}

measures!(DailyCounts {
    spend: f64,
    prompt_tokens: i64,
    completion_tokens: i64,
    api_requests: i64,
    successful_requests: i64,
    failed_requests: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
    compression_saved_tokens: i64,
    compression_savings_spend: f64,
    prompt_caching_savings_spend: f64,
    gateway_injected_caching_savings_spend: f64,
    autorouter_savings_spend: f64,
    guardrail_spend: f64,
    customer_margin_spend: f64,
    total_response_time_ms: i64,
    timed_requests: i64,
});

measures!(ModelCounts {
    spend: f64,
    prompt_tokens: i64,
    completion_tokens: i64,
    request_count: i64,
    successful_requests: i64,
    failed_requests: i64,
    total_response_time_ms: i64,
    timed_requests: i64,
});

measures!(GuardrailCounts {
    units: i64,
    cost: f64,
    untracked_units: i64
});

fn cap(value: i64) -> i64 {
    value.clamp(-MAX_COUNTER, MAX_COUNTER)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Python truncates the logged token count toward zero after bounding it"
)]
fn tokens(value: Option<f64>) -> i64 {
    value
        .unwrap_or_default()
        .clamp(-(MAX_COUNTER as f64), MAX_COUNTER as f64) as i64
}

impl Payload {
    fn daily_counts(&self) -> DailyCounts {
        let metadata = &self.metadata.rollup;
        let savings = metadata.accounting_savings.as_ref();
        let usage = metadata.usage_object.as_ref();
        let details = usage.and_then(|usage| usage.prompt_tokens_details.as_ref());
        let internal = self.origin().is_some();
        let caller = !internal && !metadata.billed_retry_attempt;
        let success = metadata.status.as_deref() != Some("failure");
        let duration = self
            .rollup
            .request_duration_ms
            .filter(|value| !internal && success && *value >= 0);
        let headroom = metadata
            .guardrail_information
            .as_ref()
            .map(Guardrails::entries)
            .unwrap_or_default()
            .iter()
            .filter(|item| item.guardrail_provider.as_deref() == Some("headroom"))
            .map(|item| {
                item.guardrail_response
                    .as_ref()
                    .and_then(|stats| stats.tokens_saved)
                    .unwrap_or_default()
                    .max(0.0)
            })
            .sum::<f64>();
        DailyCounts {
            spend: self.spend.unwrap_or_default(),
            prompt_tokens: cap(self.rollup.prompt_tokens.unwrap_or_default()),
            completion_tokens: cap(self.rollup.completion_tokens.unwrap_or_default()),
            api_requests: i64::from(caller),
            successful_requests: i64::from(caller && success),
            failed_requests: i64::from(caller && !success),
            cache_read_input_tokens: tokens(
                usage
                    .and_then(|value| value.cache_read_input_tokens)
                    .filter(|value| *value != 0.0)
                    .or_else(|| details.and_then(|value| value.cached_tokens)),
            ),
            cache_creation_input_tokens: tokens(
                usage
                    .and_then(|value| value.cache_creation_input_tokens)
                    .filter(|value| *value != 0.0)
                    .or_else(|| {
                        details.and_then(|value| {
                            value
                                .cache_write_tokens
                                .filter(|value| *value != 0.0)
                                .or(value.cache_creation_tokens)
                        })
                    }),
            ),
            compression_saved_tokens: tokens(Some(
                metadata
                    .compression_savings
                    .as_ref()
                    .and_then(|stats| stats.tokens_saved)
                    .unwrap_or_default()
                    .max(0.0)
                    + headroom,
            )),
            compression_savings_spend: savings
                .and_then(|value| value.compression)
                .unwrap_or_default(),
            prompt_caching_savings_spend: savings
                .and_then(|value| value.prompt_caching)
                .unwrap_or_default(),
            gateway_injected_caching_savings_spend: savings
                .and_then(|value| value.gateway_injected_caching)
                .unwrap_or_default(),
            autorouter_savings_spend: savings
                .and_then(|value| value.autorouter)
                .filter(|_| !internal)
                .unwrap_or_default(),
            guardrail_spend: metadata.accounting_guardrail_spend.unwrap_or_default(),
            customer_margin_spend: metadata
                .accounting_customer_margin_spend
                .unwrap_or_default(),
            total_response_time_ms: cap(duration.unwrap_or_default()),
            timed_requests: i64::from(duration.is_some()),
        }
    }

    fn cache_charge(&self, counts: &DailyCounts) -> CacheCharge {
        let costs = self.metadata.rollup.cache_cost_breakdown.as_ref();
        let read = costs
            .and_then(|value| value.cache_read_cost)
            .filter(|cost| *cost >= 0.0);
        let write = costs
            .and_then(|value| value.cache_creation_cost)
            .filter(|cost| *cost >= 0.0);
        CacheCharge {
            cache_cost: read.unwrap_or_default() + write.unwrap_or_default(),
            priced_cache_tokens: read.map_or(0, |_| counts.cache_read_input_tokens)
                + write.map_or(0, |_| counts.cache_creation_input_tokens),
        }
    }

    fn origin(&self) -> Option<&str> {
        self.metadata
            .rollup
            .internal_call_origin
            .as_deref()
            .filter(|s| !s.is_empty())
    }
}

/// Destination order matches Python's table ordering during a delivery.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::spend_log) enum DailyTable {
    Agent,
    CustomerMember,
    EndUser,
    Organization,
    Tag,
    Team,
    User,
}

#[derive(Clone, Default, Ord, PartialOrd, Eq, PartialEq, Serialize)]
struct DailyKey<'a> {
    date: String,
    api_key: &'a str,
    model: &'a str,
    custom_llm_provider: &'a str,
    mcp_namespaced_tool_name: &'a str,
    endpoint: &'a str,
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Serialize)]
struct CustomerKey<'a> {
    customer_org_id: &'a str,
    customer_team_id: &'a str,
    app_id: &'a str,
    app_group_id: &'a str,
    model_group: &'a str,
}

measures!(CacheCharge {
    cache_cost: f64,
    priced_cache_tokens: i64
});

#[derive(Serialize)]
pub(in crate::spend_log) struct DailyRow<'a> {
    id: String,
    agent_id: &'a str,
    customer_member_id: &'a str,
    end_user_id: &'a str,
    organization_id: &'a str,
    tag: &'a str,
    team_id: &'a str,
    user_id: &'a str,
    #[serde(flatten)]
    key: DailyKey<'a>,
    #[serde(flatten)]
    customer: Option<CustomerKey<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_group: Option<&'a str>,
    request_id: Option<&'a str>,
    #[serde(flatten)]
    counts: DailyCounts,
    #[serde(flatten)]
    cache: CacheCharge,
}

type DailyIdentity<'a> = (&'a str, DailyKey<'a>, Option<CustomerKey<'a>>);

#[derive(Ord, PartialOrd, Eq, PartialEq, Serialize)]
struct ModelKey<'a> {
    date: &'a str,
    model_group: &'a str,
    model: &'a str,
    custom_llm_provider: &'a str,
    task_type: &'a str,
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Serialize)]
struct GuardrailKey<'a> {
    guardrail_id: &'a str,
    date: String,
    team_id: &'a str,
    api_key: &'a str,
    usage_unit: &'a str,
}

#[derive(Serialize)]
pub(in crate::spend_log) struct Row<'a, Key, Counts> {
    #[serde(flatten)]
    key: &'a Key,
    #[serde(flatten)]
    counts: &'a Counts,
}

/// The daily projections grouped by their conflict key before applying them.
#[derive(Default)]
pub(in crate::spend_log) struct Rollups<'a> {
    daily: BTreeMap<DailyTable, BTreeMap<DailyIdentity<'a>, DailyRow<'a>>>,
    models: BTreeMap<ModelKey<'a>, ModelCounts>,
    guardrails: BTreeMap<GuardrailKey<'a>, GuardrailCounts>,
}

impl<'a> Rollups<'a> {
    pub(in crate::spend_log) fn project(payloads: &'a [Payload]) -> Result<Self, Error> {
        let mut result = Self::default();
        for payload in payloads {
            if let Some(start) = payload.start_time.as_deref() {
                let date = start.get(..10);
                let utc_date = parse_timestamp(start).map(|time| time.date().to_string());
                if let Some(date) = date {
                    result.daily_rows(payload, date, utc_date.as_deref())?;
                    result.model_row(payload, date);
                }
                if let Some(date) = utc_date {
                    result.guardrail_rows(payload, &date);
                }
            }
        }
        Ok(result)
    }

    pub(in crate::spend_log) fn daily(
        &self,
    ) -> impl Iterator<Item = (DailyTable, impl Iterator<Item = &DailyRow<'a>>)> {
        self.daily
            .iter()
            .map(|(&table, rows)| (table, rows.values()))
    }

    pub(in crate::spend_log) fn models(&self) -> Option<impl Serialize + '_> {
        (!self.models.is_empty()).then(|| {
            self.models
                .iter()
                .map(|(key, counts)| Row { key, counts })
                .collect::<Vec<_>>()
        })
    }

    pub(in crate::spend_log) fn guardrails(&self) -> Option<impl Serialize + '_> {
        (!self.guardrails.is_empty()).then(|| {
            self.guardrails
                .iter()
                .map(|(key, counts)| Row { key, counts })
                .collect::<Vec<_>>()
        })
    }

    fn daily_rows(
        &mut self,
        payload: &'a Payload,
        date: &str,
        utc_date: Option<&str>,
    ) -> Result<(), Error> {
        let fields = &payload.rollup;
        let Some(api_key) = &fields.api_key else {
            return Ok(());
        };
        if fields.mcp_namespaced_tool_name.is_none()
            && (fields.model.is_none()
                || fields.model_group.is_none()
                || fields.custom_llm_provider.is_none())
        {
            return Ok(());
        }
        let key = DailyKey {
            date: date.to_owned(),
            api_key: api_key.as_deref().unwrap_or_default(),
            model: optional(&fields.model).unwrap_or_default(),
            custom_llm_provider: optional(&fields.custom_llm_provider).unwrap_or_default(),
            mcp_namespaced_tool_name: optional(&fields.mcp_namespaced_tool_name)
                .unwrap_or_default(),
            endpoint: endpoint(
                payload
                    .metadata
                    .rollup
                    .route_call_type
                    .as_deref()
                    .or(fields.call_type.as_deref())
                    .unwrap_or_default(),
            ),
        };
        let group = optional(&fields.model_group);
        let entities = &payload.metadata.accounting_entities;
        for (table, id) in [
            (DailyTable::Agent, fields.agent_id.as_deref()),
            (
                DailyTable::EndUser,
                fields.end_user.as_deref().filter(|value| !value.is_empty()),
            ),
            (
                DailyTable::Organization,
                entities.organization_id.as_deref(),
            ),
            (DailyTable::Team, fields.team_id.as_deref()),
            (
                DailyTable::User,
                fields
                    .user_id
                    .as_ref()
                    .map(|value| value.as_deref().unwrap_or_default()),
            ),
        ] {
            if let Some(id) = id {
                self.daily_row(payload, table, id, key.clone(), group, None)?;
            }
        }
        for tag in payload.request_tags.iter().collect::<BTreeSet<_>>() {
            self.daily_row(payload, DailyTable::Tag, tag, key.clone(), group, None)?;
        }
        let customer = payload.customer();
        if let (Some(date), Some(org)) = (
            utc_date,
            customer
                .customer_org_id
                .as_deref()
                .filter(|id| !id.is_empty()),
        ) {
            let key = DailyKey {
                date: date.to_owned(),
                ..key
            };
            self.daily_row(
                payload,
                DailyTable::CustomerMember,
                customer.customer_member_id.as_deref().unwrap_or_default(),
                key,
                None,
                Some(CustomerKey {
                    customer_org_id: org,
                    customer_team_id: customer.customer_team_id.as_deref().unwrap_or_default(),
                    app_id: entities.app_id.as_deref().unwrap_or_default(),
                    app_group_id: entities.app_group_id.as_deref().unwrap_or_default(),
                    model_group: group.unwrap_or_default(),
                }),
            )?;
        }
        Ok(())
    }

    fn daily_row(
        &mut self,
        payload: &'a Payload,
        table: DailyTable,
        id: &'a str,
        key: DailyKey<'a>,
        model_group: Option<&'a str>,
        customer: Option<CustomerKey<'a>>,
    ) -> Result<(), Error> {
        if [
            id,
            &key.date,
            key.api_key,
            key.model,
            key.custom_llm_provider,
            key.mcp_namespaced_tool_name,
            key.endpoint,
        ]
        .into_iter()
        .any(|value| value.len() > 2000)
            || customer.is_some_and(|key| {
                [
                    key.customer_org_id,
                    key.customer_team_id,
                    key.app_id,
                    key.app_group_id,
                    key.model_group,
                ]
                .into_iter()
                .any(|value| value.len() > 2000)
            })
        {
            return Ok(());
        }
        let counts = payload.daily_counts();
        let cache = customer
            .map(|_| payload.cache_charge(&counts))
            .unwrap_or_default();
        let rows = self.daily.entry(table).or_default();
        let identity = (id, key.clone(), customer);
        let row = match rows.entry(identity) {
            std::collections::btree_map::Entry::Occupied(row) => row.into_mut(),
            std::collections::btree_map::Entry::Vacant(row) => {
                let row_id = match customer {
                    Some(customer) => customer_row_id(id, &key, customer)?,
                    None => Uuid::new_v4().to_string(),
                };
                row.insert(DailyRow {
                    id: row_id,
                    agent_id: id,
                    customer_member_id: id,
                    end_user_id: id,
                    organization_id: id,
                    tag: id,
                    team_id: id,
                    user_id: id,
                    key,
                    customer,
                    model_group,
                    request_id: Some(payload.rollup.request_id.as_deref().unwrap_or_default()),
                    counts: DailyCounts::default(),
                    cache: CacheCharge::default(),
                })
            }
        };
        row.counts += counts;
        row.cache += cache;
        Ok(())
    }

    fn model_row(&mut self, payload: &'a Payload, date: &'a str) {
        let origin = payload.origin();
        if origin.is_some_and(|value| value != "background_response_cost_poll") {
            return;
        }
        let fields = &payload.rollup;
        let caller = origin.is_none();
        let success = fields.status.as_deref() == Some("success");
        let duration = fields
            .request_duration_ms
            .filter(|value| caller && success && *value >= 0);
        let model = optional(&fields.model)
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown");
        let group = optional(&fields.model_group)
            .filter(|value| !value.is_empty())
            .unwrap_or(model);
        let provider = optional(&fields.custom_llm_provider)
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown");
        if [date, model, group, provider]
            .into_iter()
            .any(|value| value.len() > 2000)
        {
            return;
        }
        let task_type = payload
            .request_tags
            .iter()
            .filter_map(|tag| tag.strip_prefix("task:"))
            .find(|task| recognized_task(task))
            .unwrap_or("uncategorized");
        *self
            .models
            .entry(ModelKey {
                date,
                model_group: group,
                model,
                custom_llm_provider: provider,
                task_type,
            })
            .or_default() += ModelCounts {
            spend: payload.spend.unwrap_or_default(),
            prompt_tokens: cap(fields.prompt_tokens.unwrap_or_default()),
            completion_tokens: cap(fields.completion_tokens.unwrap_or_default()),
            request_count: i64::from(caller),
            successful_requests: i64::from(caller && success),
            failed_requests: i64::from(caller && !success),
            total_response_time_ms: cap(duration.unwrap_or_default()),
            timed_requests: i64::from(duration.is_some()),
        };
    }

    fn guardrail_rows(&mut self, payload: &'a Payload, date: &str) {
        let fields = &payload.rollup;
        if fields.request_id.as_deref().is_none_or(str::is_empty) {
            return;
        }
        let metadata = &payload.metadata.rollup;
        let items = match &metadata.guardrail_information {
            Some(Guardrails::List(items)) if !items.is_empty() => items,
            Some(Guardrails::One(_)) => return,
            _ => match &metadata.standard_logging_guardrail_information {
                Some(Guardrails::List(items)) => items,
                _ => return,
            },
        };
        let team_id = fields.team_id.as_deref().unwrap_or_default();
        let api_key = optional(&fields.api_key).unwrap_or_default();
        for item in items {
            let Some(guardrail_id) = item
                .guardrail_id
                .as_deref()
                .filter(|value| !value.is_empty())
                .or(item
                    .guardrail_name
                    .as_deref()
                    .filter(|value| !value.is_empty()))
            else {
                continue;
            };
            let Some(usage) = &item.guardrail_usage else {
                continue;
            };
            for (unit, value) in usage {
                let Some(units) = value.as_i64().filter(|units| *units > 0).map(cap) else {
                    continue;
                };
                if [guardrail_id, date, team_id, api_key, unit]
                    .into_iter()
                    .any(|value| value.len() > 2000)
                {
                    continue;
                }
                let cost = item
                    .guardrail_cost_by_unit
                    .as_ref()
                    .filter(|_| item.guardrail_cost_in_spend != Some(false))
                    .and_then(|costs| costs.get(unit))
                    .copied()
                    .flatten();
                *self
                    .guardrails
                    .entry(GuardrailKey {
                        guardrail_id,
                        date: date.to_owned(),
                        team_id,
                        api_key,
                        usage_unit: unit,
                    })
                    .or_default() += GuardrailCounts {
                    units,
                    cost: cost.unwrap_or_default(),
                    untracked_units: if cost.is_none() { units } else { 0 },
                };
            }
        }
    }
}

fn optional(field: &Option<Option<String>>) -> Option<&str> {
    field.as_ref().and_then(|value| value.as_deref())
}

fn customer_row_id(
    id: &str,
    key: &DailyKey<'_>,
    customer: CustomerKey<'_>,
) -> Result<String, Error> {
    let identity = [
        ("api_key", key.api_key),
        ("app_group_id", customer.app_group_id),
        ("app_id", customer.app_id),
        ("custom_llm_provider", key.custom_llm_provider),
        ("customer_member_id", id),
        ("customer_org_id", customer.customer_org_id),
        ("customer_team_id", customer.customer_team_id),
        ("date", &key.date),
        ("endpoint", key.endpoint),
        ("mcp_namespaced_tool_name", key.mcp_namespaced_tool_name),
        ("model", key.model),
        ("model_group", customer.model_group),
    ]
    .into_iter()
    .filter(|&(name, value)| !matches!(name, "app_id" | "app_group_id") || !value.is_empty())
    .map(|(name, value)| {
        PythonValue::List(vec![
            PythonValue::Str(name.to_owned()),
            PythonValue::Str(value.to_owned()),
        ])
    })
    .collect();
    let encoded = json::dumps_compact(&PythonValue::List(identity))
        .map_err(|error| sqlx::Error::Encode(Box::new(error)))?;
    Ok(format!("{:x}", Sha256::digest(encoded.as_bytes())))
}

fn recognized_task(task: &str) -> bool {
    matches!(
        task,
        "classification"
            | "content_writing"
            | "roleplay_fiction"
            | "conversation"
            | "research_reports"
            | "qa_knowledge"
            | "customer_support"
            | "summarization"
            | "translation"
            | "workflow_execution"
            | "multi_step_planning"
            | "tool_dispatch"
            | "code_generation"
            | "debugging"
            | "code_review"
            | "frontend_ui"
            | "file_io"
            | "shell_execution"
            | "data_extraction"
            | "data_transformation"
    )
}

fn endpoint(call: &str) -> &'static str {
    match call {
        "acompletion" => "/chat/completions",
        "atext_completion" => "/completions",
        "aembedding" => "/embeddings",
        "aimage_generation" => "/image/generations",
        "aspeech" => "/audio/speech",
        "atranscription" => "/audio/transcriptions",
        "amoderation" => "/moderations",
        "arerank" => "/rerank",
        "aresponses" => "/responses",
        "_aresponses_websocket" => "/responses",
        "alist_input_items" => "/responses/{response_id}/input_items",
        "aimage_edit" => "/images/edits",
        "acancel_responses" => "/responses/{response_id}/cancel",
        "acompact_responses" => "/responses/compact",
        "aocr" => "/ocr",
        "asearch" => "/search",
        "adecisions" => "/decisions",
        "asystemone" => "/systemone",
        "aextraction" => "/extraction",
        "avideo_generation" => "/videos",
        "avideo_list" => "/videos",
        "avideo_status" => "/videos/{video_id}",
        "avideo_content" => "/videos/{video_id}/content",
        "avideo_remix" => "/videos/{video_id}/remix",
        "avideo_create_character" => "/videos/characters",
        "avideo_get_character" => "/videos/characters/{character_id}",
        "avideo_edit" => "/videos/edits",
        "avideo_extension" => "/videos/extensions",
        "acreate_realtime_client_secret" => "/realtime/client_secrets",
        "arealtime_calls" => "/realtime/calls",
        "acreate_realtime_transcription_session" => "/realtime/transcription_sessions",
        "acreate_container" => "/containers",
        "alist_containers" => "/containers",
        "aretrieve_container" => "/containers/{container_id}",
        "adelete_container" => "/containers/{container_id}",
        "aupload_container_file" => "/containers/{container_id}/files",
        "alist_container_files" => "/containers/{container_id}/files",
        "aretrieve_container_file" => "/containers/{container_id}/files/{file_id}",
        "adelete_container_file" => "/containers/{container_id}/files/{file_id}",
        "aretrieve_container_file_content" => "/containers/{container_id}/files/{file_id}/content",
        "acreate_skill" => "/skills",
        "alist_skills" => "/skills",
        "aget_skill" => "/skills/{skill_id}",
        "adelete_skill" => "/skills/{skill_id}",
        "aingest" => "/rag/ingest",
        "acreate_interaction" => "/interactions",
        "aget_interaction" => "/interactions/{interaction_id}",
        "adelete_interaction" => "/interactions/{interaction_id}",
        "acancel_interaction" => "/interactions/{interaction_id}/cancel",
        "acreate_agent" => "/v1beta/agents",
        "alist_agents" => "/v1beta/agents",
        "aget_agent" => "/v1beta/agents/{name}",
        "adelete_agent" => "/v1beta/agents/{name}",
        "alist_agent_versions" => "/v1beta/agents/{name}/versions",
        "acreate_eval" => "/evals",
        "alist_evals" => "/evals",
        "aget_eval" => "/evals/{eval_id}",
        "aupdate_eval" => "/evals/{eval_id}",
        "adelete_eval" => "/evals/{eval_id}",
        "acancel_eval" => "/evals/{eval_id}/cancel",
        "acreate_run" => "/evals/{eval_id}/runs",
        "alist_runs" => "/evals/{eval_id}/runs",
        "aget_run" => "/evals/{eval_id}/runs/{run_id}",
        "acancel_run" => "/evals/{eval_id}/runs/{run_id}/cancel",
        "adelete_run" => "/evals/{eval_id}/runs/{run_id}",
        "acreate_batch" => "/batches",
        "aretrieve_batch" => "/batches",
        "agenerate_content" => "/models/{model}:generateContent",
        "agenerate_content_stream" => "/models/{model}:streamGenerateContent",
        _ => "",
    }
}
