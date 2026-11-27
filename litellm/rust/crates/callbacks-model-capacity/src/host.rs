use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use litellm_callbacks_parallel_request_limiter::Ctx;
use litellm_redis_store::RequestRedis;

use crate::{
    error::{CapacityError, Raised},
    model::{
        BorrowerKind, BorrowingLimits, Counter, Descriptor, GroupInfo, Increment, LimiterResponse,
        ReservationOp, Settings, Status,
    },
};

pub type SeamFuture<'a, T> = BoxFuture<'a, Result<T, CapacityError>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "tokens", rename_all = "snake_case")]
pub enum Estimate {
    NoBody,
    Tokens(i64),
    Failed,
    Unpriced,
}

pub trait CounterStore: Send + Sync {
    fn read_counter<'a>(
        &'a self,
        counter: &'a Counter,
        backfill_ttl: &'a Value,
    ) -> SeamFuture<'a, Option<Value>>;
    fn read_local_counter(&self, counter: &Counter) -> Result<Option<Value>, CapacityError>;
    fn check_and_increment<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        increments: &'a [Increment],
        ctx: Ctx<'a>,
    ) -> SeamFuture<'a, LimiterResponse>;
    fn should_rate_limit<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        read_only: bool,
        skip_parallel_check: bool,
        ctx: Ctx<'a>,
    ) -> SeamFuture<'a, LimiterResponse>;
    fn apply<'a>(
        &'a self,
        operations: &'a [ReservationOp],
        redis: Option<&'a RequestRedis>,
    ) -> SeamFuture<'a, ()>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Published {
    pub overall_code: String,
    pub statuses: Vec<Status>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Throttle {
    pub model_group: String,
    pub holder_kind: BorrowerKind,
    pub holder_id: String,
    pub holder_alias: Option<String>,
    pub priority: Value,
    pub rate_limit_type: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShareGauge {
    pub model_group: String,
    pub holder_kind: BorrowerKind,
    pub holder_id: String,
    pub holder_alias: Option<String>,
    pub priority: Value,
    pub share_tokens: i64,
    pub borrowed_tokens: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fact<T> {
    Ok(T),
    Raised(Raised),
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Facts {
    #[serde(default)]
    pub settings: Option<Fact<Settings>>,
    #[serde(default)]
    pub group: Option<Fact<Option<GroupInfo>>>,
    #[serde(default)]
    pub borrowing: Option<Fact<Vec<BorrowingLimits>>>,
    #[serde(default)]
    pub call_type: Option<Fact<Option<String>>>,
}

fn known<T: Clone>(fact: Option<&Fact<T>>, method: &'static str) -> Result<T, CapacityError> {
    match fact {
        Some(Fact::Ok(value)) => Ok(value.clone()),
        Some(Fact::Raised(raised)) => Err(CapacityError::Raised(raised.clone())),
        None => Err(CapacityError::Shape {
            method,
            detail: "the host passed no such fact".into(),
        }),
    }
}

impl Facts {
    pub fn settings(&self) -> Result<Settings, CapacityError> {
        known(self.settings.as_ref(), "settings")
    }

    pub fn group(&self) -> Result<Option<GroupInfo>, CapacityError> {
        known(self.group.as_ref(), "group")
    }

    pub fn borrowing(&self) -> Result<Vec<BorrowingLimits>, CapacityError> {
        known(self.borrowing.as_ref(), "borrowing_limits")
    }

    pub fn call_type(&self) -> Result<Option<String>, CapacityError> {
        known(self.call_type.as_ref(), "stash_call_type")
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Touched {
    Members(Vec<(String, Value)>),
    Pending(u64),
}

pub trait CapacityHost: Send + Sync {
    fn touch(
        &self,
        model: &str,
        member: &str,
        priority_class: &Value,
        window: i64,
    ) -> Result<Touched, CapacityError>;
    fn touched(&self, pending: u64) -> SeamFuture<'_, Vec<(String, Value)>>;
    fn metrics_enabled(&self) -> SeamFuture<'_, bool>;
    fn throttled<'a>(&'a self, event: &'a Throttle) -> SeamFuture<'a, ()>;
    fn set_share<'a>(&'a self, gauge: &'a ShareGauge) -> SeamFuture<'a, ()>;
    fn unlicensed_priority(&self) -> Result<(), CapacityError>;
    fn log(&self, level: &'static str, message: &str) -> Result<(), CapacityError>;
    fn model_group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, String>;
    fn estimate_tokens<'a>(
        &'a self,
        tpm_share: i64,
        model: Option<&'a str>,
        call_type: Option<&'a str>,
    ) -> SeamFuture<'a, Estimate>;
    fn post_call_success<'a>(&'a self, headers: &'a [(String, Value)]) -> SeamFuture<'a, ()>;
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Reply {
    Ok(Value),
    Raised(Raised),
}

pub trait HostTransport: Send + Sync + 'static {
    fn call(&self, method: &'static str, args: Value) -> BoxFuture<'static, Result<Value, String>>;
    fn call_now(&self, method: &'static str, args: Value) -> Result<Value, String>;
}

fn answered<T: DeserializeOwned>(method: &'static str, raw: Value) -> Result<T, CapacityError> {
    let reply: Reply = serde_json::from_value(raw).map_err(|error| CapacityError::Shape {
        method,
        detail: error.to_string(),
    })?;
    match reply {
        Reply::Ok(value) => serde_json::from_value(value).map_err(|error| CapacityError::Shape {
            method,
            detail: error.to_string(),
        }),
        Reply::Raised(raised) => Err(CapacityError::Raised(raised)),
    }
}

pub struct JsonHost<D>(pub D);

impl<D: HostTransport> JsonHost<D> {
    fn invoke<'a, T: DeserializeOwned + Send + 'a>(
        &'a self,
        method: &'static str,
        args: Value,
    ) -> SeamFuture<'a, T> {
        let pending = self.0.call(method, args);
        Box::pin(async move {
            let raw = pending
                .await
                .map_err(|detail| CapacityError::Shape { method, detail })?;
            answered(method, raw)
        })
    }

    fn invoke_now<T: DeserializeOwned>(
        &self,
        method: &'static str,
        args: Value,
    ) -> Result<T, CapacityError> {
        let raw = self
            .0
            .call_now(method, args)
            .map_err(|detail| CapacityError::Shape { method, detail })?;
        answered(method, raw)
    }
}

impl<D: HostTransport> CapacityHost for JsonHost<D> {
    fn touch(
        &self,
        model: &str,
        member: &str,
        priority_class: &Value,
        window: i64,
    ) -> Result<Touched, CapacityError> {
        self.invoke_now(
            "touch",
            json!({"model": model, "member": member, "priority_class": priority_class, "window": window}),
        )
    }

    fn touched(&self, pending: u64) -> SeamFuture<'_, Vec<(String, Value)>> {
        self.invoke("touched", json!({"pending": pending}))
    }

    fn metrics_enabled(&self) -> SeamFuture<'_, bool> {
        self.invoke("metrics_enabled", json!({}))
    }

    fn throttled<'a>(&'a self, event: &'a Throttle) -> SeamFuture<'a, ()> {
        self.invoke("throttled", json!(event))
    }

    fn set_share<'a>(&'a self, gauge: &'a ShareGauge) -> SeamFuture<'a, ()> {
        self.invoke("set_share", json!(gauge))
    }

    fn unlicensed_priority(&self) -> Result<(), CapacityError> {
        self.invoke_now("unlicensed_priority", json!({}))
    }

    fn log(&self, level: &'static str, message: &str) -> Result<(), CapacityError> {
        self.invoke_now("log", json!({"level": level, "message": message}))
    }

    fn model_group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, String> {
        self.invoke("model_group", json!({"model": model}))
    }

    fn estimate_tokens<'a>(
        &'a self,
        tpm_share: i64,
        model: Option<&'a str>,
        call_type: Option<&'a str>,
    ) -> SeamFuture<'a, Estimate> {
        self.invoke(
            "estimate_tokens",
            json!({"tpm_share": tpm_share, "model": model, "call_type": call_type}),
        )
    }

    fn post_call_success<'a>(&'a self, headers: &'a [(String, Value)]) -> SeamFuture<'a, ()> {
        self.invoke("post_call_success", json!({"headers": headers}))
    }
}
