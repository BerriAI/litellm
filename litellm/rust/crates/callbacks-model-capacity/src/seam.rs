use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{
    error::{CapacityError, Raised},
    model::{
        BorrowingLimits, Counter, Descriptor, GroupInfo, Increment, LimiterResponse, ReservationOp,
        Settings,
    },
};

pub type SeamFuture<'a, T> = BoxFuture<'a, Result<T, CapacityError>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "tokens", rename_all = "snake_case")]
pub enum Estimate {
    NoBody,
    Tokens(i64),
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trace {
    Untraced,
    Attempt,
    Key,
}

pub trait V3Limiter: Send + Sync {
    fn read_counter<'a>(
        &'a self,
        counter: &'a Counter,
        backfill_ttl: &'a Value,
    ) -> SeamFuture<'a, Option<Value>>;
    fn read_local_counter<'a>(&'a self, counter: &'a Counter) -> SeamFuture<'a, Option<Value>>;
    fn check_and_increment<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        increments: &'a [Increment],
    ) -> SeamFuture<'a, LimiterResponse>;
    fn should_rate_limit<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        read_only: bool,
        skip_parallel_check: bool,
    ) -> SeamFuture<'a, LimiterResponse>;
    fn apply<'a>(&'a self, operations: &'a [ReservationOp], trace: Trace) -> SeamFuture<'a, ()>;
    fn borrowing_limits<'a>(&'a self, model: &'a str) -> SeamFuture<'a, Option<BorrowingLimits>>;
    fn model_group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, String>;
    fn estimate_tokens<'a>(
        &'a self,
        tpm_share: i64,
        model: &'a str,
        call_type: Option<&'a str>,
    ) -> SeamFuture<'a, Estimate>;
    fn cap_output<'a>(&'a self, tpm_share: i64, call_type: Option<&'a str>) -> SeamFuture<'a, ()>;
    fn publish<'a>(&'a self, responses: &'a [u64]) -> SeamFuture<'a, ()>;
    fn stash_call_type(&self) -> SeamFuture<'_, Option<String>>;
    fn post_call_success<'a>(&'a self, headers: &'a [(String, Value)]) -> SeamFuture<'a, ()>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Throttle {
    pub model_group: String,
    pub team: String,
    pub team_alias: Option<String>,
    pub priority: Value,
    pub rate_limit_type: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShareGauge {
    pub model_group: String,
    pub team: String,
    pub team_alias: Option<String>,
    pub priority: Value,
    pub share_tokens: i64,
    pub borrowed_tokens: i64,
}

pub trait CapacityHost: Send + Sync {
    fn settings(&self) -> SeamFuture<'_, Settings>;
    fn group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, Option<GroupInfo>>;
    fn touch<'a>(
        &'a self,
        model: &'a str,
        member: &'a str,
        priority_class: &'a Value,
        window: i64,
    ) -> SeamFuture<'a, Vec<(String, Value)>>;
    fn metrics_enabled(&self) -> SeamFuture<'_, bool>;
    fn throttled<'a>(&'a self, event: &'a Throttle) -> SeamFuture<'a, ()>;
    fn set_share<'a>(&'a self, gauge: &'a ShareGauge) -> SeamFuture<'a, ()>;
    fn unlicensed_priority(&self) -> SeamFuture<'_, ()>;
    fn log<'a>(&'a self, level: &'static str, message: &'a str) -> SeamFuture<'a, ()>;
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Reply {
    Ok(Value),
    Raised(Raised),
}

pub trait Delegate: Send + Sync + 'static {
    fn call(&self, method: &'static str, args: Value) -> BoxFuture<'static, Result<Value, String>>;
}

pub struct Delegated<D>(pub D);

impl<D: Delegate> Delegated<D> {
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
            let reply: Reply =
                serde_json::from_value(raw).map_err(|error| CapacityError::Shape {
                    method,
                    detail: error.to_string(),
                })?;
            match reply {
                Reply::Ok(value) => {
                    serde_json::from_value(value).map_err(|error| CapacityError::Shape {
                        method,
                        detail: error.to_string(),
                    })
                }
                Reply::Raised(raised) => Err(CapacityError::Raised(raised)),
            }
        })
    }
}

impl<D: Delegate> V3Limiter for Delegated<D> {
    fn read_counter<'a>(
        &'a self,
        counter: &'a Counter,
        backfill_ttl: &'a Value,
    ) -> SeamFuture<'a, Option<Value>> {
        self.invoke(
            "read_counter",
            json!({"counter": counter, "backfill_ttl": backfill_ttl}),
        )
    }

    fn read_local_counter<'a>(&'a self, counter: &'a Counter) -> SeamFuture<'a, Option<Value>> {
        self.invoke("read_local_counter", json!({"counter": counter}))
    }

    fn check_and_increment<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        increments: &'a [Increment],
    ) -> SeamFuture<'a, LimiterResponse> {
        self.invoke(
            "check_and_increment",
            json!({"descriptors": descriptors, "increments": increments}),
        )
    }

    fn should_rate_limit<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        read_only: bool,
        skip_parallel_check: bool,
    ) -> SeamFuture<'a, LimiterResponse> {
        self.invoke(
            "should_rate_limit",
            json!({"descriptors": descriptors, "read_only": read_only, "skip_parallel_check": skip_parallel_check}),
        )
    }

    fn apply<'a>(&'a self, operations: &'a [ReservationOp], trace: Trace) -> SeamFuture<'a, ()> {
        self.invoke("apply", json!({"operations": operations, "trace": trace}))
    }

    fn borrowing_limits<'a>(&'a self, model: &'a str) -> SeamFuture<'a, Option<BorrowingLimits>> {
        self.invoke("borrowing_limits", json!({"model": model}))
    }

    fn model_group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, String> {
        self.invoke("model_group", json!({"model": model}))
    }

    fn estimate_tokens<'a>(
        &'a self,
        tpm_share: i64,
        model: &'a str,
        call_type: Option<&'a str>,
    ) -> SeamFuture<'a, Estimate> {
        self.invoke(
            "estimate_tokens",
            json!({"tpm_share": tpm_share, "model": model, "call_type": call_type}),
        )
    }

    fn cap_output<'a>(&'a self, tpm_share: i64, call_type: Option<&'a str>) -> SeamFuture<'a, ()> {
        self.invoke(
            "cap_output",
            json!({"tpm_share": tpm_share, "call_type": call_type}),
        )
    }

    fn publish<'a>(&'a self, responses: &'a [u64]) -> SeamFuture<'a, ()> {
        self.invoke("publish", json!({"responses": responses}))
    }

    fn stash_call_type(&self) -> SeamFuture<'_, Option<String>> {
        self.invoke("stash_call_type", json!({}))
    }

    fn post_call_success<'a>(&'a self, headers: &'a [(String, Value)]) -> SeamFuture<'a, ()> {
        self.invoke("post_call_success", json!({"headers": headers}))
    }
}

impl<D: Delegate> CapacityHost for Delegated<D> {
    fn settings(&self) -> SeamFuture<'_, Settings> {
        self.invoke("settings", json!({}))
    }

    fn group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, Option<GroupInfo>> {
        self.invoke("group", json!({"model": model}))
    }

    fn touch<'a>(
        &'a self,
        model: &'a str,
        member: &'a str,
        priority_class: &'a Value,
        window: i64,
    ) -> SeamFuture<'a, Vec<(String, Value)>> {
        self.invoke(
            "touch",
            json!({"model": model, "member": member, "priority_class": priority_class, "window": window}),
        )
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

    fn unlicensed_priority(&self) -> SeamFuture<'_, ()> {
        self.invoke("unlicensed_priority", json!({}))
    }

    fn log<'a>(&'a self, level: &'static str, message: &'a str) -> SeamFuture<'a, ()> {
        self.invoke("log", json!({"level": level, "message": message}))
    }
}
