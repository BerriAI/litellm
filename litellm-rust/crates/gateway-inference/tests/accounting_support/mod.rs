use std::sync::{Arc, Mutex};

use litellm_accounting::{
    ApplyResult, Charges, Cost, Effect, Outcome, ProviderWork, ReportedUsage, Usd,
};
use litellm_gateway_inference::{
    PluginError,
    accounting::{
        AccountingBackend, AccountingCallback, AccountingFuture, AccountingInputs,
        AccountingService, AccountingSettlement, AdmissionRequest, AdmittedCall, Fact, Report,
        ResponsesAccounting,
    },
};
use rstest::fixture;
use rusty_money::iso;
use serde_json::Value;
use tokio::sync::{Notify, Semaphore};

#[derive(Default)]
pub struct Ledger {
    pub admitted: Vec<(String, String, String)>,
    pub effects: Vec<Effect>,
    pub spend: Vec<litellm_accounting::Terminal<Value>>,
    pub released: Vec<String>,
    pub reports: Vec<Report>,
    pub callback_release_counts: Vec<usize>,
    pub prepared_credentials: Vec<bool>,
}

#[fixture]
pub fn ledger() -> Arc<Mutex<Ledger>> {
    Arc::new(Mutex::new(Ledger::default()))
}

pub struct Gate {
    pub entered: Notify,
    pub resume: Semaphore,
}
impl Gate {
    pub fn new() -> Self {
        Self {
            entered: Notify::new(),
            resume: Semaphore::new(0),
        }
    }
    async fn wait(&self) {
        self.entered.notify_one();
        self.resume.acquire().await.unwrap().forget();
    }
}

#[derive(Default)]
pub struct Policy {
    pub reject: bool,
    pub no_reservation: bool,
    pub bad_fact: bool,
    pub bad_price: bool,
    pub unknown_price: bool,
    pub avoided: bool,
    pub result: Option<(Effect, ApplyResult<PluginError>)>,
    pub admission_gate: Option<Arc<Gate>>,
    pub spend_gate: Option<Arc<Gate>>,
}

pub fn failure() -> PluginError {
    Arc::new(std::io::Error::other("injected failure"))
}
pub fn cost(amount: i64) -> Cost {
    Cost::Known(Usd::from_major(amount, iso::USD))
}

pub struct Service {
    pub ledger: Arc<Mutex<Ledger>>,
    pub policy: Arc<Policy>,
}
impl AccountingService for Service {
    fn admit(
        &self,
        request: AdmissionRequest,
    ) -> AccountingFuture<'_, Result<AdmittedCall, PluginError>> {
        Box::pin(async move {
            if self.policy.reject {
                return Err(failure());
            }
            let call_id = {
                let mut ledger = self.ledger.lock().unwrap();
                ledger.admitted.push((
                    request.public_model,
                    request.deployment_model,
                    request.caller.principal().subject().to_owned(),
                ));
                format!("call-{}", ledger.admitted.len())
            };
            if let Some(gate) = &self.policy.admission_gate {
                gate.wait().await;
            }
            Ok(AdmittedCall {
                call_id,
                budget_receipt: (!self.policy.no_reservation).then(|| "reservation".into()),
                backend: Box::new(Store {
                    ledger: self.ledger.clone(),
                    policy: self.policy.clone(),
                    usage: ReportedUsage::Unknown,
                }),
            })
        })
    }
}

struct Store {
    ledger: Arc<Mutex<Ledger>>,
    policy: Arc<Policy>,
    usage: ReportedUsage<Value>,
}
impl AccountingBackend for Store {
    fn observe(&mut self, fact: Fact) -> Result<(), PluginError> {
        if self.policy.bad_fact {
            return Err(failure());
        }
        let usage = match fact {
            Fact::Prepared(context) => {
                self.ledger
                    .lock()
                    .unwrap()
                    .prepared_credentials
                    .push(context.api_key.is_some());
                None
            }
            Fact::ProviderResponse(raw) => serde_json::from_str::<Value>(&raw.body)
                .ok()
                .and_then(|value| value.get("usage").cloned()),
            Fact::Response(response) => response.extra.get("usage").cloned(),
            Fact::Chunk(chunk) => std::str::from_utf8(&chunk)
                .ok()
                .and_then(|text| text.lines().find_map(|line| line.strip_prefix("data: ")))
                .and_then(|data| serde_json::from_str::<Value>(data).ok())
                .and_then(|event| {
                    event
                        .get("response")
                        .and_then(|response| response.get("usage"))
                        .cloned()
                }),
        };
        if let Some(usage) = usage {
            self.usage = ReportedUsage::Known(usage);
        }
        Ok(())
    }
    fn inputs(&self) -> AccountingInputs {
        AccountingInputs {
            work: if self.policy.avoided {
                ProviderWork::NotStarted
            } else {
                ProviderWork::Unknown
            },
            usage: self.usage.clone(),
        }
    }
    fn price(&mut self, _: Outcome) -> AccountingFuture<'_, Result<Charges, PluginError>> {
        Box::pin(async move {
            if self.policy.bad_price {
                return Err(failure());
            }
            Ok(Charges::new(
                if self.policy.unknown_price {
                    Cost::Unknown
                } else {
                    cost(3)
                },
                cost(1),
            )
            .unwrap())
        })
    }
    fn apply<'a>(
        &'a mut self,
        effect: Effect,
        settlement: AccountingSettlement<'a>,
    ) -> AccountingFuture<'a, ApplyResult<PluginError>> {
        Box::pin(async move {
            self.ledger.lock().unwrap().effects.push(effect);
            if effect == Effect::RecordSpend
                && let Some(gate) = &self.policy.spend_gate
            {
                gate.wait().await;
            }
            if let Some((target, result)) = &self.policy.result
                && effect == *target
            {
                return result.clone();
            }
            let mut ledger = self.ledger.lock().unwrap();
            match effect {
                Effect::RecordSpend => ledger.spend.push(settlement.terminal.clone()),
                Effect::ReleaseBudgetReservation => ledger
                    .released
                    .push(settlement.admission.budget().unwrap().clone()),
                Effect::ReconcileBudget => {}
            }
            ApplyResult::Committed
        })
    }
}

pub struct Callback {
    pub ledger: Arc<Mutex<Ledger>>,
    pub fail: bool,
}
impl AccountingCallback for Callback {
    fn completed(&self, report: Report) -> AccountingFuture<'_, Result<(), PluginError>> {
        Box::pin(async move {
            {
                let mut ledger = self.ledger.lock().unwrap();
                let released = ledger.released.len();
                ledger.callback_release_counts.push(released);
                ledger.reports.push(report);
            }
            if self.fail { Err(failure()) } else { Ok(()) }
        })
    }
}

pub fn runtime(
    ledger: &Arc<Mutex<Ledger>>,
    policy: Policy,
    callback: Option<bool>,
) -> ResponsesAccounting {
    let accounting = ResponsesAccounting::new(Arc::new(Service {
        ledger: ledger.clone(),
        policy: Arc::new(policy),
    }));
    match callback {
        Some(fail) => accounting.with_callback(Arc::new(Callback {
            ledger: ledger.clone(),
            fail,
        })),
        None => accounting,
    }
}
