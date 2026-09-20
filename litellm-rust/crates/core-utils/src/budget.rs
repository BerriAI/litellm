use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("Budget has been exceeded! Current cost: {current_cost}, Max budget: {max_budget}")]
pub struct BudgetExceeded {
    pub current_cost: f64,
    pub max_budget: f64,
}

pub fn check_budget(current_cost: f64, max_budget: Option<f64>) -> Result<(), BudgetExceeded> {
    if let Some(max_budget) = max_budget.filter(|limit| *limit != 0.0)
        && current_cost > max_budget
    {
        return Err(BudgetExceeded {
            current_cost,
            max_budget,
        });
    }
    Ok(())
}

#[derive(Default)]
pub struct Budget {
    current_cost: Mutex<f64>,
    pub max_budget: Option<f64>,
}

impl Budget {
    pub fn new(max_budget: Option<f64>, current_cost: f64) -> Self {
        Self {
            current_cost: Mutex::new(current_cost),
            max_budget,
        }
    }

    pub fn current_cost(&self) -> f64 {
        *self
            .current_cost
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    pub fn check(&self) -> Result<(), BudgetExceeded> {
        check_budget(self.current_cost(), self.max_budget)
    }

    pub fn record(&self, cost: f64) {
        *self
            .current_cost
            .lock()
            .unwrap_or_else(|error| error.into_inner()) += cost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_truthiness_and_strict_boundary() {
        assert!(check_budget(100.0, None).is_ok());
        assert!(check_budget(100.0, Some(0.0)).is_ok());
        assert!(check_budget(1.0, Some(1.0)).is_ok());
        assert_eq!(check_budget(1.5, Some(1.0)).unwrap_err().current_cost, 1.5);
    }

    #[test]
    fn concurrent_completions_do_not_lose_spend() {
        let budget = Budget::new(Some(10.0), 0.0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..100 {
                        budget.record(0.25);
                    }
                });
            }
        });
        assert_eq!(budget.current_cost(), 200.0);
        assert!(budget.check().is_err());
    }
}
