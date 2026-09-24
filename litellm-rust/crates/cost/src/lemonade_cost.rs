use crate::catalog::{ModelCostRequest, ModelInfoCatalog};

pub fn lemonade_cost_per_token(
    _catalog: &ModelInfoCatalog,
    _request: ModelCostRequest<'_>,
) -> (f64, f64) {
    (0.0, 0.0)
}
