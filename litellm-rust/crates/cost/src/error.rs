#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostError {
    InvalidRate,
    TokenCountOverflow,
    InvalidUsage,
    ModelNotFound,
}
