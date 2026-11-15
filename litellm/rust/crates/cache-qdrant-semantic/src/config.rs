#[derive(Clone, Debug, PartialEq)]
pub enum Quantization {
    Binary,
    Scalar,
    Product,
}

pub struct QdrantSemanticConfig {
    pub collection_name: String,
    pub similarity_threshold: f64,
    pub vector_size: u64,
    pub quantization: Quantization,
}
