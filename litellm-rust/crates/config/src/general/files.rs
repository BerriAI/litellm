use crate::Spelled;

#[macro_rules_attribute::apply(crate::config_section)]
pub struct FilesSettings {
    pub max_file_size_mb: Option<Spelled<i64>>,
    pub max_batch_file_size_mb: Option<Spelled<i64>>,
    pub max_batch_file_records: Option<Spelled<i64>>,
    pub max_batch_file_uploads_per_day: Option<Spelled<i64>>,
    pub max_file_downloads_per_minute: Option<Spelled<i64>>,
    pub allowed_file_extensions: Option<Box<[String]>>,
    pub blocked_file_extensions: Option<Box<[String]>>,
    pub disable_batch_input_file_rate_limiting: Option<Spelled<bool>>,
    pub skip_batch_input_file_rate_limiting_for_providers: Option<Box<[String]>>,
    pub skip_batch_input_file_rate_limiting_for_models: Option<Box<[String]>>,
}
