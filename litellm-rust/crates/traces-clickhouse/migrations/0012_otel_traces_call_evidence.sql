ALTER TABLE {database}.otel_traces
    ADD COLUMN IF NOT EXISTS WrapperCandidate Bool DEFAULT false AFTER ObservationType,
    ADD COLUMN IF NOT EXISTS CallKeys Array(String) DEFAULT [] AFTER LiteLLMRequestId,
    ADD COLUMN IF NOT EXISTS CallEvidence LowCardinality(String) DEFAULT '' AFTER CallKeys,
    ADD COLUMN IF NOT EXISTS ToolCallId String DEFAULT '' AFTER Output
