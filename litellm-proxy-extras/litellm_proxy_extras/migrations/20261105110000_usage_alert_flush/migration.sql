CREATE TABLE IF NOT EXISTS "LiteLLM_UsageAlertFlush" (
    source TEXT NOT NULL,
    hour TIMESTAMP(3) NOT NULL,
    observed_at TIMESTAMP(3) NOT NULL,
    PRIMARY KEY (source, hour)
);
