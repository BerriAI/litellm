ALTER TABLE {database}.agent_traces_by_key
    ADD COLUMN IF NOT EXISTS ReceivedMs SimpleAggregateFunction(min, UInt64) DEFAULT 0
