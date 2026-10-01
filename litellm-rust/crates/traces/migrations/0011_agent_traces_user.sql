ALTER TABLE {database}.agent_traces_by_key
    ADD COLUMN IF NOT EXISTS UserId SimpleAggregateFunction(any, String) DEFAULT ''
