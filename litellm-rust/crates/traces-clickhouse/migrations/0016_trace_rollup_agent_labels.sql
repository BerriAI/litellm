ALTER TABLE {database}.agent_traces_by_key
    ADD COLUMN IF NOT EXISTS AgentLabels SimpleAggregateFunction(groupUniqArrayArray, Array(String)) DEFAULT []
