ALTER TABLE {database}.agent_traces_by_key
    ADD COLUMN IF NOT EXISTS AgentLabels SimpleAggregateFunction(groupUniqArrayArray, Array(String)) DEFAULT [],
    ADD COLUMN IF NOT EXISTS AgentIdentities SimpleAggregateFunction(groupUniqArrayArray, Array(String)) DEFAULT [],
    ADD COLUMN IF NOT EXISTS Frameworks SimpleAggregateFunction(groupUniqArrayArray, Array(String)) DEFAULT []
