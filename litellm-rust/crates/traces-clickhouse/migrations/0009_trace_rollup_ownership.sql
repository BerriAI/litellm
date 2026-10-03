ALTER TABLE {database}.agent_traces_by_key
    ADD COLUMN IF NOT EXISTS UserIds SimpleAggregateFunction(groupUniqArrayArray, Array(String)) DEFAULT [],
    ADD COLUMN IF NOT EXISTS IdentifiedLlmCount SimpleAggregateFunction(sum, UInt64) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS ReceivedMs SimpleAggregateFunction(min, UInt64) DEFAULT 0
