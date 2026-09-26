SELECT revision, published_revision, initial_equivalent, retired, history
FROM "LiteLLM_AutoRouterBaselineComparison" WHERE scope = $1 FOR UPDATE
