UPDATE "LiteLLM_AutoRouterBaselineComparison"
SET published_revision = revision, history = $2
WHERE scope = $1
