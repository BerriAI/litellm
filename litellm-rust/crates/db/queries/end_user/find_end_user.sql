SELECT e.spend AS "spend?",
       e.blocked AS "blocked?",
       b.max_budget AS "max_budget?",
       b.rpm_limit AS "rpm_limit?",
       b.tpm_limit AS "tpm_limit?"
FROM (SELECT 1) AS one
LEFT JOIN "LiteLLM_EndUserTable" AS e ON e.user_id = $1
LEFT JOIN "LiteLLM_BudgetTable" AS assigned ON assigned.budget_id = e.budget_id
LEFT JOIN "LiteLLM_BudgetTable" AS key_budget ON key_budget.budget_id = $3
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = COALESCE(assigned.budget_id, key_budget.budget_id, $2)
