SELECT o.customer_org_id,
       o.alias,
       o.spend,
       COALESCE(o.budget_template_ids, '{}') AS "budget_template_ids!",
       o.budget_id,
       o.default_budget_template_id,
       o.created_at,
       o.created_by,
       o.updated_at,
       o.updated_by,
       to_jsonb(b) AS "budget?: Json<Budget>"
FROM "LiteLLM_CustomerOrganizationTable" AS o
LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = o.budget_id
WHERE o.customer_org_id = $1
