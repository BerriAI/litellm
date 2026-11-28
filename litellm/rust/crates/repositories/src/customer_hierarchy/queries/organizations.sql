SELECT jsonb_build_object(
    'customer_org_id', o.customer_org_id,
    'alias', o.alias,
    'spend', o.spend,
    'budget', to_jsonb(b)
) AS "row!: Json<CustomerOrgRow>"
FROM "LiteLLM_CustomerOrganizationTable" AS o
LEFT JOIN "LiteLLM_BudgetTable" AS b USING (budget_id)
WHERE o.alias = $1
LIMIT 2
