SELECT jsonb_build_object(
    'customer_member_id', m.customer_member_id,
    'alias', m.alias,
    'spend', m.spend,
    'budget', to_jsonb(mb),
    'budget_override', to_jsonb(mo),
    'budget_override_expires_at', m.budget_override_expires_at,
    'parent_org', jsonb_build_object(
        'customer_org_id', o.customer_org_id,
        'alias', o.alias,
        'spend', o.spend,
        'budget', to_jsonb(ob)
    ),
    'teams', COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'customer_team_id', t.customer_team_id,
            'alias', t.alias,
            'spend', t.spend,
            'budget', to_jsonb(tb),
            'budget_override', to_jsonb(tbo),
            'budget_override_expires_at', t.budget_override_expires_at
        ) ORDER BY parent.position)
        FROM unnest(m.parent_team_ids) WITH ORDINALITY AS parent(id, position)
        JOIN "LiteLLM_CustomerTeamTable" AS t ON t.customer_team_id = parent.id
        LEFT JOIN "LiteLLM_BudgetTable" AS tb ON tb.budget_id = t.budget_id
        LEFT JOIN "LiteLLM_BudgetTable" AS tbo ON tbo.budget_id = t.budget_override_id
    ), '[]'::jsonb)
) AS "row!: Json<CustomerMemberRow>"
FROM "LiteLLM_CustomerMemberTable" AS m
JOIN "LiteLLM_CustomerOrganizationTable" AS o ON o.customer_org_id = m.parent_org_id
LEFT JOIN "LiteLLM_BudgetTable" AS ob ON ob.budget_id = o.budget_id
LEFT JOIN "LiteLLM_BudgetTable" AS mb ON mb.budget_id = m.budget_id
LEFT JOIN "LiteLLM_BudgetTable" AS mo ON mo.budget_id = m.budget_override_id
WHERE m.customer_member_id = $1
