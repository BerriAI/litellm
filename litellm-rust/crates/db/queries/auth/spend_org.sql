SELECT COALESCE(spend, 0.0)::float8 AS "spend!" FROM "LiteLLM_OrganizationTable" WHERE organization_id=$1
