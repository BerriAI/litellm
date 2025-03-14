import { KeyResponse } from "./key_list";

export const transformKeyInfo = (apiResponse: any): KeyResponse => {
  const { key, info } = apiResponse;
  
  return {
    token: key,
    key_name: info.key_name,
    key_alias: info.key_alias,
    spend: info.spend,
    expires: info.expires,
    models: info.models,
    aliases: info.aliases,
    config: info.config,
    user_id: info.user_id,
    team_id: info.team_id,
    permissions: info.permissions,
    max_parallel_requests: info.max_parallel_requests,
    metadata: info.metadata,
    tpm_limit: info.tpm_limit,
    rpm_limit: info.rpm_limit,
    max_budget: info.max_budget,
    budget_duration: info.budget_duration,
    organization_id: info.organization_id,
    created_at: info.created_at,
    litellm_budget_table: info.litellm_budget_table,
  };
};