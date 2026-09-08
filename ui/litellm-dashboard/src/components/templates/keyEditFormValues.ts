import { z } from "zod/v4";

import { KeyResponse } from "../key_team_helpers/key_list";
import { extractLoggingSettings, formatMetadataForDisplay, stripTagsFromMetadata } from "../key_info_utils";
import { mapInternalToDisplayNames } from "../callback_info_helpers";
import { estimateChecks, estimateFields } from "./estimatedOutputTokens";
import { canonicalBudgetDuration } from "./keyEditFieldNormalizers";

export interface McpServersAndGroups {
  servers: string[];
  accessGroups: string[];
  toolsets: string[];
}

export interface AgentsAndGroups {
  agents: string[];
  accessGroups: string[];
}

export interface KeyEditFormValues {
  key_alias?: string;
  models?: string[];
  allowed_routes?: string;
  max_budget?: number | string | null;
  budget_duration?: string | null;
  tpm_limit?: number | string | null;
  tpm_limit_type?: string | null;
  rpm_limit?: number | string | null;
  rpm_limit_type?: string | null;
  throttle_on_budget_exceeded?: boolean;
  enable_prompt_caching?: boolean;
  max_parallel_requests?: number | string | null;
  model_tpm_limit?: string;
  model_rpm_limit?: string;
  default_estimated_output_tokens?: number | string | null;
  default_estimated_output_tokens_per_model?: string;
  guardrails?: string[];
  disable_global_guardrails?: boolean;
  policies?: string[];
  tags?: string[];
  prompts?: string[];
  access_group_ids?: string[];
  allowed_passthrough_routes?: string[];
  vector_stores?: string[];
  mcp_servers_and_groups?: McpServersAndGroups;
  mcp_tool_permissions?: Record<string, string[]>;
  agents_and_groups?: AgentsAndGroups;
  organization_id?: string | null;
  team_id?: string | null;
  logging_settings?: unknown[];
  metadata?: string;
  duration?: string | null;
  token?: string;
  disabled_callbacks?: string[];
  auto_rotate?: boolean;
  rotation_interval?: string;
}

const readMetadata = (keyData: KeyResponse, key: string): unknown =>
  keyData.metadata != null && typeof keyData.metadata === "object"
    ? (keyData.metadata as Record<string, unknown>)[key]
    : undefined;

export const toKeyEditFormValues = (keyData: KeyResponse): KeyEditFormValues => ({
  key_alias: keyData.key_alias,
  models: keyData.models,
  allowed_routes:
    Array.isArray(keyData.allowed_routes) && keyData.allowed_routes.length > 0 ? keyData.allowed_routes.join(", ") : "",
  max_budget: keyData.max_budget,
  budget_duration: canonicalBudgetDuration(keyData.budget_duration),
  tpm_limit: keyData.tpm_limit,
  tpm_limit_type: (keyData as { tpm_limit_type?: string | null }).tpm_limit_type ?? null,
  rpm_limit: keyData.rpm_limit,
  rpm_limit_type: (keyData as { rpm_limit_type?: string | null }).rpm_limit_type ?? null,
  throttle_on_budget_exceeded: Boolean(readMetadata(keyData, "throttle_on_budget_exceeded")),
  enable_prompt_caching: Boolean(readMetadata(keyData, "enable_prompt_caching")),
  max_parallel_requests: keyData.max_parallel_requests,
  model_tpm_limit: (keyData as { model_tpm_limit?: string }).model_tpm_limit,
  model_rpm_limit: (keyData as { model_rpm_limit?: string }).model_rpm_limit,
  ...(estimateFields(keyData.metadata as Record<string, unknown> | null | undefined) as {
    default_estimated_output_tokens?: number | string | null;
    default_estimated_output_tokens_per_model?: string;
  }),
  guardrails: readMetadata(keyData, "guardrails") as string[] | undefined,
  disable_global_guardrails: Boolean(readMetadata(keyData, "disable_global_guardrails")),
  policies: (keyData as { policies?: string[] }).policies,
  tags: readMetadata(keyData, "tags") as string[] | undefined,
  prompts: readMetadata(keyData, "prompts") as string[] | undefined,
  access_group_ids: keyData.access_group_ids || [],
  allowed_passthrough_routes: (keyData as { allowed_passthrough_routes?: string[] }).allowed_passthrough_routes,
  vector_stores: keyData.object_permission?.vector_stores || [],
  mcp_servers_and_groups: {
    servers: keyData.object_permission?.mcp_servers || [],
    accessGroups: keyData.object_permission?.mcp_access_groups || [],
    toolsets: keyData.object_permission?.mcp_toolsets || [],
  },
  mcp_tool_permissions: keyData.object_permission?.mcp_tool_permissions || {},
  agents_and_groups: {
    agents: keyData.object_permission?.agents || [],
    accessGroups: keyData.object_permission?.agent_access_groups || [],
  },
  organization_id: keyData.organization_id,
  team_id: keyData.team_id,
  logging_settings: extractLoggingSettings(keyData.metadata),
  metadata: formatMetadataForDisplay(stripTagsFromMetadata(keyData.metadata)),
  duration: (keyData as { duration?: string }).duration ?? "",
  token: keyData.token || keyData.token_id,
  disabled_callbacks: Array.isArray(readMetadata(keyData, "litellm_disabled_callbacks"))
    ? mapInternalToDisplayNames(readMetadata(keyData, "litellm_disabled_callbacks") as string[])
    : [],
  auto_rotate: keyData.auto_rotate || false,
  rotation_interval: keyData.rotation_interval,
});

export const keyEditFormSchema = z.object({
  key_alias: z.custom<string | undefined>(),
  models: z.custom<string[] | undefined>(),
  allowed_routes: z.custom<string | undefined>(),
  max_budget: z.custom<number | string | null | undefined>(),
  budget_duration: z.custom<string | null | undefined>(),
  tpm_limit: z.custom<number | string | null | undefined>(),
  tpm_limit_type: z.custom<string | null | undefined>(),
  rpm_limit: z.custom<number | string | null | undefined>(),
  rpm_limit_type: z.custom<string | null | undefined>(),
  throttle_on_budget_exceeded: z.custom<boolean | undefined>(),
  enable_prompt_caching: z.custom<boolean | undefined>(),
  max_parallel_requests: z.custom<number | string | null | undefined>(),
  model_tpm_limit: z.custom<string | undefined>(),
  model_rpm_limit: z.custom<string | undefined>(),
  default_estimated_output_tokens: z
    .custom<number | string | null | undefined>()
    .refine(estimateChecks.positive.isValid, estimateChecks.positive.message),
  default_estimated_output_tokens_per_model: z
    .custom<string | undefined>()
    .refine(estimateChecks.perModel.isValid, estimateChecks.perModel.message),
  guardrails: z.custom<string[] | undefined>(),
  disable_global_guardrails: z.custom<boolean | undefined>(),
  policies: z.custom<string[] | undefined>(),
  tags: z.custom<string[] | undefined>(),
  prompts: z.custom<string[] | undefined>(),
  access_group_ids: z.custom<string[] | undefined>(),
  allowed_passthrough_routes: z.custom<string[] | undefined>(),
  vector_stores: z.custom<string[] | undefined>(),
  mcp_servers_and_groups: z.custom<McpServersAndGroups | undefined>(),
  mcp_tool_permissions: z.custom<Record<string, string[]> | undefined>(),
  agents_and_groups: z.custom<AgentsAndGroups | undefined>(),
  organization_id: z.custom<string | null | undefined>(),
  team_id: z.custom<string | null | undefined>(),
  logging_settings: z.custom<unknown[] | undefined>(),
  metadata: z.custom<string | undefined>(),
  duration: z.custom<string | null | undefined>(),
  token: z.custom<string | undefined>(),
  disabled_callbacks: z.custom<string[] | undefined>(),
  auto_rotate: z.custom<boolean | undefined>(),
  rotation_interval: z.custom<string | undefined>(),
});

export interface MountedFieldGates {
  canViewPolicies: boolean;
  canViewPrompts: boolean;
}

export const toSubmittedValues = (
  values: KeyEditFormValues,
  { canViewPolicies, canViewPrompts }: MountedFieldGates,
): Record<string, unknown> => ({
  key_alias: values.key_alias,
  models: values.models,
  allowed_routes: values.allowed_routes,
  max_budget: values.max_budget,
  budget_duration: values.budget_duration,
  tpm_limit: values.tpm_limit,
  tpm_limit_type: values.tpm_limit_type,
  rpm_limit: values.rpm_limit,
  rpm_limit_type: values.rpm_limit_type,
  throttle_on_budget_exceeded: values.throttle_on_budget_exceeded,
  enable_prompt_caching: values.enable_prompt_caching,
  max_parallel_requests: values.max_parallel_requests,
  model_tpm_limit: values.model_tpm_limit,
  model_rpm_limit: values.model_rpm_limit,
  default_estimated_output_tokens: values.default_estimated_output_tokens,
  default_estimated_output_tokens_per_model: values.default_estimated_output_tokens_per_model,
  guardrails: values.guardrails,
  disable_global_guardrails: values.disable_global_guardrails,
  ...(canViewPolicies ? { policies: values.policies } : {}),
  tags: values.tags,
  ...(canViewPrompts ? { prompts: values.prompts } : {}),
  access_group_ids: values.access_group_ids,
  allowed_passthrough_routes: values.allowed_passthrough_routes,
  vector_stores: values.vector_stores,
  mcp_servers_and_groups: values.mcp_servers_and_groups,
  mcp_tool_permissions: values.mcp_tool_permissions,
  agents_and_groups: values.agents_and_groups,
  organization_id: values.organization_id,
  team_id: values.team_id,
  logging_settings: values.logging_settings,
  metadata: values.metadata,
  duration: values.duration,
  token: values.token,
  disabled_callbacks: values.disabled_callbacks,
  auto_rotate: values.auto_rotate,
  rotation_interval: values.rotation_interval,
});
