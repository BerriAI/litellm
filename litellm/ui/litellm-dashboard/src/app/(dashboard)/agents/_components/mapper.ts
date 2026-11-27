import type { Agent } from "@/components/agents/types";
import type { AgentCreateInfo } from "@/components/networking";
import { AGENT_FORM_CONFIG, buildAgentDataFromForm, buildMcpObjectPermission, getDefaultFormValues } from "./agent_config";
import type { DiscoveredAgentCardSelection } from "./agent_card_discovery";
import { overlayDiscoveredCardParams } from "./agent_discovery_utils";
import { unmountedA2AFieldNames } from "./agent_form_fields";
import { withAgentIdentity } from "./agent_identity";
import { omitFieldValues, type AgentFormValues, type AgentRequestPayload } from "./AgentFormKit";
import { buildDynamicAgentData, unmountedDynamicFieldNames } from "./dynamic_agent_form_fields";

export const CUSTOM_AGENT_TYPE = "custom";

const SHARED_INITIAL_VALUES: AgentFormValues = {
  allowed_mcp_servers_and_groups: { servers: [], accessGroups: [] },
  mcp_tool_permissions: {},
  entitlement_models: [],
  entitlement_agents: [],
  access_group_ids: [],
  guardrails: [],
};

export const buildInitialValues = (agentType: string): AgentFormValues =>
  agentType === "a2a" ? { ...getDefaultFormValues(), ...SHARED_INITIAL_VALUES } : { ...SHARED_INITIAL_VALUES };

const buildAgentData = (
  values: AgentFormValues,
  agentType: string,
  selectedAgentTypeInfo: AgentCreateInfo | undefined,
  appliedDiscoveredSelection: DiscoveredAgentCardSelection | null,
): AgentRequestPayload | null => {
  if (agentType === CUSTOM_AGENT_TYPE) {
    if (values.identity_provider === "microsoft_entra") return { agent_name: values.agent_name };
    return {
      agent_name: values.agent_name,
      agent_card_params: {
        protocolVersion: "1.0",
        name: values.agent_name,
        description: values.description || "",
        url: "",
        version: "1.0.0",
        defaultInputModes: ["text"],
        defaultOutputModes: ["text"],
        capabilities: { streaming: false },
        skills: [],
      },
    };
  }

  if (agentType === "a2a") {
    return overlayDiscoveredCardParams(buildAgentDataFromForm(values), appliedDiscoveredSelection?.selected_card);
  }

  if (!selectedAgentTypeInfo) return null;

  if (!selectedAgentTypeInfo.use_a2a_form_fields) {
    return overlayDiscoveredCardParams(
      buildDynamicAgentData(values, selectedAgentTypeInfo),
      appliedDiscoveredSelection?.selected_card,
    );
  }

  const agentData: AgentRequestPayload = buildAgentDataFromForm(values);
  if (selectedAgentTypeInfo.litellm_params_template) {
    agentData.litellm_params = {
      ...agentData.litellm_params,
      ...selectedAgentTypeInfo.litellm_params_template,
    };
  }
  const credentialParams = Object.fromEntries(
    selectedAgentTypeInfo.credential_fields
      .filter((field) => values[field.key] && field.include_in_litellm_params !== false)
      .map((field) => [field.key, values[field.key]]),
  );
  if (Object.keys(credentialParams).length > 0) {
    agentData.litellm_params = { ...agentData.litellm_params, ...credentialParams };
  }
  return overlayDiscoveredCardParams(agentData, appliedDiscoveredSelection?.selected_card);
};

interface CreateAgentOptions {
  requireTraceIdInbound: boolean;
  requireTraceIdOutbound: boolean;
  maxIterations: number | null;
  maxBudgetPerSession: number | null;
}

export const mapAgentCreate = (
  values: AgentFormValues,
  { agentType, selectedAgentTypeInfo, appliedDiscoveredSelection }: {
    agentType: string;
    selectedAgentTypeInfo: AgentCreateInfo | undefined;
    appliedDiscoveredSelection: DiscoveredAgentCardSelection | null;
  },
  { requireTraceIdInbound, requireTraceIdOutbound, maxIterations, maxBudgetPerSession }: CreateAgentOptions,
) => {
  const built = buildAgentData(values, agentType, selectedAgentTypeInfo, appliedDiscoveredSelection);
  if (!built) return null;
  const agentData = withAgentIdentity(built, values);

  const mcpServersAndGroups = values.allowed_mcp_servers_and_groups ?? {};
  const toolPermissions = values.mcp_tool_permissions ?? {};
  const entitlementModels = values.entitlement_models ?? [];
  const entitlementAgents = values.entitlement_agents ?? [];
  const objectPermission: Record<string, unknown> = {
    ...(mcpServersAndGroups.servers?.length ? { mcp_servers: mcpServersAndGroups.servers } : {}),
    ...(mcpServersAndGroups.accessGroups?.length ? { mcp_access_groups: mcpServersAndGroups.accessGroups } : {}),
    ...(mcpServersAndGroups.toolsets?.length ? { mcp_toolsets: mcpServersAndGroups.toolsets } : {}),
    ...(Object.keys(toolPermissions).length ? { mcp_tool_permissions: toolPermissions } : {}),
    ...(entitlementModels.length ? { models: entitlementModels } : {}),
    ...(entitlementAgents.length ? { agents: entitlementAgents } : {}),
  };
  if (Object.keys(objectPermission).length > 0) {
    agentData.object_permission = objectPermission;
  }
  if (values.access_group_ids?.length) {
    agentData.access_group_ids = values.access_group_ids;
  }

  if (requireTraceIdInbound || requireTraceIdOutbound) {
    agentData.litellm_params = {
      ...agentData.litellm_params,
      ...(requireTraceIdInbound ? { require_trace_id_on_calls_to_agent: true } : {}),
      ...(requireTraceIdOutbound ? { require_trace_id_on_calls_by_agent: true } : {}),
      ...(requireTraceIdOutbound && maxIterations ? { max_iterations: maxIterations } : {}),
      ...(requireTraceIdOutbound && maxBudgetPerSession ? { max_budget_per_session: maxBudgetPerSession } : {}),
    };
  }

  const selectedGuardrails = values.guardrails ?? [];
  if (selectedGuardrails.length > 0) {
    agentData.litellm_params = { ...agentData.litellm_params, guardrails: selectedGuardrails };
  }

  const selectedTeamId = values.team_id || null;
  if (selectedTeamId) {
    agentData.team_id = selectedTeamId;
  }
  return { agentData, selectedTeamId };
};

export const mapAgentEdit = (
  submitted: AgentFormValues,
  agent: Agent,
  { dynamicInfo, mountedPanels, appliedDiscoveredSelection, isDirty }: {
    dynamicInfo: AgentCreateInfo | undefined;
    mountedPanels: readonly string[];
    appliedDiscoveredSelection: DiscoveredAgentCardSelection | null;
    isDirty: (name: string) => boolean;
  },
): AgentRequestPayload => {
  const values = omitFieldValues(
    submitted,
    dynamicInfo
      ? unmountedDynamicFieldNames(mountedPanels)
      : unmountedA2AFieldNames(mountedPanels),
  );

  const built: AgentRequestPayload = dynamicInfo
    ? { ...buildDynamicAgentData(values, dynamicInfo, agent), agent_name: values.agent_name }
    : buildAgentDataFromForm(values, agent);

  const updateData = appliedDiscoveredSelection
    ? overlayDiscoveredCardParams(built, appliedDiscoveredSelection.selected_card)
    : built;
  const cardEdited =
    Boolean(appliedDiscoveredSelection) ||
    [AGENT_FORM_CONFIG.basic, AGENT_FORM_CONFIG.skills, AGENT_FORM_CONFIG.capabilities, AGENT_FORM_CONFIG.optional]
      .flatMap((section) => section.fields)
      .some((field) => isDirty(field.name));

  return {
    ...withAgentIdentity(updateData, values, agent, cardEdited),
    object_permission: buildMcpObjectPermission(values),
    access_group_ids: values.access_group_ids ?? [],
  };
};
