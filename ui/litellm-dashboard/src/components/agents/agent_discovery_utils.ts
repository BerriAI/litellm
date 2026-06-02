import {
  AgentCreateInfo,
  DiscoveredAgentCard,
  DiscoveryMode,
} from "../networking";

export interface DiscoveryRequestPlan {
  url: string;
  discovery_mode: DiscoveryMode;
  params?: Record<string, any>;
  display_url?: string;
}

export const skillId = (skill: any, idx: number): string =>
  skill?.id ?? skill?.name ?? `skill-${idx}`;

export const ALLOWED_CAPABILITY_KEYS = ["streaming"] as const;

export const filterCapabilitiesForUI = (
  capabilities: Record<string, any> | undefined,
): Record<string, boolean> => {
  if (!capabilities) return {};
  return ALLOWED_CAPABILITY_KEYS.reduce<Record<string, boolean>>((acc, key) => {
    if (key in capabilities) acc[key] = Boolean(capabilities[key]);
    return acc;
  }, {});
};

/**
 * After fetching the full upstream card, pre-select only skills and
 * capabilities that already exist on the agent record in the DB.
 */
export const selectionsFromSavedAgentCard = (
  upstreamCard: DiscoveredAgentCard,
  savedCard: DiscoveredAgentCard | undefined | null,
): {
  editedName: string;
  editedDescription: string;
  selectedSkillIds: Set<string>;
  selectedCapabilities: Record<string, boolean>;
} => {
  const upstreamSkills = upstreamCard.skills ?? [];
  const savedSkills = savedCard?.skills ?? [];

  const savedSkillIds = new Set(
    savedSkills.map((s) => s?.id).filter(Boolean) as string[],
  );
  const savedSkillNames = new Set(
    savedSkills.map((s) => s?.name).filter(Boolean) as string[],
  );

  const selectedSkillIds = new Set<string>();
  upstreamSkills.forEach((skill, idx) => {
    const id = skillId(skill, idx);
    const matchesById = skill.id && savedSkillIds.has(skill.id);
    const matchesByName = skill.name && savedSkillNames.has(skill.name);
    if (matchesById || matchesByName) {
      selectedSkillIds.add(id);
    }
  });

  const selectedCapabilities = filterCapabilitiesForUI(upstreamCard.capabilities);
  if (savedCard?.capabilities) {
    for (const key of ALLOWED_CAPABILITY_KEYS) {
      if (key in savedCard.capabilities) {
        selectedCapabilities[key] = Boolean(savedCard.capabilities[key]);
      }
    }
  }

  return {
    editedName: savedCard?.name ?? upstreamCard.name ?? "",
    editedDescription: savedCard?.description ?? upstreamCard.description ?? "",
    selectedSkillIds,
    selectedCapabilities,
  };
};

/** Default for create flow: select everything the upstream advertises. */
export const selectionsFromUpstreamCard = (
  upstreamCard: DiscoveredAgentCard,
): {
  editedName: string;
  editedDescription: string;
  selectedSkillIds: Set<string>;
  selectedCapabilities: Record<string, boolean>;
} => {
  const upstreamSkills = upstreamCard.skills ?? [];
  return {
    editedName: upstreamCard.name ?? "",
    editedDescription: upstreamCard.description ?? "",
    selectedSkillIds: new Set(upstreamSkills.map((s, i) => skillId(s, i))),
    selectedCapabilities: filterCapabilitiesForUI(upstreamCard.capabilities),
  };
};

/**
 * Overlay the admin's discovery selections onto the ``agent_card_params``
 * built from the form. Dynamic agent forms (e.g. LangGraph) don't register
 * Form.Items for name / description / skills / capabilities, so AntD's
 * setFieldsValue silently drops those keys and the values never make it back
 * through buildAgentData — we re-apply them here from the selection.
 */
export const overlayDiscoveredCardParams = (
  agentData: Record<string, any>,
  discovered: DiscoveredAgentCard | null | undefined,
): Record<string, any> => {
  if (!discovered) return agentData;
  return {
    ...agentData,
    agent_card_params: {
      ...agentData.agent_card_params,
      name: discovered.name ?? agentData.agent_card_params?.name,
      description:
        discovered.description ?? agentData.agent_card_params?.description,
      ...(Array.isArray(discovered.skills) && {
        skills: discovered.skills,
      }),
      ...(discovered.capabilities && {
        capabilities: discovered.capabilities,
      }),
      ...(Array.isArray(discovered.defaultInputModes) &&
        discovered.defaultInputModes.length > 0 && {
          defaultInputModes: discovered.defaultInputModes,
        }),
      ...(Array.isArray(discovered.defaultOutputModes) &&
        discovered.defaultOutputModes.length > 0 && {
          defaultOutputModes: discovered.defaultOutputModes,
        }),
      ...(discovered.provider && { provider: discovered.provider }),
      ...(discovered.iconUrl && { iconUrl: discovered.iconUrl }),
      ...(discovered.documentationUrl && {
        documentationUrl: discovered.documentationUrl,
      }),
    },
  };
};

export const buildDiscoveryRequest = (
  agentType: string,
  values: Record<string, any>,
  selectedAgentTypeInfo?: AgentCreateInfo,
): DiscoveryRequestPlan | undefined => {
  const trim = (v: unknown) => (v ?? "").toString().trim();
  const stripTrailingSlash = (s: string) => s.replace(/\/+$/, "");

  if (agentType === "langgraph") {
    const base = stripTrailingSlash(trim(values.api_base));
    const assistantId = trim(values.assistant_id);
    if (!base || !assistantId) return undefined;
    const query = `?assistant_id=${encodeURIComponent(assistantId)}`;
    return {
      url: base,
      discovery_mode: "langgraph_platform",
      params: { assistant_id: assistantId },
      display_url: `${base}/.well-known/agent-card.json${query}`,
    };
  }

  if (agentType === "a2a" || selectedAgentTypeInfo?.use_a2a_form_fields) {
    const base = stripTrailingSlash(trim(values.url));
    if (!base) return undefined;
    return {
      url: base,
      discovery_mode: "well_known_fallback",
      display_url: `${base}/.well-known/agent-card.json`,
    };
  }

  // Non-A2A agent runtimes (Azure AI Foundry, Bedrock AgentCore, Vertex,
  // etc.) don't expose well-known agent cards on their credential URLs, so
  // we deliberately don't auto-fire discovery for them. The
  // ``AgentCardDiscovery`` widget falls back to a manual URL input the admin
  // can use as an escape hatch.
  return undefined;
};
