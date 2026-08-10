"use client";
import { keyKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useTags } from "@/app/(dashboard)/hooks/tags/useTags";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useQueryClient } from "@tanstack/react-query";
import { Accordion, AccordionBody, AccordionHeader, Button, Col, Grid, Text, TextInput, Title } from "@tremor/react";
import { Button as Button2, Form, Input, Modal, Radio, Select, Switch, Tag, Tooltip, Typography } from "antd";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { rolesWithWriteAccess } from "../../utils/roles";
import AgentSelector from "../agent_management/AgentSelector";
import { mapDisplayToInternalNames } from "../callback_info_helpers";
import AccessGroupSelector from "../common_components/AccessGroupSelector";
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import SchemaFormFields from "../common_components/check_openapi_schema";
import KeyLifecycleSettings from "../common_components/KeyLifecycleSettings";
import ModelAliasManager from "../common_components/ModelAliasManager";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";
import PremiumLoggingSettings from "../common_components/PremiumLoggingSettings";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "../common_components/RouterSettingsAccordion";
import TeamDropdown from "../common_components/team_dropdown";
import OrganizationDropdown from "../common_components/OrganizationDropdown";
import ProjectDropdown from "../common_components/ProjectDropdown";
import { CreateUserButton } from "../CreateUserButton";
import { BudgetFallbacksEditor } from "../key_team_helpers/BudgetFallbacksEditor";
import { BudgetWindowEntry, BudgetWindowsEditor } from "../key_team_helpers/BudgetWindowsEditor";
import { TagRateLimitEditor, TagRateLimitEntry, tagRowsToLimits } from "../key_team_helpers/TagRateLimitEditor";
import {
  excludeProxyWideSentinel,
  getModelDisplayName,
  hasAllModelsSentinel,
} from "../key_team_helpers/fetch_available_models_team_key";
import { Team } from "../key_team_helpers/key_list";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import { NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import NotificationsManager from "../molecules/notifications_manager";
import {
  getAgentsList,
  getGuardrailsList,
  getPoliciesList,
  getPossibleUserRoles,
  getPromptsList,
  keyCreateCall,
  keyCreateServiceAccountCall,
  modelAvailableCall,
  proxyBaseUrl,
  userFilterUICall,
} from "../networking";
import CreatedKeyDisplay from "../shared/CreatedKeyDisplay";
import NumericalInput from "../shared/numerical_input";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import { simplifyKeyGenerateError } from "./utils";

const { Option } = Select;

/**
 * Interface for pre-filling the create key form from URL parameters
 */
export interface CreateKeyPrefillData {
  owned_by?: "you" | "service_account" | "another_user";
  team_id?: string;
  key_alias?: string;
  models?: string[];
  key_type?: "default" | "llm_api" | "management";
}

interface CreateKeyProps {
  team: Team | null;
  data: any[] | null;
  teams: Team[] | null;
  addKey: (data: any) => void;
  autoOpenCreate?: boolean;
  prefillData?: CreateKeyPrefillData;
  buttonLabel?: string;
}

interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

interface UserOption {
  label: string;
  value: string;
  user: User;
}

export const fetchTeamModels = async (
  userID: string,
  userRole: string,
  accessToken: string,
  teamID: string | null,
): Promise<string[]> => {
  try {
    if (userID === null || userRole === null) {
      return [];
    }

    if (accessToken !== null) {
      const model_available = await modelAvailableCall(accessToken, userID, userRole, true, teamID, true);
      let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
      return available_model_names;
    }
    return [];
  } catch (error) {
    console.error("Error fetching user models:", error);
    return [];
  }
};

export const fetchUserModels = async (
  userID: string,
  userRole: string,
  accessToken: string,
  setUserModels: (models: string[]) => void,
) => {
  try {
    if (userID === null || userRole === null) {
      return;
    }

    if (accessToken !== null) {
      const model_available = await modelAvailableCall(accessToken, userID, userRole);
      let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
      setUserModels(available_model_names);
    }
  } catch (error) {
    console.error("Error fetching user models:", error);
  }
};

/**
 * ─────────────────────────────────────────────────────────────────────────
 * @deprecated
 * This component is being DEPRECATED in favor of src/app/(dashboard)/virtual-keys/components/CreateKey.tsx
 * Please contribute to the new refactor.
 * ─────────────────────────────────────────────────────────────────────────
 */
const CreateKey: React.FC<CreateKeyProps> = ({
  team,
  teams,
  data,
  addKey,
  autoOpenCreate,
  prefillData,
  buttonLabel,
}) => {
  const { t } = useTranslation("gateway");
  const { accessToken, userId: userID, userRole, premiumUser } = useAuthorized();
  const canEditGuardrails = premiumUser || (userRole != null && rolesWithWriteAccess.includes(userRole));
  const { data: organizations, isLoading: isOrganizationsLoading } = useOrganizations();
  const { data: projects, isLoading: isProjectsLoading } = useProjects();
  const { data: uiSettingsData } = useUISettings();
  const { data: tagsData } = useTags();
  const enableProjectsUI = Boolean(uiSettingsData?.values?.enable_projects_ui);
  const disableCustomApiKeys = Boolean(uiSettingsData?.values?.disable_custom_api_keys);
  const tagOptions = tagsData ? Object.values(tagsData).map((tag) => ({ value: tag.name, label: tag.name })) : [];
  const queryClient = useQueryClient();
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiKey, setApiKey] = useState(null);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [keyOwner, setKeyOwner] = useState("you");
  const [hasPrefilled, setHasPrefilled] = useState(false);
  const [pendingPrefillModels, setPendingPrefillModels] = useState<string[] | null>(null);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [policiesList, setPoliciesList] = useState<string[]>([]);
  const [promptsList, setPromptsList] = useState<string[]>([]);
  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [selectedCreateKeyTeam, setSelectedCreateKeyTeam] = useState<Team | null>(team);
  const [selectedOrganizationId, setSelectedOrganizationId] = useState<string | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [isCreateUserModalVisible, setIsCreateUserModalVisible] = useState(false);
  const [possibleUIRoles, setPossibleUIRoles] = useState<Record<string, Record<string, string>>>({});
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [userSearchLoading, setUserSearchLoading] = useState<boolean>(false);
  const [disabledCallbacks, setDisabledCallbacks] = useState<string[]>([]);
  const [keyType, setKeyType] = useState<string>("llm_api");
  const [modelAliases, setModelAliases] = useState<{ [key: string]: string }>({});
  const [autoRotationEnabled, setAutoRotationEnabled] = useState<boolean>(false);
  const [rotationInterval, setRotationInterval] = useState<string>("30d");
  const [routerSettings, setRouterSettings] = useState<RouterSettingsAccordionValue | null>(null);
  const [budgetLimits, setBudgetLimits] = useState<BudgetWindowEntry[]>([]);
  const [tagRateLimits, setTagRateLimits] = useState<TagRateLimitEntry[]>([]);
  const [budgetFallbacks, setBudgetFallbacks] = useState<Record<string, string[]>>({});
  const [budgetFallbacksKey, setBudgetFallbacksKey] = useState<number>(0);
  const [routerSettingsKey, setRouterSettingsKey] = useState<number>(0);
  const [agentsList, setAgentsList] = useState<{ agent_id: string; agent_name: string }[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const selectedModels: string[] = Form.useWatch("models", form) ?? [];
  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setDisabledCallbacks([]);
    setKeyType("llm_api");
    setModelAliases({});
    setAutoRotationEnabled(false);
    setRotationInterval("30d");
    setRouterSettings(null);
    setRouterSettingsKey((prev) => prev + 1);
    setSelectedAgentId(null);
    setSelectedOrganizationId(null);
    setSelectedProjectId(null);
    setBudgetLimits([]);
    setTagRateLimits([]);
    setBudgetFallbacks({});
    setBudgetFallbacksKey((k) => k + 1);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiKey(null);
    setSelectedCreateKeyTeam(null);
    form.resetFields();
    setLoggingSettings([]);
    setDisabledCallbacks([]);
    setKeyType("llm_api");
    setModelAliases({});
    setAutoRotationEnabled(false);
    setRotationInterval("30d");
    setRouterSettings(null);
    setRouterSettingsKey((prev) => prev + 1);
    setSelectedAgentId(null);
    setSelectedOrganizationId(null);
    setSelectedProjectId(null);
    setBudgetLimits([]);
    setTagRateLimits([]);
    setBudgetFallbacks({});
    setBudgetFallbacksKey((k) => k + 1);
  };

  useEffect(() => {
    if (userID && userRole && accessToken) {
      fetchUserModels(userID, userRole, accessToken, setUserModels);
    }
  }, [accessToken, userID, userRole]);

  useEffect(() => {
    if (accessToken) {
      getAgentsList(accessToken)
        .then((res) => setAgentsList(res?.agents || []))
        .catch(() => setAgentsList([]));
    }
  }, [accessToken]);

  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    const fetchPolicies = async () => {
      try {
        const response = await getPoliciesList(accessToken);
        const policyNames = response.policies.map((p: { policy_name: string }) => p.policy_name);
        setPoliciesList(policyNames);
      } catch (error) {
        console.error("Failed to fetch policies:", error);
      }
    };

    const fetchPrompts = async () => {
      try {
        const response = await getPromptsList(accessToken);
        setPromptsList(response.prompts.map((prompt) => prompt.prompt_id));
      } catch (error) {
        console.error("Failed to fetch prompts:", error);
      }
    };

    fetchGuardrails();
    fetchPolicies();
    fetchPrompts();
  }, [accessToken]);

  // Fetch possible user roles when component mounts
  useEffect(() => {
    const fetchPossibleRoles = async () => {
      try {
        if (accessToken) {
          // Check if roles are cached in session storage
          const cachedRoles = sessionStorage.getItem("possibleUserRoles");
          if (cachedRoles) {
            setPossibleUIRoles(JSON.parse(cachedRoles));
          } else {
            const availableUserRoles = await getPossibleUserRoles(accessToken);
            sessionStorage.setItem("possibleUserRoles", JSON.stringify(availableUserRoles));
            setPossibleUIRoles(availableUserRoles);
          }
        }
      } catch (error) {
        console.error("Error fetching possible user roles:", error);
      }
    };

    fetchPossibleRoles();
  }, [accessToken]);

  // Auto-open modal and prefill form from URL params (deep link).
  // Guarded by write access so we don't open for read-only users.
  useEffect(() => {
    if (autoOpenCreate && !hasPrefilled && teams && userRole && rolesWithWriteAccess.includes(userRole)) {
      // Open the modal
      setIsModalVisible(true);
      setHasPrefilled(true);

      // Apply prefill data if provided
      if (prefillData) {
        // Set key owner (owned_by) - validate that "another_user" is only allowed for Admin
        if (prefillData.owned_by) {
          if (prefillData.owned_by === "another_user" && userRole !== "Admin") {
            // Ignore invalid owned_by for non-admin users, fall back to default
            setKeyOwner("you");
          } else {
            setKeyOwner(prefillData.owned_by);
          }
        }

        // Set team - find the team by ID and set it (only if team exists in user's teams)
        if (prefillData.team_id) {
          const selectedTeam = teams?.find((t) => t.team_id === prefillData.team_id) || null;
          if (selectedTeam) {
            setSelectedCreateKeyTeam(selectedTeam);
            form.setFieldsValue({ team_id: prefillData.team_id });
          }
          // Silently ignore invalid team_id - don't prefill with a team user doesn't have access to
        }

        // Set key alias
        if (prefillData.key_alias) {
          form.setFieldsValue({ key_alias: prefillData.key_alias });
        }

        // Defer model selection until we load the allowed model list.
        if (prefillData.models && prefillData.models.length > 0) {
          setPendingPrefillModels(prefillData.models);
        }

        // Set key type
        if (prefillData.key_type) {
          setKeyType(prefillData.key_type);
          form.setFieldsValue({ key_type: prefillData.key_type });
        }
      }
    }
  }, [autoOpenCreate, prefillData, teams, hasPrefilled, form, userRole]);

  // Check if team selection is required
  const isTeamSelectionRequired = modelsToPick.includes("no-default-models");
  const isFormDisabled = isTeamSelectionRequired && !selectedCreateKeyTeam;

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      const newKeyAlias = formValues?.key_alias ?? "";
      const newKeyTeamId = formValues?.team_id ?? null;

      const existingKeyAliases = data?.filter((k) => k.team_id === newKeyTeamId).map((k) => k.key_alias) ?? [];

      if (existingKeyAliases.includes(newKeyAlias)) {
        throw new Error(t("virtualKeys.createKey.aliasExists", { alias: newKeyAlias, teamId: newKeyTeamId }));
      }

      NotificationsManager.info(t("virtualKeys.createKey.sendingRequest"));
      setIsModalVisible(true);

      if (keyOwner === "you") {
        formValues.user_id = userID;
      } else if (keyOwner === "agent") {
        if (!selectedAgentId) {
          NotificationsManager.fromBackend(t("virtualKeys.createKey.selectAgentError"));
          return;
        }
        formValues.agent_id = selectedAgentId;
      }

      // Handle metadata for all key types
      let metadata: Record<string, any> = {};
      try {
        metadata = JSON.parse(formValues.metadata || "{}");
      } catch (error) {
        console.error("Error parsing metadata:", error);
      }

      // If it's a service account, add the service_account_id to the metadata
      if (keyOwner === "service_account") {
        metadata["service_account_id"] = formValues.key_alias;
      }

      // Add logging settings to the metadata
      if (loggingSettings.length > 0) {
        metadata = {
          ...metadata,
          logging: loggingSettings.filter((config) => config.callback_name),
        };
      }

      // Add disabled callbacks to the metadata
      if (disabledCallbacks.length > 0) {
        // Map display names to internal callback values
        const mappedDisabledCallbacks = mapDisplayToInternalNames(disabledCallbacks);
        metadata = {
          ...metadata,
          litellm_disabled_callbacks: mappedDisabledCallbacks,
        };
      }

      // Add auto-rotation settings as top-level fields
      if (autoRotationEnabled) {
        formValues.auto_rotate = true;
        formValues.rotation_interval = rotationInterval;
      }

      // Handle duration field for key expiry - convert empty string to null
      if (!formValues.duration || formValues.duration.trim() === "") {
        formValues.duration = null;
      }

      // Update the formValues with the final metadata
      formValues.metadata = JSON.stringify(metadata);

      // disable_global_guardrails is premium-gated server-side; only send it when enabled
      // so non-premium key creation isn't blocked by that gate.
      if (!formValues.disable_global_guardrails) {
        delete formValues.disable_global_guardrails;
      }

      // Transform allowed_vector_store_ids and allowed_mcp_servers_and_groups into object_permission format
      if (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) {
        formValues.object_permission = {
          vector_stores: formValues.allowed_vector_store_ids,
        };
        // Remove the original field as it's now part of object_permission
        delete formValues.allowed_vector_store_ids;
      }

      // Transform allowed_mcp_servers_and_groups into object_permission format
      if (
        formValues.allowed_mcp_servers_and_groups &&
        (formValues.allowed_mcp_servers_and_groups.servers?.length > 0 ||
          formValues.allowed_mcp_servers_and_groups.accessGroups?.length > 0 ||
          formValues.allowed_mcp_servers_and_groups.toolsets?.length > 0)
      ) {
        if (!formValues.object_permission) {
          formValues.object_permission = {};
        }
        const { servers, accessGroups, toolsets } = formValues.allowed_mcp_servers_and_groups;
        if (servers && servers.length > 0) {
          formValues.object_permission.mcp_servers = servers;
        }
        if (accessGroups && accessGroups.length > 0) {
          formValues.object_permission.mcp_access_groups = accessGroups;
        }
        if (toolsets && toolsets.length > 0) {
          formValues.object_permission.mcp_toolsets = toolsets;
        }
        // Remove the original field as it's now part of object_permission
        delete formValues.allowed_mcp_servers_and_groups;
      }

      // Add MCP tool permissions to object_permission
      const mcpToolPermissions = formValues.mcp_tool_permissions || {};
      if (Object.keys(mcpToolPermissions).length > 0) {
        if (!formValues.object_permission) {
          formValues.object_permission = {};
        }
        formValues.object_permission.mcp_tool_permissions = mcpToolPermissions;
      }
      delete formValues.mcp_tool_permissions;

      // Transform allowed_mcp_access_groups into object_permission format
      if (formValues.allowed_mcp_access_groups && formValues.allowed_mcp_access_groups.length > 0) {
        if (!formValues.object_permission) {
          formValues.object_permission = {};
        }
        formValues.object_permission.mcp_access_groups = formValues.allowed_mcp_access_groups;
        // Remove the original field as it's now part of object_permission
        delete formValues.allowed_mcp_access_groups;
      }

      // Transform allowed_agents_and_groups into object_permission format
      if (
        formValues.allowed_agents_and_groups &&
        (formValues.allowed_agents_and_groups.agents?.length > 0 ||
          formValues.allowed_agents_and_groups.accessGroups?.length > 0)
      ) {
        if (!formValues.object_permission) {
          formValues.object_permission = {};
        }
        const { agents, accessGroups } = formValues.allowed_agents_and_groups;
        if (agents && agents.length > 0) {
          formValues.object_permission.agents = agents;
        }
        if (accessGroups && accessGroups.length > 0) {
          formValues.object_permission.agent_access_groups = accessGroups;
        }
        // Remove the original field as it's now part of object_permission
        delete formValues.allowed_agents_and_groups;
      }

      // Add model_aliases if any are defined
      if (Object.keys(modelAliases).length > 0) {
        formValues.aliases = JSON.stringify(modelAliases);
      }

      // Add router_settings if any are defined
      if (routerSettings?.router_settings) {
        // Only include router_settings if it has at least one non-null value
        const hasValues = Object.values(routerSettings.router_settings).some(
          (value) => value !== null && value !== undefined && value !== "",
        );
        if (hasValues) {
          formValues.router_settings = routerSettings.router_settings;
        }
      }

      // Add multi-window budget limits (filter out incomplete entries)
      const validWindows = budgetLimits.filter(
        (w) => w.budget_duration && w.max_budget !== null && w.max_budget !== undefined,
      );
      if (validWindows.length > 0) {
        formValues.budget_limits = validWindows;
      }

      // Add per-tag rate limits (only when at least one row is configured)
      const { tag_rpm_limit } = tagRowsToLimits(tagRateLimits);
      if (Object.keys(tag_rpm_limit).length > 0) {
        formValues.tag_rpm_limit = tag_rpm_limit;
      }

      if (Object.keys(budgetFallbacks).length > 0) {
        formValues.budget_fallbacks = budgetFallbacks;
      }

      let response;
      if (keyOwner === "service_account") {
        response = await keyCreateServiceAccountCall(accessToken, formValues);
      } else {
        response = await keyCreateCall(accessToken, userID, formValues);
      }

      // Add the data to the state in the parent component
      // Also directly update the keys list in VirtualKeysTable without an API call
      addKey(response);

      // Invalidate and refetch all keys list queries to update the table
      // This will trigger a refetch of all key list queries regardless of pagination
      queryClient.invalidateQueries({ queryKey: keyKeys.lists() });

      setApiKey(response["key"]);
      NotificationsManager.success(t("virtualKeys.createKey.createdSuccess"));
      form.resetFields();
      setBudgetLimits([]);
      setTagRateLimits([]);
      setBudgetFallbacks({});
      setBudgetFallbacksKey((k) => k + 1);
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      const simplifiedError = simplifyKeyGenerateError(error);
      NotificationsManager.fromBackend(simplifiedError);
    }
  };

  // Fetch available models when team or auth changes.
  // Note: Model prefill from URL params is handled by the useEffect below, which
  // watches for pendingPrefillModels + modelsToPick to both be populated.
  useEffect(() => {
    if (selectedProjectId) {
      // When a project is selected, use the project's models
      const project = projects?.find((p) => p.project_id === selectedProjectId);
      const projectModels = project?.models ?? [];
      setModelsToPick(projectModels);
      form.setFieldValue("models", []);
      return;
    }
    if (userID && userRole && accessToken) {
      fetchTeamModels(userID, userRole, accessToken, selectedCreateKeyTeam?.team_id ?? null).then((models) => {
        const allModels = excludeProxyWideSentinel(
          Array.from(new Set([...(selectedCreateKeyTeam?.models ?? []), ...models])),
        );
        setModelsToPick(allModels);
      });
    }
    // Only clear models if we don't have pending prefill models
    if (!pendingPrefillModels) {
      form.setFieldValue("models", []);
    }
    // Clear MCP server selection when team changes (available servers may differ)
    form.setFieldValue("allowed_mcp_servers_and_groups", { servers: [], accessGroups: [] });
  }, [selectedCreateKeyTeam, selectedProjectId, accessToken, userID, userRole, form]);

  // Apply deferred model prefill once the available model list arrives.
  // This handles timing where prefill data arrives before or after models are fetched.
  useEffect(() => {
    if (!pendingPrefillModels || pendingPrefillModels.length === 0) {
      return;
    }
    if (!modelsToPick || modelsToPick.length === 0) {
      return;
    }

    const validModels = pendingPrefillModels.filter((model) => modelsToPick.includes(model));
    if (validModels.length > 0) {
      form.setFieldsValue({ models: validModels });
    }
    setPendingPrefillModels(null);
  }, [pendingPrefillModels, modelsToPick, form]);

  // Sync team when project is selected but teams loaded later (race condition)
  useEffect(() => {
    if (!selectedProjectId || !teams) return;
    const project = projects?.find((p) => p.project_id === selectedProjectId);
    if (!project?.team_id) return;
    // If team is already set correctly, skip
    if (selectedCreateKeyTeam?.team_id === project.team_id) return;
    const projectTeam = teams.find((t) => t.team_id === project.team_id) || null;
    if (projectTeam) {
      setSelectedCreateKeyTeam(projectTeam);
      form.setFieldValue("team_id", projectTeam.team_id);
    }
  }, [teams, selectedProjectId, projects]);

  // Add a callback function to handle user creation
  const handleUserCreated = (userId: string) => {
    form.setFieldsValue({ user_id: userId });
    setIsCreateUserModalVisible(false);
  };

  const fetchUsers = async (searchText: string): Promise<void> => {
    if (!searchText) {
      setUserOptions([]);
      return;
    }

    setUserSearchLoading(true);
    try {
      const params = new URLSearchParams();
      params.append("user_email", searchText); // Always search by email
      if (accessToken == null) {
        return;
      }
      const response = await userFilterUICall(accessToken, params);

      const data: User[] = response;
      const options: UserOption[] = data.map((user) => ({
        label: `${user.user_email} (${user.user_id})`,
        value: user.user_id,
        user,
      }));

      setUserOptions(options);
    } catch (error) {
      console.error("Error fetching users:", error);
      NotificationsManager.fromBackend(t("virtualKeys.createKey.userSearchError"));
    } finally {
      setUserSearchLoading(false);
    }
  };

  const handleUserSearch = useDebouncedCallback((text: string) => fetchUsers(text), { wait: DEBOUNCE_WAIT_MS });

  const handleUserSelect = (_value: string, option: UserOption): void => {
    const selectedUser = option.user;
    form.setFieldsValue({
      user_id: selectedUser.user_id,
    });
  };

  return (
    <div>
      {userRole && rolesWithWriteAccess.includes(userRole) && (
        <Button className="mx-auto" onClick={() => setIsModalVisible(true)} data-testid="create-key-button">
          {buttonLabel ?? t("virtualKeys.createAction")}
        </Button>
      )}
      <Modal open={isModalVisible} width={1000} footer={null} onOk={handleOk} onCancel={handleCancel}>
        <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
          {/* Section 1: Key Ownership */}
          <div className="mb-8">
            <Title className="mb-4">{t("virtualKeys.createKey.ownership")}</Title>
            <Form.Item
              label={
                <span>
                  {t("virtualKeys.createKey.ownedBy")}{" "}
                  <Tooltip title={t("virtualKeys.createKey.ownedByTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              className="mb-4"
            >
              <Radio.Group onChange={(e) => setKeyOwner(e.target.value)} value={keyOwner}>
                <Radio value="you">{t("virtualKeys.createKey.ownerYou")}</Radio>
                <Radio value="service_account">{t("virtualKeys.createKey.ownerServiceAccount")}</Radio>
                {userRole === "Admin" && (
                  <Radio value="another_user">{t("virtualKeys.createKey.ownerAnotherUser")}</Radio>
                )}
                <Radio value="agent">
                  {t("virtualKeys.createKey.ownerAgent")}{" "}
                  <Tag color="purple">{t("virtualKeys.createKey.newBadge")}</Tag>
                </Radio>
              </Radio.Group>
            </Form.Item>

            {keyOwner === "another_user" && (
              <Form.Item
                label={
                  <span>
                    {t("virtualKeys.createKey.userId")}{" "}
                    <Tooltip title={t("virtualKeys.createKey.userIdTooltip")}>
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="user_id"
                className="mt-4"
                rules={[
                  {
                    required: keyOwner === "another_user",
                    message: t("virtualKeys.createKey.userIdRequired"),
                  },
                ]}
              >
                <div>
                  <div style={{ display: "flex", marginBottom: "8px" }}>
                    <Select
                      showSearch
                      placeholder={t("virtualKeys.createKey.userSearchPlaceholder")}
                      filterOption={false}
                      onSearch={handleUserSearch}
                      onSelect={(value, option) => handleUserSelect(value, option as UserOption)}
                      options={userOptions}
                      loading={userSearchLoading}
                      allowClear
                      style={{ width: "100%" }}
                      notFoundContent={
                        userSearchLoading ? t("virtualKeys.createKey.searching") : t("virtualKeys.createKey.noUsers")
                      }
                    />
                    <Button2 onClick={() => setIsCreateUserModalVisible(true)} style={{ marginLeft: "8px" }}>
                      {t("virtualKeys.createKey.createUser")}
                    </Button2>
                  </div>
                  <div className="text-xs text-gray-500">{t("virtualKeys.createKey.userSearchHint")}</div>
                </div>
              </Form.Item>
            )}
            {keyOwner === "agent" && (
              <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-md">
                <div className="mb-3">
                  <span className="text-sm font-medium text-gray-700">
                    {t("virtualKeys.createKey.selectAgent")} <span className="text-red-500">*</span>
                  </span>
                </div>
                <Select
                  showSearch
                  placeholder={t("virtualKeys.createKey.selectAgentPlaceholder")}
                  style={{ width: "100%" }}
                  value={selectedAgentId}
                  onChange={(value) => setSelectedAgentId(value)}
                  filterOption={(input, option) =>
                    (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
                  }
                  options={agentsList.map((a) => ({
                    label: a.agent_name || a.agent_id,
                    value: a.agent_id,
                  }))}
                />
                <div className="text-xs text-gray-500 mt-2">{t("virtualKeys.createKey.agentHint")}</div>
              </div>
            )}
            <Form.Item
              label={
                <span>
                  {t("virtualKeys.createKey.organization")}{" "}
                  <Tooltip title={t("virtualKeys.createKey.organizationTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="organization_id"
              className="mt-4"
            >
              <OrganizationDropdown
                organizations={organizations}
                loading={isOrganizationsLoading}
                disabled={userRole !== "Admin"}
                placeholder={t("virtualKeys.createKey.allOrganizations")}
                onChange={(orgId) => {
                  setSelectedOrganizationId(orgId || null);
                  // Clear team and project when org changes
                  setSelectedCreateKeyTeam(null);
                  setSelectedProjectId(null);
                  form.setFieldValue("team_id", undefined);
                  form.setFieldValue("project_id", undefined);
                }}
              />
            </Form.Item>
            <Form.Item
              label={
                <span>
                  {t("virtualKeys.createKey.team")}{" "}
                  <Tooltip title={t("virtualKeys.createKey.teamTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="team_id"
              initialValue={team ? team.team_id : null}
              className="mt-4"
              rules={[
                {
                  required: keyOwner === "service_account",
                  message: t("virtualKeys.createKey.teamRequired"),
                },
              ]}
              help={keyOwner === "service_account" ? t("virtualKeys.createKey.required") : ""}
            >
              <TeamDropdown
                disabled={selectedProjectId !== null}
                organizationId={selectedOrganizationId}
                placeholder={t("virtualKeys.createKey.searchTeam")}
                emptyLabel={t("virtualKeys.createKey.noTeams")}
                onTeamSelect={(team) => {
                  setSelectedCreateKeyTeam(team);
                  setSelectedProjectId(null);
                  form.setFieldValue("project_id", undefined);
                  // Auto-populate org from team for non-admin users
                  if (team?.organization_id) {
                    setSelectedOrganizationId(team.organization_id);
                    form.setFieldValue("organization_id", team.organization_id);
                  } else if (!team) {
                    setSelectedOrganizationId(null);
                    form.setFieldValue("organization_id", undefined);
                  }
                }}
              />
            </Form.Item>
            {enableProjectsUI && (
              <Form.Item
                label={
                  <span>
                    {t("virtualKeys.createKey.project")}{" "}
                    <Tooltip title={t("virtualKeys.createKey.projectTooltip")}>
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="project_id"
                className="mt-4"
              >
                <ProjectDropdown
                  projects={projects}
                  teamId={selectedCreateKeyTeam?.team_id}
                  loading={isProjectsLoading || !teams}
                  onChange={(projectId) => {
                    if (!projectId) {
                      setSelectedProjectId(null);
                      setSelectedCreateKeyTeam(null);
                      form.setFieldValue("team_id", undefined);
                      return;
                    }
                    setSelectedProjectId(projectId);
                  }}
                />
              </Form.Item>
            )}
          </div>

          {/* Show message when team selection is required */}
          {isFormDisabled && (
            <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-md">
              <Text className="text-blue-800 text-sm">{t("virtualKeys.createKey.selectTeamMessage")}</Text>
            </div>
          )}

          {/* Section 2: Key Details */}
          {!isFormDisabled && (
            <div className="mb-8">
              <Title className="mb-4">{t("virtualKeys.createKey.details")}</Title>
              <Form.Item
                label={
                  <span>
                    {keyOwner === "you" || keyOwner === "another_user"
                      ? t("virtualKeys.createKey.keyName")
                      : t("virtualKeys.createKey.serviceAccountId")}{" "}
                    <Tooltip
                      title={
                        keyOwner === "you" || keyOwner === "another_user"
                          ? t("virtualKeys.createKey.keyNameTooltip")
                          : t("virtualKeys.createKey.serviceAccountIdTooltip")
                      }
                    >
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="key_alias"
                rules={[
                  {
                    required: true,
                    message:
                      keyOwner === "you" || keyOwner === "another_user"
                        ? t("virtualKeys.createKey.keyNameRequired")
                        : t("virtualKeys.createKey.serviceAccountIdRequired"),
                  },
                ]}
                help={t("virtualKeys.createKey.required")}
              >
                <TextInput placeholder="" />
              </Form.Item>

              <Form.Item
                label={
                  <span>
                    {t("virtualKeys.createKey.models")}{" "}
                    <Tooltip title={t("virtualKeys.createKey.modelsTooltip")}>
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="models"
                rules={[]}
                help={
                  keyType === "management" || keyType === "read_only"
                    ? t("virtualKeys.createKey.modelsDisabled")
                    : t("virtualKeys.createKey.modelsOptional")
                }
                className="mt-4"
              >
                <Select
                  mode="multiple"
                  placeholder={t("virtualKeys.createKey.selectModels")}
                  style={{ width: "100%" }}
                  disabled={keyType === "management" || keyType === "read_only"}
                  onChange={(values) => {
                    if (values.includes("all-team-models")) {
                      form.setFieldsValue({ models: ["all-team-models"] });
                    } else if (values.includes("all-proxy-models")) {
                      form.setFieldsValue({ models: ["all-proxy-models"] });
                    }
                  }}
                >
                  {!selectedProjectId && selectedCreateKeyTeam && (
                    <Option key="all-team-models" value="all-team-models">
                      {t("virtualKeys.createKey.allTeamModels")}
                    </Option>
                  )}
                  {!selectedProjectId && !selectedCreateKeyTeam && (
                    <Option key="all-proxy-models" value="all-proxy-models">
                      {t("virtualKeys.createKey.allProxyModels")}
                    </Option>
                  )}
                  {modelsToPick.map((model: string) => (
                    <Option key={model} value={model} disabled={hasAllModelsSentinel(selectedModels)}>
                      {getModelDisplayName(model)}
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item
                label={
                  <span>
                    {t("virtualKeys.createKey.keyType")}{" "}
                    <Tooltip title={t("virtualKeys.createKey.keyTypeTooltip")}>
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="key_type"
                initialValue="llm_api"
                className="mt-4"
              >
                <Select
                  defaultValue="llm_api"
                  placeholder={t("virtualKeys.createKey.selectKeyType")}
                  style={{ width: "100%" }}
                  optionLabelProp="label"
                  onChange={(value) => {
                    setKeyType(value);
                    // Clear models field and disable if management or read_only
                    if (value === "management" || value === "read_only") {
                      form.setFieldsValue({ models: [] });
                    }
                  }}
                >
                  <Option value="llm_api" label={t("virtualKeys.createKey.aiApis")}>
                    <div style={{ padding: "4px 0" }}>
                      <Typography.Text strong>{t("virtualKeys.createKey.aiApis")}</Typography.Text>
                      <Typography.Paragraph type="secondary" style={{ fontSize: 11, margin: "2px 0 0" }}>
                        {t("virtualKeys.createKey.aiApisDescription")}
                      </Typography.Paragraph>
                    </div>
                  </Option>
                  <Option value="management" label={t("virtualKeys.createKey.management")}>
                    <div style={{ padding: "4px 0" }}>
                      <Typography.Text strong>{t("virtualKeys.createKey.management")}</Typography.Text>
                      <Typography.Paragraph type="secondary" style={{ fontSize: 11, margin: "2px 0 0" }}>
                        {t("virtualKeys.createKey.managementDescription")}
                      </Typography.Paragraph>
                    </div>
                  </Option>
                  <Option value="default" label={t("virtualKeys.createKey.fullAccess")}>
                    <div style={{ padding: "4px 0" }}>
                      <Typography.Text strong>{t("virtualKeys.createKey.fullAccess")}</Typography.Text>
                      <Typography.Paragraph type="secondary" style={{ fontSize: 11, margin: "2px 0 0" }}>
                        {t("virtualKeys.createKey.fullAccessDescription")}
                      </Typography.Paragraph>
                    </div>
                  </Option>
                </Select>
              </Form.Item>
            </div>
          )}

          {/* Section 3: Optional Settings */}
          {!isFormDisabled && (
            <div className="mb-8">
              <Accordion className="mt-4 mb-4">
                <AccordionHeader>
                  <Title className="m-0">{t("virtualKeys.createKey.optionalSettings")}</Title>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.maxBudget")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.maxBudgetTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="max_budget"
                    help={t("virtualKeys.createKey.optional.teamMaxBudget", {
                      value:
                        team?.max_budget !== null && team?.max_budget !== undefined
                          ? team.max_budget
                          : t("virtualKeys.createKey.optional.unlimited"),
                    })}
                    rules={[
                      {
                        validator: async (_, value) => {
                          if (value && team && team.max_budget !== null && value > team.max_budget) {
                            throw new Error(
                              `Budget cannot exceed team max budget: $${formatNumberWithCommas(team.max_budget, 4)}`,
                            );
                          }
                        },
                      },
                    ]}
                  >
                    <NumericalInput step={0.01} precision={2} width={200} />
                  </Form.Item>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.resetBudget")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.resetBudgetTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="budget_duration"
                    help={t("virtualKeys.createKey.optional.teamResetBudget", {
                      value:
                        team?.budget_duration !== null && team?.budget_duration !== undefined
                          ? team.budget_duration
                          : t("virtualKeys.createKey.optional.none"),
                    })}
                  >
                    <BudgetDurationDropdown
                      placeholder={t("virtualKeys.createKey.optional.neverResets")}
                      onChange={(value) => form.setFieldValue("budget_duration", value)}
                    />
                  </Form.Item>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.budgetWindows")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.budgetWindowsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                  >
                    <BudgetWindowsEditor
                      value={budgetLimits}
                      onChange={setBudgetLimits}
                      labels={{
                        hourly: t("virtualKeys.createKey.optional.budgetWindowHourly"),
                        hourlyHint: t("virtualKeys.createKey.optional.budgetWindowHourlyHint"),
                        daily: t("virtualKeys.createKey.optional.budgetWindowDaily"),
                        dailyHint: t("virtualKeys.createKey.optional.budgetWindowDailyHint"),
                        weekly: t("virtualKeys.createKey.optional.budgetWindowWeekly"),
                        weeklyHint: t("virtualKeys.createKey.optional.budgetWindowWeeklyHint"),
                        monthly: t("virtualKeys.createKey.optional.budgetWindowMonthly"),
                        monthlyHint: t("virtualKeys.createKey.optional.budgetWindowMonthlyHint"),
                        maxSpend: t("virtualKeys.createKey.optional.budgetWindowMaxSpend"),
                        addWindow: t("virtualKeys.createKey.optional.addBudgetWindow"),
                      }}
                    />
                  </Form.Item>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.budgetFallbacks")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.budgetFallbacksTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                  >
                    <BudgetFallbacksEditor
                      key={budgetFallbacksKey}
                      value={budgetFallbacks}
                      onChange={setBudgetFallbacks}
                      availableModels={modelsToPick}
                      labels={{
                        description: t("virtualKeys.createKey.optional.budgetFallbackDescription"),
                        addFallback: t("virtualKeys.createKey.optional.addBudgetFallback"),
                        primaryModel: t("virtualKeys.createKey.optional.primaryModel"),
                        selectModel: t("virtualKeys.createKey.optional.selectModel"),
                        budgetExceeded: t("virtualKeys.createKey.optional.budgetExceededTry"),
                        fallbackModels: t("virtualKeys.createKey.optional.fallbackModels"),
                        selectFallbackModels: t("virtualKeys.createKey.optional.selectFallbackModels"),
                        selectPrimaryFirst: t("virtualKeys.createKey.optional.selectPrimaryFirst"),
                        more: t("virtualKeys.createKey.optional.more"),
                        triedInOrder: t("virtualKeys.createKey.optional.fallbackOrderHint"),
                        removeFallback: t("virtualKeys.createKey.optional.removeFallback"),
                      }}
                    />
                  </Form.Item>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.tpm")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.tpmTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="tpm_limit"
                    help={t("virtualKeys.createKey.optional.teamTpm", {
                      value:
                        team?.tpm_limit !== null && team?.tpm_limit !== undefined
                          ? team.tpm_limit
                          : t("virtualKeys.createKey.optional.unlimited"),
                    })}
                    rules={[
                      {
                        validator: async (_, value) => {
                          if (value && team && team.tpm_limit !== null && value > team.tpm_limit) {
                            throw new Error(
                              t("virtualKeys.createKey.optional.tpmValidation", { value: team.tpm_limit }),
                            );
                          }
                        },
                      },
                    ]}
                  >
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <RateLimitTypeFormItem
                    type="tpm"
                    name="tpm_limit_type"
                    className="mt-4"
                    initialValue={null}
                    form={form}
                    showDetailedDescriptions={true}
                    labels={{
                      fieldLabel: t("virtualKeys.createKey.optional.rateLimitType", { type: "TPM" }),
                      tooltip: t("virtualKeys.createKey.optional.rateLimitTypeTooltip", { type: "TPM" }),
                      placeholder: t("virtualKeys.createKey.optional.selectRateLimitType"),
                      defaultLabel: t("virtualKeys.createKey.optional.defaultRateLimit"),
                      defaultDescription: t("virtualKeys.createKey.optional.defaultRateLimitDescription"),
                      guaranteedLabel: t("virtualKeys.createKey.optional.guaranteedRateLimit"),
                      guaranteedDescription: t("virtualKeys.createKey.optional.guaranteedRateLimitDescription"),
                      dynamicLabel: t("virtualKeys.createKey.optional.dynamicRateLimit"),
                      dynamicDescription: t("virtualKeys.createKey.optional.dynamicRateLimitDescription"),
                      bestEffortLabel: t("virtualKeys.createKey.optional.bestEffortRateLimit"),
                    }}
                  />
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.rpm")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.rpmTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="rpm_limit"
                    help={t("virtualKeys.createKey.optional.teamRpm", {
                      value:
                        team?.rpm_limit !== null && team?.rpm_limit !== undefined
                          ? team.rpm_limit
                          : t("virtualKeys.createKey.optional.unlimited"),
                    })}
                    rules={[
                      {
                        validator: async (_, value) => {
                          if (value && team && team.rpm_limit !== null && value > team.rpm_limit) {
                            throw new Error(
                              t("virtualKeys.createKey.optional.rpmValidation", { value: team.rpm_limit }),
                            );
                          }
                        },
                      },
                    ]}
                  >
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <RateLimitTypeFormItem
                    type="rpm"
                    name="rpm_limit_type"
                    className="mt-4"
                    initialValue={null}
                    form={form}
                    showDetailedDescriptions={true}
                    labels={{
                      fieldLabel: t("virtualKeys.createKey.optional.rateLimitType", { type: "RPM" }),
                      tooltip: t("virtualKeys.createKey.optional.rateLimitTypeTooltip", { type: "RPM" }),
                      placeholder: t("virtualKeys.createKey.optional.selectRateLimitType"),
                      defaultLabel: t("virtualKeys.createKey.optional.defaultRateLimit"),
                      defaultDescription: t("virtualKeys.createKey.optional.defaultRateLimitDescription"),
                      guaranteedLabel: t("virtualKeys.createKey.optional.guaranteedRateLimit"),
                      guaranteedDescription: t("virtualKeys.createKey.optional.guaranteedRateLimitDescription"),
                      dynamicLabel: t("virtualKeys.createKey.optional.dynamicRateLimit"),
                      dynamicDescription: t("virtualKeys.createKey.optional.dynamicRateLimitDescription"),
                      bestEffortLabel: t("virtualKeys.createKey.optional.bestEffortRateLimit"),
                    }}
                  />
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.perTagLimits")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.perTagLimitsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                  >
                    <TagRateLimitEditor
                      value={tagRateLimits}
                      onChange={setTagRateLimits}
                      labels={{
                        tagPlaceholder: t("virtualKeys.createKey.optional.tagPlaceholder"),
                        rpmPlaceholder: t("virtualKeys.createKey.optional.tagRpmPlaceholder"),
                        addLimit: t("virtualKeys.createKey.optional.addTagLimit"),
                        removeLimit: t("virtualKeys.createKey.optional.removeTagLimit"),
                      }}
                    />
                  </Form.Item>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.throttle")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.throttleTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="throttle_on_budget_exceeded"
                    valuePropName="checked"
                  >
                    <Switch
                      checkedChildren={t("virtualKeys.createKey.optional.yes")}
                      unCheckedChildren={t("virtualKeys.createKey.optional.no")}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.guardrails")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.guardrailsTooltip")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="guardrails"
                    className="mt-4"
                    help={
                      canEditGuardrails
                        ? t("virtualKeys.createKey.optional.guardrailsHelp")
                        : t("virtualKeys.createKey.optional.guardrailsPremium")
                    }
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      disabled={!canEditGuardrails}
                      placeholder={
                        !canEditGuardrails
                          ? t("virtualKeys.createKey.optional.guardrailsPremium")
                          : t("virtualKeys.createKey.optional.selectGuardrails")
                      }
                      options={guardrailsList.map((name) => ({ value: name, label: name }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.disableGlobalGuardrails")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.disableGlobalGuardrailsTooltip")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="disable_global_guardrails"
                    className="mt-4"
                    valuePropName="checked"
                    help={
                      canEditGuardrails
                        ? t("virtualKeys.createKey.optional.bypassGlobalGuardrails")
                        : t("virtualKeys.createKey.optional.disableGlobalGuardrailsPremium")
                    }
                  >
                    <Switch
                      disabled={!canEditGuardrails}
                      checkedChildren={t("virtualKeys.createKey.optional.yes")}
                      unCheckedChildren={t("virtualKeys.createKey.optional.no")}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.policies")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.policiesTooltip")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="policies"
                    className="mt-4"
                    help={
                      premiumUser
                        ? t("virtualKeys.createKey.optional.policiesHelp")
                        : t("virtualKeys.createKey.optional.policiesPremium")
                    }
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      disabled={!premiumUser}
                      placeholder={
                        !premiumUser
                          ? t("virtualKeys.createKey.optional.policiesPremium")
                          : t("virtualKeys.createKey.optional.selectPolicies")
                      }
                      options={policiesList.map((name) => ({ value: name, label: name }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.prompts")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.promptsTooltip")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/prompt_management"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="prompts"
                    className="mt-4"
                    help={
                      premiumUser
                        ? t("virtualKeys.createKey.optional.promptsHelp")
                        : t("virtualKeys.createKey.optional.promptsPremium")
                    }
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      disabled={!premiumUser}
                      placeholder={
                        !premiumUser
                          ? t("virtualKeys.createKey.optional.promptsPremium")
                          : t("virtualKeys.createKey.optional.selectPrompts")
                      }
                      options={promptsList.map((name) => ({ value: name, label: name }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.accessGroups")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.accessGroupsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="access_group_ids"
                    className="mt-4"
                    help={t("virtualKeys.createKey.optional.accessGroupsHelp")}
                  >
                    <AccessGroupSelector placeholder={t("virtualKeys.createKey.optional.selectAccessGroups")} />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.passThroughRoutes")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.passThroughRoutesTooltip")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/pass_through"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="allowed_passthrough_routes"
                    className="mt-4"
                    help={
                      premiumUser
                        ? t("virtualKeys.createKey.optional.passThroughRoutesHelp")
                        : t("virtualKeys.createKey.optional.passThroughRoutesPremium")
                    }
                  >
                    <PassThroughRoutesSelector
                      accessToken={accessToken}
                      placeholder={
                        !premiumUser
                          ? t("virtualKeys.createKey.optional.passThroughRoutesPremium")
                          : t("virtualKeys.createKey.optional.selectPassThroughRoutes")
                      }
                      disabled={!premiumUser}
                      teamId={selectedCreateKeyTeam ? selectedCreateKeyTeam.team_id : null}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.vectorStores")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.vectorStoresTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="allowed_vector_store_ids"
                    className="mt-4"
                    help={t("virtualKeys.createKey.optional.vectorStoresHelp")}
                  >
                    <VectorStoreSelector
                      onChange={(values: string[]) => form.setFieldValue("allowed_vector_store_ids", values)}
                      value={form.getFieldValue("allowed_vector_store_ids")}
                      accessToken={accessToken}
                      placeholder={t("virtualKeys.createKey.optional.selectVectorStores")}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.metadata")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.metadataTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="metadata"
                    className="mt-4"
                  >
                    <Input.TextArea rows={4} placeholder={t("virtualKeys.createKey.optional.metadataPlaceholder")} />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("virtualKeys.createKey.optional.tags")}{" "}
                        <Tooltip title={t("virtualKeys.createKey.optional.tagsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="tags"
                    className="mt-4"
                    help={t("virtualKeys.createKey.optional.tagsHelp")}
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      placeholder={t("virtualKeys.createKey.optional.selectTags")}
                      tokenSeparators={[","]}
                      options={tagOptions}
                    />
                  </Form.Item>
                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <b>{t("virtualKeys.createKey.optional.mcpSettings")}</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <Form.Item
                        label={
                          <span>
                            {t("virtualKeys.createKey.optional.allowedMcpServers")}{" "}
                            <Tooltip title={t("virtualKeys.createKey.optional.allowedMcpServersTooltip")}>
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="allowed_mcp_servers_and_groups"
                        help={t("virtualKeys.createKey.optional.allowedMcpServersHelp")}
                      >
                        <MCPServerSelector
                          onChange={(val: any) => form.setFieldValue("allowed_mcp_servers_and_groups", val)}
                          value={form.getFieldValue("allowed_mcp_servers_and_groups")}
                          accessToken={accessToken}
                          teamId={selectedCreateKeyTeam?.team_id ?? null}
                          placeholder={t("virtualKeys.createKey.optional.selectMcpServers")}
                          allowNoMcpServers
                        />
                      </Form.Item>

                      {/* Hidden field to register mcp_tool_permissions with the form */}
                      <Form.Item name="mcp_tool_permissions" initialValue={{}} hidden>
                        <Input type="hidden" />
                      </Form.Item>

                      <Form.Item
                        noStyle
                        shouldUpdate={(prevValues, currentValues) =>
                          prevValues.allowed_mcp_servers_and_groups !== currentValues.allowed_mcp_servers_and_groups ||
                          prevValues.mcp_tool_permissions !== currentValues.mcp_tool_permissions
                        }
                      >
                        {() => (
                          <div className="mt-6">
                            <MCPToolPermissions
                              accessToken={accessToken}
                              selectedServers={(
                                form.getFieldValue("allowed_mcp_servers_and_groups")?.servers || []
                              ).filter((s: string) => s !== NO_MCP_SERVERS_SENTINEL)}
                              toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
                              onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
                            />
                          </div>
                        )}
                      </Form.Item>
                    </AccordionBody>
                  </Accordion>

                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <b>{t("virtualKeys.createKey.optional.agentSettings")}</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <Form.Item
                        label={
                          <span>
                            {t("virtualKeys.createKey.optional.allowedAgents")}{" "}
                            <Tooltip title={t("virtualKeys.createKey.optional.allowedAgentsTooltip")}>
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="allowed_agents_and_groups"
                        help={t("virtualKeys.createKey.optional.allowedAgentsHelp")}
                      >
                        <AgentSelector
                          onChange={(val: any) => form.setFieldValue("allowed_agents_and_groups", val)}
                          value={form.getFieldValue("allowed_agents_and_groups")}
                          accessToken={accessToken}
                          placeholder={t("virtualKeys.createKey.optional.selectAgents")}
                        />
                      </Form.Item>
                    </AccordionBody>
                  </Accordion>

                  {premiumUser ? (
                    <Accordion className="mt-4 mb-4">
                      <AccordionHeader>
                        <b>{t("virtualKeys.createKey.optional.loggingSettings")}</b>
                      </AccordionHeader>
                      <AccordionBody>
                        <div className="mt-4">
                          <PremiumLoggingSettings
                            value={loggingSettings}
                            onChange={setLoggingSettings}
                            premiumUser={true}
                            disabledCallbacks={disabledCallbacks}
                            onDisabledCallbacksChange={setDisabledCallbacks}
                          />
                        </div>
                      </AccordionBody>
                    </Accordion>
                  ) : (
                    <Tooltip
                      title={
                        <span>
                          {t("virtualKeys.createKey.optional.loggingPremium")} —
                          <a href="https://www.litellm.ai/enterprise" target="_blank">
                            https://www.litellm.ai/enterprise
                          </a>
                        </span>
                      }
                      placement="top"
                    >
                      <div style={{ position: "relative" }}>
                        <div style={{ opacity: 0.5 }}>
                          <Accordion className="mt-4 mb-4">
                            <AccordionHeader>
                              <b>{t("virtualKeys.createKey.optional.loggingSettings")}</b>
                            </AccordionHeader>
                            <AccordionBody>
                              <div className="mt-4">
                                <PremiumLoggingSettings
                                  value={loggingSettings}
                                  onChange={setLoggingSettings}
                                  premiumUser={false}
                                  disabledCallbacks={disabledCallbacks}
                                  onDisabledCallbacksChange={setDisabledCallbacks}
                                />
                              </div>
                            </AccordionBody>
                          </Accordion>
                        </div>
                        <div style={{ position: "absolute", inset: 0, cursor: "not-allowed" }} />
                      </div>
                    </Tooltip>
                  )}

                  <Accordion key={`router-settings-accordion-${routerSettingsKey}`} className="mt-4 mb-4">
                    <AccordionHeader>
                      <b>{t("virtualKeys.createKey.optional.routerSettings")}</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <div className="mt-4 w-full">
                        <RouterSettingsAccordion
                          key={routerSettingsKey}
                          accessToken={accessToken || ""}
                          value={routerSettings || undefined}
                          onChange={setRouterSettings}
                          modelData={
                            userModels.length > 0
                              ? { data: userModels.map((model) => ({ model_name: model })) }
                              : undefined
                          }
                          labels={{
                            loadBalancing: t("virtualKeys.createKey.optional.loadBalancing"),
                            fallbacks: t("virtualKeys.createKey.optional.routerFallbacks"),
                            routingSettings: t("virtualKeys.createKey.optional.routingSettings"),
                            routingDescription: t("virtualKeys.createKey.optional.routingDescription"),
                            routingStrategy: t("virtualKeys.createKey.optional.routingStrategy"),
                            routingStrategyDescription: t("virtualKeys.createKey.optional.routingStrategyDescription"),
                            tagFiltering: t("virtualKeys.createKey.optional.tagFiltering"),
                            tagFilteringDescription: t("virtualKeys.createKey.optional.tagFilteringDescription"),
                            learnMore: t("virtualKeys.createKey.optional.learnMore"),
                            reliability: t("virtualKeys.createKey.optional.reliability"),
                            reliabilityDescription: t("virtualKeys.createKey.optional.reliabilityDescription"),
                            fieldLabels: {
                              allowed_fails: t("virtualKeys.createKey.optional.allowedFails"),
                              cooldown_time: t("virtualKeys.createKey.optional.cooldownTime"),
                              num_retries: t("virtualKeys.createKey.optional.numRetries"),
                              timeout: t("virtualKeys.createKey.optional.timeout"),
                              retry_after: t("virtualKeys.createKey.optional.retryAfter"),
                              model_group_alias: t("virtualKeys.createKey.optional.modelGroupAlias"),
                            },
                            fieldDescriptions: {
                              allowed_fails: t("virtualKeys.createKey.optional.allowedFailsDescription"),
                              cooldown_time: t("virtualKeys.createKey.optional.cooldownTimeDescription"),
                              num_retries: t("virtualKeys.createKey.optional.numRetriesDescription"),
                              timeout: t("virtualKeys.createKey.optional.timeoutDescription"),
                              retry_after: t("virtualKeys.createKey.optional.retryAfterDescription"),
                              model_group_alias: t("virtualKeys.createKey.optional.modelGroupAliasDescription"),
                            },
                            fallbackLabels: {
                              group: t("virtualKeys.createKey.optional.fallbackGroup"),
                              atLeastOne: t("virtualKeys.createKey.optional.fallbackAtLeastOne"),
                              empty: t("virtualKeys.createKey.optional.fallbackGroupsEmpty"),
                              createFirst: t("virtualKeys.createKey.optional.fallbackCreateFirst"),
                              primaryModel: t("virtualKeys.createKey.optional.primaryModel"),
                              selectPrimary: t("virtualKeys.createKey.optional.fallbackSelectPrimary"),
                              selectPrimaryHint: t("virtualKeys.createKey.optional.fallbackSelectPrimaryHint"),
                              ifFails: t("virtualKeys.createKey.optional.fallbackIfFails"),
                              fallbackChain: t("virtualKeys.createKey.optional.fallbackChain"),
                              maxFallbacks: t("virtualKeys.createKey.optional.fallbackMax"),
                              selectFallbacks: t("virtualKeys.createKey.optional.fallbackSelect"),
                              maxReached: t("virtualKeys.createKey.optional.fallbackMaxReached"),
                              more: t("virtualKeys.createKey.optional.more"),
                              selectionHint: t("virtualKeys.createKey.optional.fallbackSelectionHint"),
                              maxReachedHint: t("virtualKeys.createKey.optional.fallbackMaxReachedHint"),
                              noFallbacks: t("virtualKeys.createKey.optional.fallbackNone"),
                              addFromDropdown: t("virtualKeys.createKey.optional.fallbackAddFromDropdown"),
                              removeFallback: t("virtualKeys.createKey.optional.removeFallback"),
                            },
                          }}
                        />
                      </div>
                    </AccordionBody>
                  </Accordion>

                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <b>{t("virtualKeys.createKey.optional.modelAliases")}</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <div className="mt-4">
                        <Text className="text-sm text-gray-600 mb-4">
                          {t("virtualKeys.createKey.optional.modelAliasesDescription")}
                        </Text>
                        <ModelAliasManager
                          accessToken={accessToken}
                          initialModelAliases={modelAliases}
                          onAliasUpdate={setModelAliases}
                          showExampleConfig={false}
                          labels={{
                            addNew: t("virtualKeys.createKey.optional.aliasAddNew"),
                            aliasName: t("virtualKeys.createKey.optional.aliasName"),
                            targetModel: t("virtualKeys.createKey.optional.aliasTargetModel"),
                            aliasPlaceholder: t("virtualKeys.createKey.optional.aliasPlaceholder"),
                            selectTarget: t("virtualKeys.createKey.optional.aliasSelectTarget"),
                            add: t("virtualKeys.createKey.optional.aliasAdd"),
                            manage: t("virtualKeys.createKey.optional.aliasManage"),
                            actions: t("virtualKeys.createKey.optional.actions"),
                            empty: t("virtualKeys.createKey.optional.aliasEmpty"),
                            save: t("virtualKeys.createKey.optional.save"),
                            cancel: t("virtualKeys.createKey.optional.cancel"),
                            requiredError: t("virtualKeys.createKey.optional.aliasRequiredError"),
                            duplicateError: t("virtualKeys.createKey.optional.aliasDuplicateError"),
                            added: t("virtualKeys.createKey.optional.aliasAdded"),
                            updated: t("virtualKeys.createKey.optional.aliasUpdated"),
                            deleted: t("virtualKeys.createKey.optional.aliasDeleted"),
                          }}
                        />
                      </div>
                    </AccordionBody>
                  </Accordion>

                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <b>{t("virtualKeys.createKey.optional.keyLifecycle")}</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <div className="mt-4">
                        <KeyLifecycleSettings
                          form={form}
                          autoRotationEnabled={autoRotationEnabled}
                          onAutoRotationChange={setAutoRotationEnabled}
                          rotationInterval={rotationInterval}
                          onRotationIntervalChange={setRotationInterval}
                          isCreateMode={true}
                          labels={{
                            expirySettings: t("virtualKeys.createKey.optional.expirySettings"),
                            expireKey: t("virtualKeys.createKey.optional.expireKey"),
                            expiryTooltip: t("virtualKeys.createKey.optional.expiryTooltip"),
                            neverExpire: t("virtualKeys.createKey.optional.neverExpire"),
                            createPlaceholder: t("virtualKeys.createKey.optional.expiryCreatePlaceholder"),
                            editPlaceholder: t("virtualKeys.createKey.optional.expiryEditPlaceholder"),
                            rotationSettings: t("virtualKeys.createKey.optional.rotationSettings"),
                            enableRotation: t("virtualKeys.createKey.optional.enableRotation"),
                            rotationTooltip: t("virtualKeys.createKey.optional.rotationTooltip"),
                            rotationInterval: t("virtualKeys.createKey.optional.rotationInterval"),
                            rotationIntervalTooltip: t("virtualKeys.createKey.optional.rotationIntervalTooltip"),
                            selectInterval: t("virtualKeys.createKey.optional.selectInterval"),
                            days: t("virtualKeys.createKey.optional.days"),
                            customInterval: t("virtualKeys.createKey.optional.customInterval"),
                            customPlaceholder: t("virtualKeys.createKey.optional.customIntervalPlaceholder"),
                            supportedFormats: t("virtualKeys.createKey.optional.supportedFormats"),
                            rotationNotice: t("virtualKeys.createKey.optional.rotationNotice"),
                          }}
                        />
                      </div>
                    </AccordionBody>
                  </Accordion>
                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <div className="flex items-center gap-2">
                        <b>{t("virtualKeys.createKey.optional.advancedSettings")}</b>
                        <Tooltip
                          title={
                            <span>
                              {t("virtualKeys.createKey.optional.advancedHelpPrefix")}{" "}
                              <a
                                href={
                                  proxyBaseUrl
                                    ? `${proxyBaseUrl}/#/key%20management/generate_key_fn_key_generate_post`
                                    : `/#/key%20management/generate_key_fn_key_generate_post`
                                }
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-400 hover:text-blue-300"
                              >
                                {t("virtualKeys.createKey.optional.documentation")}
                              </a>
                            </span>
                          }
                        >
                          <InfoCircleOutlined className="text-gray-400 hover:text-gray-300 cursor-help" />
                        </Tooltip>
                      </div>
                    </AccordionHeader>
                    <AccordionBody>
                      <SchemaFormFields
                        schemaComponent="GenerateKeyRequest"
                        form={form}
                        overrideLabels={{
                          spend: t("virtualKeys.createKey.optional.advancedSpend"),
                          user_id: t("virtualKeys.createKey.optional.advancedUserId"),
                          agent_id: t("virtualKeys.createKey.optional.advancedAgentId"),
                          max_parallel_requests: t("virtualKeys.createKey.optional.advancedMaxParallelRequests"),
                          budget_limits: t("virtualKeys.createKey.optional.advancedBudgetLimits"),
                          allowed_cache_controls: t("virtualKeys.createKey.optional.advancedAllowedCacheControls"),
                          config: t("virtualKeys.createKey.optional.advancedConfig"),
                          permissions: t("virtualKeys.createKey.optional.advancedPermissions"),
                          model_max_budget: t("virtualKeys.createKey.optional.advancedModelMaxBudget"),
                          model_rpm_limit: t("virtualKeys.createKey.optional.advancedModelRpmLimit"),
                          model_tpm_limit: t("virtualKeys.createKey.optional.advancedModelTpmLimit"),
                          mcp_rpm_limit: t("virtualKeys.createKey.optional.advancedMcpRpmLimit"),
                          blocked: t("virtualKeys.createKey.optional.advancedBlocked"),
                          aliases: t("virtualKeys.createKey.optional.advancedAliases"),
                          object_permission: t("virtualKeys.createKey.optional.advancedObjectPermission"),
                          key: t("virtualKeys.createKey.optional.advancedCustomKey"),
                          budget_id: t("virtualKeys.createKey.optional.advancedBudgetId"),
                          enforced_params: t("virtualKeys.createKey.optional.advancedEnforcedParams"),
                          allowed_routes: t("virtualKeys.createKey.optional.advancedAllowedRoutes"),
                          allowed_vector_store_indexes: t("virtualKeys.createKey.optional.advancedVectorStoreIndexes"),
                          soft_budget: t("virtualKeys.createKey.optional.advancedSoftBudget"),
                          send_invite_email: t("virtualKeys.createKey.optional.advancedSendInviteEmail"),
                        }}
                        overrideTooltips={{
                          spend: t("virtualKeys.createKey.optional.advancedSpendTooltip"),
                          user_id: t("virtualKeys.createKey.optional.advancedUserIdTooltip"),
                          agent_id: t("virtualKeys.createKey.optional.advancedAgentIdTooltip"),
                          max_parallel_requests: t("virtualKeys.createKey.optional.advancedMaxParallelRequestsTooltip"),
                          budget_limits: t("virtualKeys.createKey.optional.advancedBudgetLimitsTooltip"),
                          allowed_cache_controls: t(
                            "virtualKeys.createKey.optional.advancedAllowedCacheControlsTooltip",
                          ),
                          config: t("virtualKeys.createKey.optional.advancedConfigTooltip"),
                          permissions: t("virtualKeys.createKey.optional.advancedPermissionsTooltip"),
                          model_max_budget: t("virtualKeys.createKey.optional.advancedModelMaxBudgetTooltip"),
                          model_rpm_limit: t("virtualKeys.createKey.optional.advancedModelRpmLimitTooltip"),
                          model_tpm_limit: t("virtualKeys.createKey.optional.advancedModelTpmLimitTooltip"),
                          mcp_rpm_limit: t("virtualKeys.createKey.optional.advancedMcpRpmLimitTooltip"),
                          blocked: t("virtualKeys.createKey.optional.advancedBlockedTooltip"),
                          aliases: t("virtualKeys.createKey.optional.advancedAliasesTooltip"),
                          object_permission: t("virtualKeys.createKey.optional.advancedObjectPermissionTooltip"),
                          key: t("virtualKeys.createKey.optional.advancedCustomKeyTooltip"),
                          budget_id: t("virtualKeys.createKey.optional.advancedBudgetIdTooltip"),
                          enforced_params: t("virtualKeys.createKey.optional.advancedEnforcedParamsTooltip"),
                          allowed_routes: t("virtualKeys.createKey.optional.advancedAllowedRoutesTooltip"),
                          allowed_vector_store_indexes: t(
                            "virtualKeys.createKey.optional.advancedVectorStoreIndexesTooltip",
                          ),
                          soft_budget: t("virtualKeys.createKey.optional.advancedSoftBudgetTooltip"),
                          send_invite_email: t("virtualKeys.createKey.optional.advancedSendInviteEmailTooltip"),
                        }}
                        overrideHelpTexts={{
                          spend: t("virtualKeys.createKey.optional.numericInput"),
                          user_id: t("virtualKeys.createKey.optional.textInput"),
                          agent_id: t("virtualKeys.createKey.optional.textInput"),
                          max_parallel_requests: t("virtualKeys.createKey.optional.numericInput"),
                          budget_limits: t("virtualKeys.createKey.optional.textInput"),
                          allowed_cache_controls: t("virtualKeys.createKey.optional.textInput"),
                          config: t("virtualKeys.createKey.optional.jsonInput"),
                          permissions: t("virtualKeys.createKey.optional.advancedPermissionsHelp"),
                          model_max_budget: t("virtualKeys.createKey.optional.numericInput"),
                          model_rpm_limit: t("virtualKeys.createKey.optional.textInput"),
                          model_tpm_limit: t("virtualKeys.createKey.optional.textInput"),
                          mcp_rpm_limit: t("virtualKeys.createKey.optional.textInput"),
                          blocked: t("virtualKeys.createKey.optional.advancedBlockedHelp"),
                          aliases: t("virtualKeys.createKey.optional.jsonInput"),
                          object_permission: t("virtualKeys.createKey.optional.textInput"),
                          key: t("virtualKeys.createKey.optional.textInput"),
                          budget_id: t("virtualKeys.createKey.optional.textInput"),
                          enforced_params: t("virtualKeys.createKey.optional.jsonInput"),
                          allowed_routes: t("virtualKeys.createKey.optional.textInput"),
                          allowed_vector_store_indexes: t("virtualKeys.createKey.optional.textInput"),
                          soft_budget: t("virtualKeys.createKey.optional.numericInput"),
                          send_invite_email: t("virtualKeys.createKey.optional.advancedBooleanInput"),
                        }}
                        jsonPlaceholder={t("virtualKeys.createKey.optional.jsonPlaceholder")}
                        validJsonError={t("virtualKeys.createKey.optional.validJsonError")}
                        requiredError={t("virtualKeys.createKey.optional.requiredError")}
                        errorPrefix={t("virtualKeys.createKey.optional.errorPrefix")}
                        excludedFields={[
                          "key_alias",
                          "team_id",
                          "organization_id",
                          "models",
                          "duration",
                          "metadata",
                          "tags",
                          "guardrails",
                          "max_budget",
                          "budget_duration",
                          "tpm_limit",
                          "rpm_limit",
                          "budget_fallbacks",
                          "tag_rpm_limit",
                          "policies",
                          "prompts",
                          "disable_global_guardrails",
                          "throttle_on_budget_exceeded",
                          "allowed_passthrough_routes",
                          "rpm_limit_type",
                          "tpm_limit_type",
                          "router_settings",
                          "access_group_ids",
                          "key_type",
                          "auto_rotate",
                          "rotation_interval",
                          "project_id",
                          ...(disableCustomApiKeys ? ["key"] : []),
                        ]}
                      />
                    </AccordionBody>
                  </Accordion>
                </AccordionBody>
              </Accordion>
            </div>
          )}

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit" disabled={isFormDisabled} style={{ opacity: isFormDisabled ? 0.5 : 1 }}>
              {t("virtualKeys.createKey.create")}
            </Button2>
          </div>
        </Form>
      </Modal>

      {/* Add the Create User Modal */}
      {isCreateUserModalVisible && (
        <Modal
          title={t("virtualKeys.createKey.createNewUser")}
          open={isCreateUserModalVisible}
          onCancel={() => setIsCreateUserModalVisible(false)}
          footer={null}
          width={800}
        >
          <CreateUserButton
            userID={userID}
            accessToken={accessToken}
            teams={teams}
            possibleUIRoles={possibleUIRoles}
            onUserCreated={handleUserCreated}
            isEmbedded={true}
          />
        </Modal>
      )}

      {apiKey && (
        <Modal open={isModalVisible} onOk={handleOk} onCancel={handleCancel} footer={null}>
          <Grid numItems={1} className="gap-2 w-full">
            <Title>{t("virtualKeys.createKey.saveKey")}</Title>
            <Col numColSpan={1}>
              {apiKey != null ? (
                <CreatedKeyDisplay apiKey={apiKey} />
              ) : (
                <Text>{t("virtualKeys.createKey.creating")}</Text>
              )}
            </Col>
          </Grid>
        </Modal>
      )}
    </div>
  );
};

export default CreateKey;
