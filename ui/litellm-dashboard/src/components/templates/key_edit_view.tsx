import GuardrailSelector from "@/components/guardrails/GuardrailSelector";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import PolicySelector from "@/components/policies/PolicySelector";
import { InfoCircleOutlined } from "@ant-design/icons";
import { TextInput, Button as TremorButton } from "@tremor/react";
import { Form, Input, Select, Switch, Tooltip } from "antd";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { rolesWithWriteAccess } from "../../utils/roles";
import AgentSelector from "../agent_management/AgentSelector";
import AccessGroupSelector from "../common_components/AccessGroupSelector";
import { mapInternalToDisplayNames } from "../callback_info_helpers";
import KeyLifecycleSettings from "../common_components/KeyLifecycleSettings";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";
import OrganizationDropdown from "../common_components/OrganizationDropdown";
import { extractLoggingSettings, formatMetadataForDisplay, stripTagsFromMetadata } from "../key_info_utils";
import { BudgetWindowEntry, BudgetWindowsEditor } from "../key_team_helpers/BudgetWindowsEditor";
import { KeyResponse } from "../key_team_helpers/key_list";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import { NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import NotificationsManager from "../molecules/notifications_manager";
import { getPromptsList, modelAvailableCall, tagListCall } from "../networking";
import { fetchTeamModels } from "../organisms/create_key_button";
import NumericalInput from "../shared/numerical_input";
import { Tag } from "../tag_management/types";
import EditLoggingSettings from "../team/EditLoggingSettings";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";

interface KeyEditViewProps {
  keyData: KeyResponse;
  onCancel: () => void;
  onSubmit: (values: any) => Promise<void>;
  teams?: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  premiumUser?: boolean;
}

// Add this helper function
const getAvailableModelsForKey = (keyData: KeyResponse, teams: any[] | null): string[] => {
  // If no teams data is available, return empty array
  if (!teams || !keyData.team_id) {
    return [];
  }

  // Find the team that matches the key's team_id
  const keyTeam = teams.find((team) => team.team_id === keyData.team_id);

  // If team found and has models, return those models
  if (keyTeam?.models) {
    return keyTeam.models;
  }

  return [];
};

// Helper function to determine key_type display value from allowed_routes
const getKeyTypeFromRoutes = (allowedRoutes: string[] | null | undefined): string => {
  if (!allowedRoutes || allowedRoutes.length === 0) {
    return "default";
  }

  if (allowedRoutes.includes("llm_api_routes")) {
    return "llm_api";
  }

  if (allowedRoutes.includes("management_routes")) {
    return "management";
  }

  if (allowedRoutes.includes("info_routes")) {
    return "read_only";
  }

  return "default";
};

export function KeyEditView({
  keyData,
  onCancel,
  onSubmit,
  teams,
  accessToken,
  userID,
  userRole,
  premiumUser = false,
}: KeyEditViewProps) {
  const { t } = useTranslation();
  const canEditGuardrails = premiumUser || (userRole != null && rolesWithWriteAccess.includes(userRole));
  const [form] = Form.useForm();
  const [promptsList, setPromptsList] = useState<string[]>([]);
  const [tagsList, setTagsList] = useState<Record<string, Tag>>({});
  const team = teams?.find((team) => team.team_id === keyData.team_id);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [disabledCallbacks, setDisabledCallbacks] = useState<string[]>(
    Array.isArray(keyData.metadata?.litellm_disabled_callbacks)
      ? mapInternalToDisplayNames(keyData.metadata.litellm_disabled_callbacks)
      : [],
  );
  const [selectedOrganizationId, setSelectedOrganizationId] = useState<string | null>(keyData.organization_id || null);
  const [autoRotationEnabled, setAutoRotationEnabled] = useState<boolean>(keyData.auto_rotate || false);
  const [rotationInterval, setRotationInterval] = useState<string>(keyData.rotation_interval || "");
  const [neverExpire, setNeverExpire] = useState<boolean>(!keyData.expires);
  const [isKeySaving, setIsKeySaving] = useState(false);
  const [budgetLimits, setBudgetLimits] = useState<BudgetWindowEntry[]>(
    Array.isArray(keyData.budget_limits) ? keyData.budget_limits : [],
  );
  const { data: organizations, isLoading: isOrganizationsLoading } = useOrganizations();
  const { data: projects } = useProjects();
  const { data: uiSettingsData } = useUISettings();
  const enableProjectsUI = Boolean(uiSettingsData?.values?.enable_projects_ui);
  const hasProject = Boolean(keyData.project_id);
  const projectDisplay = (() => {
    if (!keyData.project_id) return null;
    const project = projects?.find((p) => p.project_id === keyData.project_id);
    return project?.project_alias ? `${project.project_alias} (${keyData.project_id})` : keyData.project_id;
  })();

  useEffect(() => {
    const fetchModels = async () => {
      if (!userID || !userRole || !accessToken) return;

      try {
        if (keyData.team_id === null) {
          // Fetch user models if no team
          const model_available = await modelAvailableCall(accessToken, userID, userRole);
          const available_model_names = model_available["data"].map((element: { id: string }) => element.id);
          setAvailableModels(available_model_names);
        } else if (team?.team_id) {
          // Fetch team models if team exists
          const models = await fetchTeamModels(userID, userRole, accessToken, team.team_id);
          setAvailableModels(Array.from(new Set([...team.models, ...models])));
        }
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    const fetchPrompts = async () => {
      if (!accessToken) return;
      try {
        const response = await getPromptsList(accessToken);
        setPromptsList(response.prompts.map((prompt) => prompt.prompt_id));
      } catch (error) {
        console.error("Failed to fetch prompts:", error);
      }
    };

    fetchPrompts();
    fetchModels();
  }, [userID, userRole, accessToken, team, keyData.team_id]);

  // Sync disabled callbacks with form when component mounts
  useEffect(() => {
    form.setFieldValue("disabled_callbacks", disabledCallbacks);
  }, [form, disabledCallbacks]);

  // Convert API budget duration to form format
  const getBudgetDuration = (duration: string | null) => {
    if (!duration) return null;
    const durationMap: Record<string, string> = {
      "24h": "daily",
      "7d": "weekly",
      "30d": "monthly",
    };
    return durationMap[duration] || null;
  };

  // Set initial form values
  const initialValues = {
    ...keyData,
    token: keyData.token || keyData.token_id,
    budget_duration: getBudgetDuration(keyData.budget_duration),
    metadata: formatMetadataForDisplay(stripTagsFromMetadata(keyData.metadata)),
    guardrails: keyData.metadata?.guardrails,
    disable_global_guardrails: keyData.metadata?.disable_global_guardrails || false,
    prompts: keyData.metadata?.prompts,
    tags: keyData.metadata?.tags,
    vector_stores: keyData.object_permission?.vector_stores || [],
    mcp_servers_and_groups: {
      servers: keyData.object_permission?.mcp_servers || [],
      accessGroups: keyData.object_permission?.mcp_access_groups || [],
    },
    mcp_tool_permissions: keyData.object_permission?.mcp_tool_permissions || {},
    agents_and_groups: {
      agents: keyData.object_permission?.agents || [],
      accessGroups: keyData.object_permission?.agent_access_groups || [],
    },
    logging_settings: extractLoggingSettings(keyData.metadata),
    disabled_callbacks: Array.isArray(keyData.metadata?.litellm_disabled_callbacks)
      ? mapInternalToDisplayNames(keyData.metadata.litellm_disabled_callbacks)
      : [],
    access_group_ids: keyData.access_group_ids || [],
    auto_rotate: keyData.auto_rotate || false,
    ...(keyData.rotation_interval && { rotation_interval: keyData.rotation_interval }),
    allowed_routes:
      Array.isArray(keyData.allowed_routes) && keyData.allowed_routes.length > 0
        ? keyData.allowed_routes.join(", ")
        : "",
  };

  useEffect(() => {
    form.setFieldsValue({
      ...keyData,
      token: keyData.token || keyData.token_id,
      budget_duration: getBudgetDuration(keyData.budget_duration),
      metadata: formatMetadataForDisplay(stripTagsFromMetadata(keyData.metadata)),
      guardrails: keyData.metadata?.guardrails,
      disable_global_guardrails: keyData.metadata?.disable_global_guardrails || false,
      prompts: keyData.metadata?.prompts,
      tags: keyData.metadata?.tags,
      vector_stores: keyData.object_permission?.vector_stores || [],
      mcp_servers_and_groups: {
        servers: keyData.object_permission?.mcp_servers || [],
        accessGroups: keyData.object_permission?.mcp_access_groups || [],
      },
      mcp_tool_permissions: keyData.object_permission?.mcp_tool_permissions || {},
      logging_settings: extractLoggingSettings(keyData.metadata),
      disabled_callbacks: Array.isArray(keyData.metadata?.litellm_disabled_callbacks)
        ? mapInternalToDisplayNames(keyData.metadata.litellm_disabled_callbacks)
        : [],
      access_group_ids: keyData.access_group_ids || [],
      auto_rotate: keyData.auto_rotate || false,
      ...(keyData.rotation_interval && { rotation_interval: keyData.rotation_interval }),
      allowed_routes:
        Array.isArray(keyData.allowed_routes) && keyData.allowed_routes.length > 0
          ? keyData.allowed_routes.join(", ")
          : "",
    });
  }, [keyData, form]);

  // Sync auto-rotation state with form values
  useEffect(() => {
    form.setFieldValue("auto_rotate", autoRotationEnabled);
  }, [autoRotationEnabled, form]);

  useEffect(() => {
    if (rotationInterval) {
      form.setFieldValue("rotation_interval", rotationInterval);
    }
  }, [rotationInterval, form]);

  // Fetch tags for selector
  useEffect(() => {
    const fetchTags = async () => {
      if (!accessToken) return;
      try {
        const response = await tagListCall(accessToken);
        setTagsList(response);
      } catch (error) {
        NotificationsManager.fromBackend(t("templates.keyEditView.errorFetchingTags", { error }));
      }
    };
    fetchTags();
  }, [accessToken]);

  const handleSubmit = async (values: any) => {
    try {
      setIsKeySaving(true);

      // Parse allowed_routes from comma-separated string to array
      if (typeof values.allowed_routes === "string") {
        const trimmedInput = values.allowed_routes.trim();
        if (trimmedInput === "") {
          values.allowed_routes = [];
        } else {
          values.allowed_routes = trimmedInput
            .split(",")
            .map((route: string) => route.trim())
            .filter((route: string) => route.length > 0);
        }
      }
      // If it's already an array (shouldn't happen, but handle it), keep as is

      // Backend rejects non-empty allowed_routes from non-admins, so re-sending
      // an unchanged value 403s a team admin. Set compare tolerates reorder.
      const originalRoutesSet = new Set<string>(Array.isArray(keyData.allowed_routes) ? keyData.allowed_routes : []);
      const submittedRoutesSet = new Set<string>(Array.isArray(values.allowed_routes) ? values.allowed_routes : []);
      const allowedRoutesUnchanged =
        originalRoutesSet.size === submittedRoutesSet.size &&
        [...submittedRoutesSet].every((r) => originalRoutesSet.has(r));
      if (allowedRoutesUnchanged) {
        delete values.allowed_routes;
      }

      if (neverExpire) {
        values.duration = null;
      }

      // Include multi-window budget limits (filter out incomplete entries)
      const validWindows = budgetLimits.filter(
        (w) => w.budget_duration && w.max_budget !== null && w.max_budget !== undefined,
      );
      values.budget_limits = validWindows.length > 0 ? validWindows : undefined;

      await onSubmit(values);
    } finally {
      setIsKeySaving(false);
    }
  };

  return (
    <Form form={form} onFinish={handleSubmit} initialValues={initialValues} layout="vertical">
      <Form.Item label={t("templates.keyEditView.keyAlias")} name="key_alias">
        <TextInput />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.models")} name="models">
        <Form.Item
          noStyle
          shouldUpdate={(prevValues, currentValues) =>
            prevValues.allowed_routes !== currentValues.allowed_routes || prevValues.models !== currentValues.models
          }
        >
          {({ getFieldValue, setFieldValue }) => {
            const allowedRoutesValue = getFieldValue("allowed_routes") || "";
            // Convert string to array for checking
            const allowedRoutes =
              typeof allowedRoutesValue === "string" && allowedRoutesValue.trim() !== ""
                ? allowedRoutesValue
                    .split(",")
                    .map((r: string) => r.trim())
                    .filter((r: string) => r.length > 0)
                : [];
            const isDisabled = allowedRoutes.includes("management_routes") || allowedRoutes.includes("info_routes");
            const models = getFieldValue("models") || [];

            return (
              <>
                <Select
                  mode="multiple"
                  placeholder={t("templates.keyEditView.selectModels")}
                  style={{ width: "100%" }}
                  disabled={isDisabled}
                  value={isDisabled ? [] : models}
                  onChange={(value) => setFieldValue("models", value)}
                >
                  {/* Only show All Team Models if team has models */}
                  {availableModels.length > 0 && (
                    <Select.Option value="all-team-models">{t("templates.keyEditView.allTeamModels")}</Select.Option>
                  )}
                  {availableModels.map((model) => (
                    <Select.Option key={model} value={model}>
                      {model}
                    </Select.Option>
                  ))}
                </Select>
                {isDisabled && (
                  <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                    {t("templates.keyEditView.modelsDisabled")}
                  </div>
                )}
              </>
            );
          }}
        </Form.Item>
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.keyType")}>
        <Form.Item
          noStyle
          shouldUpdate={(prevValues, currentValues) => prevValues.allowed_routes !== currentValues.allowed_routes}
        >
          {({ getFieldValue, setFieldValue }) => {
            const allowedRoutesValue = getFieldValue("allowed_routes") || "";
            // Convert string to array for getKeyTypeFromRoutes
            const allowedRoutes =
              typeof allowedRoutesValue === "string" && allowedRoutesValue.trim() !== ""
                ? allowedRoutesValue
                    .split(",")
                    .map((r: string) => r.trim())
                    .filter((r: string) => r.length > 0)
                : [];
            const keyTypeValue = getKeyTypeFromRoutes(allowedRoutes);

            return (
              <Select
                placeholder={t("templates.keyEditView.selectKeyType")}
                style={{ width: "100%" }}
                optionLabelProp="label"
                value={keyTypeValue}
                onChange={(value) => {
                  switch (value) {
                    case "default":
                      setFieldValue("allowed_routes", "");
                      break;
                    case "llm_api":
                      setFieldValue("allowed_routes", "llm_api_routes");
                      break;
                    case "management":
                      setFieldValue("allowed_routes", "management_routes");
                      setFieldValue("models", []);
                      break;
                  }
                }}
              >
                <Select.Option value="default" label={t("templates.keyEditView.fullAccess")}>
                  <div style={{ padding: "4px 0" }}>
                    <div style={{ fontWeight: 500 }}>{t("templates.keyEditView.fullAccess")}</div>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                      {t("templates.keyEditView.fullAccessDesc")}
                    </div>
                  </div>
                </Select.Option>
                <Select.Option value="llm_api" label={t("templates.keyEditView.aiApis")}>
                  <div style={{ padding: "4px 0" }}>
                    <div style={{ fontWeight: 500 }}>{t("templates.keyEditView.aiApis")}</div>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                      {t("templates.keyEditView.aiApisDesc")}
                    </div>
                  </div>
                </Select.Option>
                <Select.Option value="management" label={t("templates.keyEditView.management")}>
                  <div style={{ padding: "4px 0" }}>
                    <div style={{ fontWeight: 500 }}>{t("templates.keyEditView.management")}</div>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                      {t("templates.keyEditView.managementDesc")}
                    </div>
                  </div>
                </Select.Option>
              </Select>
            );
          }}
        </Form.Item>
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("templates.keyEditView.allowedRoutes")}{" "}
            <Tooltip title={t("templates.keyEditView.allowedRoutesTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="allowed_routes"
      >
        <Input placeholder={t("templates.keyEditView.allowedRoutesPlaceholder")} />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.maxBudget")} name="max_budget">
        <NumericalInput
          step={0.01}
          style={{ width: "100%" }}
          placeholder={t("templates.keyEditView.enterNumericalValue")}
        />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.resetBudget")} name="budget_duration">
        <Select placeholder="n/a">
          <Select.Option value="daily">{t("templates.keyEditView.daily")}</Select.Option>
          <Select.Option value="weekly">{t("templates.keyEditView.weekly")}</Select.Option>
          <Select.Option value="monthly">{t("templates.keyEditView.monthly")}</Select.Option>
        </Select>
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("templates.keyEditView.budgetWindows")}{" "}
            <Tooltip title={t("templates.keyEditView.budgetWindowsTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
      >
        <BudgetWindowsEditor value={budgetLimits} onChange={setBudgetLimits} />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.tpmLimit")} name="tpm_limit">
        <NumericalInput min={0} />
      </Form.Item>

      <RateLimitTypeFormItem type="tpm" name="tpm_limit_type" showDetailedDescriptions={false} />

      <Form.Item label={t("templates.keyEditView.rpmLimit")} name="rpm_limit">
        <NumericalInput min={0} />
      </Form.Item>

      <RateLimitTypeFormItem type="rpm" name="rpm_limit_type" showDetailedDescriptions={false} />

      <Form.Item label={t("templates.keyEditView.maxParallelRequests")} name="max_parallel_requests">
        <NumericalInput min={0} />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.modelTpmLimit")} name="model_tpm_limit">
        <Input.TextArea rows={4} placeholder='{"gpt-4": 100, "claude-v1": 200}' />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.modelRpmLimit")} name="model_rpm_limit">
        <Input.TextArea rows={4} placeholder='{"gpt-4": 100, "claude-v1": 200}' />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.guardrails")} name="guardrails">
        {accessToken && (
          <GuardrailSelector
            onChange={(v) => {
              form.setFieldValue("guardrails", v);
            }}
            accessToken={accessToken}
            disabled={!canEditGuardrails}
          />
        )}
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("templates.keyEditView.disableGlobalGuardrails")}{" "}
            <Tooltip title={t("templates.keyEditView.disableGlobalGuardrailsTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="disable_global_guardrails"
        valuePropName="checked"
      >
        <Switch disabled={!canEditGuardrails} checkedChildren={t("common.yes")} unCheckedChildren={t("common.no")} />
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("templates.keyEditView.policies")}{" "}
            <Tooltip title={t("templates.keyEditView.policiesTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="policies"
      >
        {accessToken && (
          <PolicySelector
            onChange={(v) => {
              form.setFieldValue("policies", v);
            }}
            accessToken={accessToken}
            disabled={!premiumUser}
          />
        )}
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.tags")} name="tags">
        <Select
          mode="tags"
          style={{ width: "100%" }}
          placeholder={t("templates.keyEditView.selectOrEnterTags")}
          options={Object.values(tagsList).map((tag) => ({
            value: tag.name,
            label: tag.name,
            title: tag.description || tag.name,
          }))}
        />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.prompts")} name="prompts">
        <Tooltip title={!premiumUser ? t("templates.keyEditView.promptsPremiumTooltip") : ""} placement="top">
          <Select
            mode="tags"
            style={{ width: "100%" }}
            disabled={!premiumUser}
            placeholder={
              !premiumUser
                ? t("templates.keyEditView.promptsPremiumPlaceholder")
                : Array.isArray(keyData.metadata?.prompts) && keyData.metadata.prompts.length > 0
                  ? t("templates.keyEditView.currentPrompts", { prompts: keyData.metadata.prompts.join(", ") })
                  : t("templates.keyEditView.selectOrEnterPrompts")
            }
            options={promptsList.map((name) => ({ value: name, label: name }))}
          />
        </Tooltip>
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("templates.keyEditView.accessGroups")}{" "}
            <Tooltip title={t("templates.keyEditView.accessGroupsTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="access_group_ids"
      >
        <AccessGroupSelector placeholder={t("templates.keyEditView.selectAccessGroups")} />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.allowedPassThroughRoutes")} name="allowed_passthrough_routes">
        <Tooltip title={!premiumUser ? t("templates.keyEditView.passThroughPremiumTooltip") : ""} placement="top">
          <PassThroughRoutesSelector
            onChange={(values: string[]) => form.setFieldValue("allowed_passthrough_routes", values)}
            value={form.getFieldValue("allowed_passthrough_routes")}
            accessToken={accessToken || ""}
            placeholder={
              !premiumUser
                ? t("templates.keyEditView.passThroughPremiumPlaceholder")
                : Array.isArray(keyData.metadata?.allowed_passthrough_routes) &&
                    keyData.metadata.allowed_passthrough_routes.length > 0
                  ? t("templates.keyEditView.currentPassThroughRoutes", {
                      routes: keyData.metadata.allowed_passthrough_routes.join(", "),
                    })
                  : t("templates.keyEditView.selectOrEnterPassThroughRoutes")
            }
            disabled={!premiumUser}
          />
        </Tooltip>
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.vectorStores")} name="vector_stores">
        <VectorStoreSelector
          onChange={(values: string[]) => form.setFieldValue("vector_stores", values)}
          value={form.getFieldValue("vector_stores")}
          accessToken={accessToken || ""}
          placeholder={t("templates.keyEditView.selectVectorStores")}
        />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.mcpServers")} name="mcp_servers_and_groups">
        <MCPServerSelector
          onChange={(val) => form.setFieldValue("mcp_servers_and_groups", val)}
          value={form.getFieldValue("mcp_servers_and_groups")}
          accessToken={accessToken || ""}
          placeholder={t("templates.keyEditView.selectMcpServers")}
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
          prevValues.mcp_servers_and_groups !== currentValues.mcp_servers_and_groups ||
          prevValues.mcp_tool_permissions !== currentValues.mcp_tool_permissions
        }
      >
        {() => (
          <div className="mb-6">
            <MCPToolPermissions
              accessToken={accessToken || ""}
              selectedServers={(form.getFieldValue("mcp_servers_and_groups")?.servers || []).filter(
                (s: string) => s !== NO_MCP_SERVERS_SENTINEL,
              )}
              toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
              onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
            />
          </div>
        )}
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.agents")} name="agents_and_groups">
        <AgentSelector
          onChange={(val) => form.setFieldValue("agents_and_groups", val)}
          value={form.getFieldValue("agents_and_groups")}
          accessToken={accessToken || ""}
          placeholder={t("templates.keyEditView.selectAgents")}
        />
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("templates.keyEditView.organization")}{" "}
            <Tooltip title={t("templates.keyEditView.organizationTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="organization_id"
      >
        <OrganizationDropdown
          organizations={organizations}
          loading={isOrganizationsLoading}
          disabled={userRole !== "Admin"}
          onChange={(orgId) => {
            setSelectedOrganizationId(orgId || null);
            form.setFieldValue("team_id", undefined);
          }}
        />
      </Form.Item>

      <Form.Item
        label={t("templates.keyEditView.teamId")}
        name="team_id"
        help={enableProjectsUI && hasProject ? t("templates.keyEditView.teamLockedByProject") : undefined}
      >
        <Select
          placeholder={t("templates.keyEditView.selectTeam")}
          showSearch
          disabled={enableProjectsUI && hasProject}
          style={{ width: "100%" }}
          onChange={(teamId) => {
            const selectedTeam = teams?.find((t) => t.team_id === teamId) || null;
            if (selectedTeam?.organization_id) {
              setSelectedOrganizationId(selectedTeam.organization_id);
              form.setFieldValue("organization_id", selectedTeam.organization_id);
            } else if (!teamId) {
              setSelectedOrganizationId(null);
              form.setFieldValue("organization_id", undefined);
            }
          }}
          filterOption={(input, option) => {
            const filteredTeams = selectedOrganizationId
              ? teams?.filter((t) => t.organization_id === selectedOrganizationId)
              : teams;
            const team = filteredTeams?.find((t) => t.team_id === option?.value);
            if (!team) return false;
            return team.team_alias?.toLowerCase().includes(input.toLowerCase()) ?? false;
          }}
        >
          {(selectedOrganizationId ? teams?.filter((t) => t.organization_id === selectedOrganizationId) : teams)?.map(
            (team) => (
              <Select.Option key={team.team_id} value={team.team_id}>
                {`${team.team_alias} (${team.team_id})`}
              </Select.Option>
            ),
          )}
        </Select>
      </Form.Item>
      {enableProjectsUI && hasProject && (
        <Form.Item label={t("templates.keyEditView.project")}>
          <Input value={projectDisplay ?? ""} disabled />
        </Form.Item>
      )}
      <Form.Item label={t("templates.keyEditView.loggingSettings")} name="logging_settings">
        <EditLoggingSettings
          value={form.getFieldValue("logging_settings")}
          onChange={(values) => form.setFieldValue("logging_settings", values)}
          disabledCallbacks={disabledCallbacks}
          onDisabledCallbacksChange={(internalValues) => {
            // Convert internal values back to display names for UI state
            const displayNames = mapInternalToDisplayNames(internalValues);
            setDisabledCallbacks(displayNames);
            // Store internal values in form for submission
            form.setFieldValue("disabled_callbacks", internalValues);
          }}
        />
      </Form.Item>

      <Form.Item label={t("templates.keyEditView.metadata")} name="metadata">
        <Input.TextArea rows={10} />
      </Form.Item>

      {/* Auto-Rotation Settings */}
      <div className="mb-4">
        <KeyLifecycleSettings
          form={form}
          autoRotationEnabled={autoRotationEnabled}
          onAutoRotationChange={setAutoRotationEnabled}
          rotationInterval={rotationInterval}
          onRotationIntervalChange={setRotationInterval}
          neverExpire={neverExpire}
          onNeverExpireChange={setNeverExpire}
        />
        <Form.Item name="duration" hidden initialValue="">
          <Input />
        </Form.Item>
      </div>

      {/* Hidden form field for token */}
      <Form.Item name="token" hidden>
        <Input />
      </Form.Item>

      {/* Hidden form field for disabled callbacks */}
      <Form.Item name="disabled_callbacks" hidden>
        <Input />
      </Form.Item>

      {/* Hidden form fields for auto-rotation */}
      <Form.Item name="auto_rotate" hidden>
        <Input />
      </Form.Item>
      <Form.Item name="rotation_interval" hidden>
        <Input />
      </Form.Item>

      <div className="sticky z-10 bg-white p-4 border-t border-gray-200 bottom-[-1.5rem] inset-x-[-1.5rem]">
        <div className="flex justify-end items-center gap-2">
          <TremorButton variant="secondary" onClick={onCancel} disabled={isKeySaving}>
            {t("common.cancel")}
          </TremorButton>
          <TremorButton type="submit" loading={isKeySaving}>
            {t("templates.keyEditView.saveChanges")}
          </TremorButton>
        </div>
      </div>
    </Form>
  );
}
