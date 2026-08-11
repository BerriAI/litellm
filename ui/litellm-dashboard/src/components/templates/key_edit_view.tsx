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
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import { mapInternalToDisplayNames } from "../callback_info_helpers";
import KeyLifecycleSettings from "../common_components/KeyLifecycleSettings";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";
import OrganizationDropdown from "../common_components/OrganizationDropdown";
import { extractLoggingSettings, formatMetadataForDisplay, stripTagsFromMetadata } from "../key_info_utils";
import { BudgetFallbacksEditor } from "../key_team_helpers/BudgetFallbacksEditor";
import { BudgetWindowEntry, BudgetWindowsEditor } from "../key_team_helpers/BudgetWindowsEditor";
import {
  TagRateLimitEditor,
  TagRateLimitEntry,
  tagLimitsToRows,
  tagRowsToLimits,
} from "../key_team_helpers/TagRateLimitEditor";
import { excludeProxyWideSentinel, hasAllModelsSentinel } from "../key_team_helpers/fetch_available_models_team_key";
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
  const { t } = useTranslation("gateway");
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
  const [tagRateLimits, setTagRateLimits] = useState<TagRateLimitEntry[]>(
    tagLimitsToRows(keyData.metadata?.tag_rpm_limit),
  );
  const [budgetFallbacks, setBudgetFallbacks] = useState<Record<string, string[]>>(
    keyData.budget_fallbacks && typeof keyData.budget_fallbacks === "object" ? keyData.budget_fallbacks : {},
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
          setAvailableModels(excludeProxyWideSentinel(available_model_names));
        } else if (team?.team_id) {
          // Fetch team models if team exists
          const models = await fetchTeamModels(userID, userRole, accessToken, team.team_id);
          setAvailableModels(excludeProxyWideSentinel(Array.from(new Set([...team.models, ...models]))));
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

  // Normalize any legacy word-form budget duration to the canonical value the dropdown uses
  const getBudgetDuration = (duration: string | null) => {
    if (!duration) return null;
    const wordToCanonical: Record<string, string> = {
      hourly: "1h",
      daily: "24h",
      weekly: "7d",
      monthly: "30d",
    };
    return wordToCanonical[duration] ?? duration;
  };

  // Set initial form values
  const initialValues = {
    ...keyData,
    token: keyData.token || keyData.token_id,
    budget_duration: getBudgetDuration(keyData.budget_duration),
    metadata: formatMetadataForDisplay(stripTagsFromMetadata(keyData.metadata)),
    guardrails: keyData.metadata?.guardrails,
    disable_global_guardrails: keyData.metadata?.disable_global_guardrails || false,
    throttle_on_budget_exceeded: keyData.metadata?.throttle_on_budget_exceeded || false,
    prompts: keyData.metadata?.prompts,
    tags: keyData.metadata?.tags,
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
        toolsets: keyData.object_permission?.mcp_toolsets || [],
      },
      mcp_tool_permissions: keyData.object_permission?.mcp_tool_permissions || {},
      throttle_on_budget_exceeded: keyData.metadata?.throttle_on_budget_exceeded || false,
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
        NotificationsManager.fromBackend(t("virtualKeys.edit.tagFetchFailed", { error: String(error) }));
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

      if (keyData.budget_duration && !values.budget_duration) {
        values.budget_duration = null;
      }

      // Reconcile multi-window budget limits from the editor state, dropping
      // incomplete entries (no max_budget). The backend treats any budget_limits
      // in a /key/update request as an admin-only budget change, so re-sending
      // the stored windows on an unrelated edit 403s a non-admin key owner
      // (issue #33246). Only send the field when the user actually changed the
      // windows, mirroring how allowed_routes is dropped above when unchanged:
      // compare on (duration, cap), ignoring server-owned reset_at and order.
      // Sending [] clears every window, so send it only when the user removed
      // the last one; otherwise leave the field off (JSON.stringify drops the
      // undefined key) so an unchanged or incomplete editor state never touches
      // storage.
      const windowSignature = (windows: Array<{ budget_duration: string; max_budget: number | null }> | undefined) =>
        (windows ?? [])
          .filter((w) => w.budget_duration && w.max_budget !== null && w.max_budget !== undefined)
          .map((w) => `${w.budget_duration}:${w.max_budget}`)
          .sort()
          .join("|");
      const validWindows = budgetLimits.filter(
        (w) => w.budget_duration && w.max_budget !== null && w.max_budget !== undefined,
      );
      const budgetLimitsUnchanged = windowSignature(keyData.budget_limits) === windowSignature(validWindows);
      if (budgetLimitsUnchanged) {
        // no-op: leave budget_limits off the payload
      } else if (validWindows.length > 0) {
        values.budget_limits = validWindows;
      } else if (budgetLimits.length === 0) {
        values.budget_limits = [];
      }

      // Always send the current per-tag limit map so removing every row
      // clears the stored limits ({} overwrites the metadata field).
      const { tag_rpm_limit } = tagRowsToLimits(tagRateLimits);
      values.tag_rpm_limit = tag_rpm_limit;

      const hadExistingFallbacks = keyData.budget_fallbacks != null && Object.keys(keyData.budget_fallbacks).length > 0;
      if (Object.keys(budgetFallbacks).length > 0) {
        values.budget_fallbacks = budgetFallbacks;
      } else if (hadExistingFallbacks) {
        values.budget_fallbacks = {};
      }

      await onSubmit(values);
    } finally {
      setIsKeySaving(false);
    }
  };

  return (
    <Form form={form} onFinish={handleSubmit} initialValues={initialValues} layout="vertical">
      <Form.Item label={t("virtualKeys.edit.keyAlias")} name="key_alias">
        <TextInput />
      </Form.Item>

      <Form.Item label={t("virtualKeys.createKey.models")} name="models">
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
                  placeholder={t("virtualKeys.createKey.selectModels")}
                  style={{ width: "100%" }}
                  disabled={isDisabled}
                  value={isDisabled ? [] : models}
                  onChange={(value) => {
                    if (value.includes("all-team-models")) {
                      setFieldValue("models", ["all-team-models"]);
                    } else if (value.includes("all-proxy-models")) {
                      setFieldValue("models", ["all-proxy-models"]);
                    } else {
                      setFieldValue("models", value);
                    }
                  }}
                >
                  {keyData.team_id != null ? (
                    team != null && (
                      <Select.Option value="all-team-models">{t("virtualKeys.createKey.allTeamModels")}</Select.Option>
                    )
                  ) : (
                    <Select.Option value="all-proxy-models">{t("virtualKeys.createKey.allProxyModels")}</Select.Option>
                  )}
                  {availableModels.map((model) => (
                    <Select.Option key={model} value={model} disabled={hasAllModelsSentinel(models)}>
                      {model}
                    </Select.Option>
                  ))}
                </Select>
                {isDisabled && (
                  <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                    {t("virtualKeys.createKey.modelsDisabled")}
                  </div>
                )}
              </>
            );
          }}
        </Form.Item>
      </Form.Item>

      <Form.Item label={t("virtualKeys.createKey.keyType")}>
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
                placeholder={t("virtualKeys.createKey.selectKeyType")}
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
                <Select.Option value="default" label={t("virtualKeys.createKey.fullAccess")}>
                  <div style={{ padding: "4px 0" }}>
                    <div style={{ fontWeight: 500 }}>{t("virtualKeys.createKey.fullAccess")}</div>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                      {t("virtualKeys.createKey.fullAccessDescription")}
                    </div>
                  </div>
                </Select.Option>
                <Select.Option value="llm_api" label={t("virtualKeys.createKey.aiApis")}>
                  <div style={{ padding: "4px 0" }}>
                    <div style={{ fontWeight: 500 }}>{t("virtualKeys.createKey.aiApis")}</div>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                      {t("virtualKeys.createKey.aiApisDescription")}
                    </div>
                  </div>
                </Select.Option>
                <Select.Option value="management" label={t("virtualKeys.createKey.management")}>
                  <div style={{ padding: "4px 0" }}>
                    <div style={{ fontWeight: 500 }}>{t("virtualKeys.createKey.management")}</div>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                      {t("virtualKeys.createKey.managementDescription")}
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
            {t("virtualKeys.edit.allowedRoutes")}{" "}
            <Tooltip title={t("virtualKeys.edit.allowedRoutesTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="allowed_routes"
      >
        <Input placeholder={t("virtualKeys.edit.allowedRoutesPlaceholder")} />
      </Form.Item>

      <Form.Item label={t("virtualKeys.createKey.optional.maxBudget")} name="max_budget">
        <NumericalInput
          step={0.01}
          style={{ width: "100%" }}
          placeholder={t("virtualKeys.createKey.optional.numericInput")}
        />
      </Form.Item>

      <Form.Item label={t("virtualKeys.createKey.optional.resetBudget")} name="budget_duration">
        <BudgetDurationDropdown placeholder={t("virtualKeys.createKey.optional.neverResets")} />
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("virtualKeys.createKey.optional.budgetWindows")}{" "}
            <Tooltip title={t("virtualKeys.edit.budgetWindowsTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
      >
        <BudgetWindowsEditor value={budgetLimits} onChange={setBudgetLimits} />
      </Form.Item>

      <Form.Item
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
          value={budgetFallbacks}
          onChange={setBudgetFallbacks}
          availableModels={availableModels}
        />
      </Form.Item>

      <Form.Item label={t("virtualKeys.edit.tpmLimit")} name="tpm_limit">
        <NumericalInput min={0} />
      </Form.Item>

      <RateLimitTypeFormItem type="tpm" name="tpm_limit_type" showDetailedDescriptions={false} />

      <Form.Item label={t("virtualKeys.edit.rpmLimit")} name="rpm_limit">
        <NumericalInput min={0} />
      </Form.Item>

      <RateLimitTypeFormItem type="rpm" name="rpm_limit_type" showDetailedDescriptions={false} />

      <Form.Item
        label={
          <span>
            {t("virtualKeys.createKey.optional.throttle")}{" "}
            <Tooltip title={t("virtualKeys.edit.throttleTooltip")}>
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

      <Form.Item label={t("virtualKeys.createKey.optional.advancedMaxParallelRequests")} name="max_parallel_requests">
        <NumericalInput min={0} />
      </Form.Item>

      <Form.Item label={t("virtualKeys.edit.modelTpmLimit")} name="model_tpm_limit">
        <Input.TextArea rows={4} placeholder='{"gpt-4": 100, "claude-v1": 200}' />
      </Form.Item>

      <Form.Item label={t("virtualKeys.edit.modelRpmLimit")} name="model_rpm_limit">
        <Input.TextArea rows={4} placeholder='{"gpt-4": 100, "claude-v1": 200}' />
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("virtualKeys.createKey.optional.perTagLimits")}{" "}
            <Tooltip title={t("virtualKeys.edit.perTagLimitsTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
      >
        <TagRateLimitEditor value={tagRateLimits} onChange={setTagRateLimits} />
      </Form.Item>

      <Form.Item label={t("virtualKeys.createKey.optional.guardrails")} name="guardrails">
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
            {t("virtualKeys.createKey.optional.disableGlobalGuardrails")}{" "}
            <Tooltip title={t("virtualKeys.edit.disableGlobalGuardrailsTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="disable_global_guardrails"
        valuePropName="checked"
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

      <Form.Item label={t("virtualKeys.createKey.optional.tags")} name="tags">
        <Select
          mode="tags"
          style={{ width: "100%" }}
          placeholder={t("virtualKeys.createKey.optional.selectTags")}
          options={Object.values(tagsList).map((tag) => ({
            value: tag.name,
            label: tag.name,
            title: tag.description || tag.name,
          }))}
        />
      </Form.Item>

      <Form.Item label={t("virtualKeys.createKey.optional.prompts")} name="prompts">
        <Tooltip title={!premiumUser ? t("virtualKeys.edit.promptsPremiumTooltip") : ""} placement="top">
          <Select
            mode="tags"
            style={{ width: "100%" }}
            disabled={!premiumUser}
            placeholder={
              !premiumUser
                ? t("virtualKeys.createKey.optional.promptsPremium")
                : Array.isArray(keyData.metadata?.prompts) && keyData.metadata.prompts.length > 0
                  ? t("virtualKeys.edit.currentValues", { values: keyData.metadata.prompts.join(", ") })
                  : t("virtualKeys.createKey.optional.selectPrompts")
            }
            options={promptsList.map((name) => ({ value: name, label: name }))}
          />
        </Tooltip>
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
      >
        <AccessGroupSelector placeholder={t("virtualKeys.createKey.optional.selectAccessGroups")} />
      </Form.Item>

      <Form.Item
        label={t("virtualKeys.createKey.optional.passThroughRoutes")}
        name="allowed_passthrough_routes"
        tooltip={!premiumUser ? t("virtualKeys.edit.passThroughPremiumTooltip") : undefined}
      >
        <PassThroughRoutesSelector
          accessToken={accessToken || ""}
          placeholder={
            !premiumUser
              ? t("virtualKeys.createKey.optional.passThroughRoutesPremium")
              : Array.isArray(keyData.metadata?.allowed_passthrough_routes) &&
                  keyData.metadata.allowed_passthrough_routes.length > 0
                ? t("virtualKeys.edit.currentValues", {
                    values: keyData.metadata.allowed_passthrough_routes.join(", "),
                  })
                : t("virtualKeys.createKey.optional.selectPassThroughRoutes")
          }
          disabled={!premiumUser}
        />
      </Form.Item>

      <Form.Item label={t("virtualKeys.createKey.optional.vectorStores")} name="vector_stores">
        <VectorStoreSelector
          onChange={(values: string[]) => form.setFieldValue("vector_stores", values)}
          value={form.getFieldValue("vector_stores")}
          accessToken={accessToken || ""}
          placeholder={t("virtualKeys.createKey.optional.selectVectorStores")}
        />
      </Form.Item>

      <Form.Item label={t("virtualKeys.edit.mcpServersAndGroups")} name="mcp_servers_and_groups">
        <MCPServerSelector
          onChange={(val) => form.setFieldValue("mcp_servers_and_groups", val)}
          value={form.getFieldValue("mcp_servers_and_groups")}
          accessToken={accessToken || ""}
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

      <Form.Item label={t("virtualKeys.edit.agentsAndGroups")} name="agents_and_groups">
        <AgentSelector
          onChange={(val) => form.setFieldValue("agents_and_groups", val)}
          value={form.getFieldValue("agents_and_groups")}
          accessToken={accessToken || ""}
          placeholder={t("virtualKeys.createKey.optional.selectAgents")}
        />
      </Form.Item>

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
        label={t("virtualKeys.edit.teamId")}
        name="team_id"
        help={enableProjectsUI && hasProject ? t("virtualKeys.edit.teamLockedByProject") : undefined}
      >
        <Select
          placeholder={t("virtualKeys.edit.selectTeam")}
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
        <Form.Item label={t("virtualKeys.createKey.project")}>
          <Input value={projectDisplay ?? ""} disabled />
        </Form.Item>
      )}
      <Form.Item label={t("virtualKeys.createKey.optional.loggingSettings")} name="logging_settings">
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

      <Form.Item label={t("virtualKeys.createKey.optional.metadata")} name="metadata">
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

      <div className="sticky z-10 bg-white p-4 border-t border-gray-200 -bottom-6 -inset-x-6">
        <div className="flex justify-end items-center gap-2">
          <TremorButton variant="secondary" onClick={onCancel} disabled={isKeySaving}>
            {t("virtualKeys.edit.cancel")}
          </TremorButton>
          <TremorButton type="submit" loading={isKeySaving}>
            {t("virtualKeys.edit.saveChanges")}
          </TremorButton>
        </div>
      </div>
    </Form>
  );
}
