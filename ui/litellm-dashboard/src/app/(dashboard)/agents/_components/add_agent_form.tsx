import React, { useState, useEffect } from "react";
import { Modal, Form, Select, Input, Steps, Radio, Tag, Divider, Switch, InputNumber } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { Logo } from "@/components/molecules/logo/Logo";
import { Button } from "@tremor/react";
import { CheckCircleFilled, KeyOutlined, RobotOutlined, AppstoreOutlined, InfoCircleOutlined } from "@ant-design/icons";
import CreatedKeyDisplay from "@/components/shared/CreatedKeyDisplay";
import {
  createAgentCall,
  getAgentCreateMetadata,
  getAgentsList,
  keyCreateForAgentCall,
  keyListCall,
  keyUpdateCall,
  modelAvailableCall,
  AgentCreateInfo,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { Team } from "@/components/key_team_helpers/key_list";
import TeamDropdown from "@/components/common_components/team_dropdown";
import AgentFormFields from "./agent_form_fields";
import AgentCardDiscovery, { DiscoveredAgentCardSelection } from "./agent_card_discovery";
import { buildDiscoveryRequest, overlayDiscoveredCardParams } from "./agent_discovery_utils";
import DynamicAgentFormFields, { buildDynamicAgentData } from "./dynamic_agent_form_fields";
import { getDefaultFormValues, buildAgentDataFromForm } from "./agent_config";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "@/components/mcp_server_management/MCPToolPermissions";
import GuardrailSelector from "@/components/guardrails/GuardrailSelector";
import { useTranslation } from "react-i18next";

const { Step } = Steps;

const CUSTOM_AGENT_TYPE = "custom";

interface AddAgentFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
  teams?: Team[] | null;
}

const AddAgentForm: React.FC<AddAgentFormProps> = ({ visible, onClose, accessToken, onSuccess, teams }) => {
  const { t } = useTranslation("gateway");
  const { userId, userRole } = useAuthorized();
  const [form] = Form.useForm();
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [agentType, setAgentType] = useState<string>("a2a");
  const [agentTypeMetadata, setAgentTypeMetadata] = useState<AgentCreateInfo[]>([]);

  // Step 3: key assignment state
  const [keyAssignOption, setKeyAssignOption] = useState<"create_new" | "existing_key" | "skip">("create_new");
  const [newKeyName, setNewKeyName] = useState<string>("");
  const [newKeyModels, setNewKeyModels] = useState<string[]>([]);
  const [existingKeys, setExistingKeys] = useState<any[]>([]);
  const [selectedExistingKey, setSelectedExistingKey] = useState<string | null>(null);
  const [loadingKeys, setLoadingKeys] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [availableAgents, setAvailableAgents] = useState<{ agent_id: string; agent_name: string }[]>([]);
  const [loadingAgents, setLoadingAgents] = useState(false);

  // Step 4: results
  const [createdAgentName, setCreatedAgentName] = useState<string>("");
  const [createdKeyValue, setCreatedKeyValue] = useState<string | null>(null);
  const [assignedKeyAlias, setAssignedKeyAlias] = useState<string | null>(null);

  // Tracing & guardrails state
  const [requireTraceIdInbound, setRequireTraceIdInbound] = useState(false);
  const [requireTraceIdOutbound, setRequireTraceIdOutbound] = useState(false);
  const [maxIterations, setMaxIterations] = useState<number | null>(null);
  const [maxBudgetPerSession, setMaxBudgetPerSession] = useState<number | null>(null);

  // Latest upstream card selection from auto-discovery (skills, capabilities,
  // name, description). Dynamic agent forms don't render Form.Items for those
  // fields, so we overlay this onto agent_card_params at submit.
  const [appliedDiscoveredSelection, setAppliedDiscoveredSelection] = useState<DiscoveredAgentCardSelection | null>(
    null,
  );

  // Fetch agent type metadata on mount
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const metadata = await getAgentCreateMetadata();
        setAgentTypeMetadata(metadata);
      } catch (error) {
        console.error("Error fetching agent metadata:", error);
      }
    };
    fetchMetadata();
  }, []);

  // Fetch existing keys when Agent Management step becomes active (step 3)
  useEffect(() => {
    if (currentStep === 3 && accessToken && existingKeys.length === 0) {
      const fetchKeys = async () => {
        setLoadingKeys(true);
        try {
          const result = await keyListCall(accessToken, null, null, null, null, null, 1, 100);
          setExistingKeys(result?.keys || []);
        } catch (error) {
          console.error("Error fetching keys:", error);
        } finally {
          setLoadingKeys(false);
        }
      };
      fetchKeys();
    }
  }, [currentStep, accessToken]);

  // Fetch available models when Agent Management step is active (same list as key generation)
  useEffect(() => {
    if ((currentStep !== 1 && currentStep !== 3) || !accessToken || !userId || !userRole) return;
    let cancelled = false;
    setLoadingModels(true);
    modelAvailableCall(accessToken, userId, userRole)
      .then((response) => {
        if (cancelled) return;
        const modelsArray = response?.data ?? (Array.isArray(response) ? response : []);
        const ids = modelsArray
          .map((m: { id?: string; model_name?: string }) => m.id ?? m.model_name)
          .filter(Boolean) as string[];
        setAvailableModels(ids);
      })
      .catch((error) => {
        if (!cancelled) console.error("Error fetching models:", error);
      })
      .finally(() => {
        if (!cancelled) setLoadingModels(false);
      });
    return () => {
      cancelled = true;
    };
  }, [currentStep, accessToken, userId, userRole]);

  useEffect(() => {
    if (currentStep !== 1 || !accessToken) return;
    let cancelled = false;
    setLoadingAgents(true);
    getAgentsList(accessToken)
      .then((response) => {
        if (cancelled) return;
        const agents = response?.agents ?? [];
        setAvailableAgents(agents.map((a: any) => ({ agent_id: a.agent_id, agent_name: a.agent_name })));
      })
      .catch((error) => {
        if (!cancelled) console.error("Error fetching agents:", error);
      })
      .finally(() => {
        if (!cancelled) setLoadingAgents(false);
      });
    return () => {
      cancelled = true;
    };
  }, [currentStep, accessToken]);

  const selectedAgentTypeInfo = agentTypeMetadata.find((info) => info.agent_type === agentType);

  // Watch every form field so we can recompute the discovery plan whenever
  // the user types into a relevant credential field below.
  const watchedFormValues = Form.useWatch([], form);

  // Build the discovery plan for the proxy. Different agent runtimes publish
  // their cards at different URL shapes:
  //
  //   - LangGraph Platform: one well-known endpoint on the base URL,
  //     ``?assistant_id=<id>`` selects the assistant.
  //   - Pure A2A (the default): card lives at one of the well-known paths
  //     on the agent's own base URL.
  //
  // Returns undefined when nothing usable is filled in yet, which causes the
  // component to fall back to a manual URL input.
  const discoveryRequest = React.useMemo(
    () => buildDiscoveryRequest(agentType, watchedFormValues || {}, selectedAgentTypeInfo),
    [watchedFormValues, selectedAgentTypeInfo, agentType],
  );

  const handleNext = async () => {
    try {
      if (currentStep === 0) {
        await form.validateFields();
        const agentName = form.getFieldValue("agent_name");
        if (agentName && !newKeyName) {
          setNewKeyName(`${agentName}-key`);
        }
      }
      setCurrentStep((s) => s + 1);
    } catch {
      // validation failed — stay on current step
    }
  };

  const handleBack = () => {
    setCurrentStep((s) => Math.max(0, s - 1));
  };

  const buildAgentData = (values: any) => {
    if (agentType === CUSTOM_AGENT_TYPE) {
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

    let agentData: Record<string, any>;
    if (agentType === "a2a") {
      agentData = buildAgentDataFromForm(values);
    } else if (selectedAgentTypeInfo?.use_a2a_form_fields) {
      agentData = buildAgentDataFromForm(values);
      if (selectedAgentTypeInfo.litellm_params_template) {
        agentData.litellm_params = {
          ...agentData.litellm_params,
          ...selectedAgentTypeInfo.litellm_params_template,
        };
      }
      for (const field of selectedAgentTypeInfo.credential_fields) {
        const value = values[field.key];
        if (value && field.include_in_litellm_params !== false) {
          agentData.litellm_params[field.key] = value;
        }
      }
    } else if (selectedAgentTypeInfo) {
      agentData = buildDynamicAgentData(values, selectedAgentTypeInfo);
    } else {
      return null;
    }

    return overlayDiscoveredCardParams(agentData, appliedDiscoveredSelection?.selected_card);
  };

  const handleCreateAgent = async () => {
    if (!accessToken) {
      MessageManager.error(t("agents.create.errors.noToken"));
      return;
    }

    setIsSubmitting(true);
    try {
      await form.validateFields();
      const values = { ...form.getFieldsValue(true) };
      const agentData = buildAgentData(values);
      if (!agentData) {
        MessageManager.error(t("agents.create.errors.buildFailed"));
        setIsSubmitting(false);
        return;
      }

      // Build object_permission from MCP Tools step (allowed_mcp_servers_and_groups, mcp_tool_permissions)
      const mcpServersAndGroups = values.allowed_mcp_servers_and_groups;
      const mcpToolPermissions = values.mcp_tool_permissions || {};
      const entitlementModels = values.entitlement_models || [];
      const entitlementAgents = values.entitlement_agents || [];
      const hasObjectPermission =
        mcpServersAndGroups?.servers?.length > 0 ||
        mcpServersAndGroups?.accessGroups?.length > 0 ||
        Object.keys(mcpToolPermissions).length > 0 ||
        entitlementModels.length > 0 ||
        entitlementAgents.length > 0;
      if (hasObjectPermission) {
        agentData.object_permission = {};
        if (mcpServersAndGroups?.servers?.length > 0) {
          agentData.object_permission.mcp_servers = mcpServersAndGroups.servers;
        }
        if (mcpServersAndGroups?.accessGroups?.length > 0) {
          agentData.object_permission.mcp_access_groups = mcpServersAndGroups.accessGroups;
        }
        if (Object.keys(mcpToolPermissions).length > 0) {
          agentData.object_permission.mcp_tool_permissions = mcpToolPermissions;
        }
        if (entitlementModels.length > 0) {
          agentData.object_permission.models = entitlementModels;
        }
        if (entitlementAgents.length > 0) {
          agentData.object_permission.agents = entitlementAgents;
        }
      }

      // Wire trace-id flags and budget controls into agent litellm_params (before create call)
      if (requireTraceIdInbound || requireTraceIdOutbound) {
        if (!agentData.litellm_params) agentData.litellm_params = {};
        if (requireTraceIdInbound) {
          agentData.litellm_params.require_trace_id_on_calls_to_agent = true;
        }
        if (requireTraceIdOutbound) {
          agentData.litellm_params.require_trace_id_on_calls_by_agent = true;
          if (maxIterations) agentData.litellm_params.max_iterations = maxIterations;
          if (maxBudgetPerSession) agentData.litellm_params.max_budget_per_session = maxBudgetPerSession;
        }
      }

      const selectedGuardrails = values.guardrails || [];
      if (selectedGuardrails.length > 0) {
        if (!agentData.litellm_params) agentData.litellm_params = {};
        agentData.litellm_params.guardrails = selectedGuardrails;
      }

      const selectedTeamId = values.team_id || null;
      if (selectedTeamId) {
        agentData.team_id = selectedTeamId;
      }

      const agentResponse = await createAgentCall(accessToken, agentData);
      const agentId: string = agentResponse.agent_id;
      const agentName: string = agentResponse.agent_name || values.agent_name || agentId;
      setCreatedAgentName(agentName);

      if (keyAssignOption === "create_new" && newKeyName) {
        const keyResponse = await keyCreateForAgentCall(
          accessToken,
          agentId,
          newKeyName,
          newKeyModels,
          undefined,
          selectedTeamId,
        );
        setCreatedKeyValue(keyResponse.key || null);
      } else if (keyAssignOption === "existing_key") {
        if (!selectedExistingKey) {
          MessageManager.error(t("agents.create.errors.selectKey"));
          setIsSubmitting(false);
          return;
        }
        await keyUpdateCall(accessToken, {
          key: selectedExistingKey,
          agent_id: agentId,
        });
        const keyInfo = existingKeys.find((k) => k.token === selectedExistingKey);
        setAssignedKeyAlias(keyInfo?.key_alias || selectedExistingKey.slice(0, 12) + "…");
      }

      setCurrentStep(4);
      onSuccess();
    } catch (error) {
      console.error("Error creating agent:", error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      MessageManager.error(
        errorMessage
          ? t("agents.create.errors.createFailedWithReason", { reason: errorMessage })
          : t("agents.create.errors.createFailed"),
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    form.resetFields();
    setAgentType("a2a");
    setCurrentStep(0);
    setKeyAssignOption("create_new");
    setNewKeyName("");
    setNewKeyModels([]);
    setSelectedExistingKey(null);
    setCreatedAgentName("");
    setCreatedKeyValue(null);
    setAssignedKeyAlias(null);
    setRequireTraceIdInbound(false);
    setRequireTraceIdOutbound(false);
    setMaxIterations(null);
    setMaxBudgetPerSession(null);
    setAppliedDiscoveredSelection(null);
    onClose();
  };

  const renderEntitlementsStep = () => (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">{t("agents.create.entitlements.description")}</p>

      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">{t("agents.create.entitlements.models")}</span>}
        name="entitlement_models"
        tooltip={t("agents.create.entitlements.modelsHint")}
      >
        <Select
          mode="tags"
          style={{ width: "100%" }}
          placeholder={
            loadingModels ? t("agents.create.entitlements.loadingModels") : t("agents.create.entitlements.selectModels")
          }
          tokenSeparators={[","]}
          loading={loadingModels}
          showSearch
          options={availableModels.map((m) => ({
            label: getModelDisplayName(m),
            value: m,
          }))}
        />
      </Form.Item>

      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">{t("agents.create.entitlements.agents")}</span>}
        name="entitlement_agents"
        tooltip={t("agents.create.entitlements.agentsHint")}
      >
        <Select
          mode="multiple"
          style={{ width: "100%" }}
          placeholder={
            loadingAgents ? t("agents.create.entitlements.loadingAgents") : t("agents.create.entitlements.selectAgents")
          }
          loading={loadingAgents}
          showSearch
          filterOption={(input, option) =>
            ((option?.label as string) ?? "").toLowerCase().includes(input.toLowerCase())
          }
          options={availableAgents.map((a) => ({
            label: a.agent_name,
            value: a.agent_id,
          }))}
        />
      </Form.Item>

      <Divider className="my-2" />

      <Form.Item
        label={
          <span>
            {t("agents.create.entitlements.mcpServers")}{" "}
            <InfoCircleOutlined title={t("agents.create.entitlements.mcpServersHint")} style={{ marginLeft: "4px" }} />
          </span>
        }
        name="allowed_mcp_servers_and_groups"
        initialValue={{ servers: [], accessGroups: [] }}
      >
        <MCPServerSelector
          onChange={(val: { servers?: string[]; accessGroups?: string[] }) =>
            form.setFieldValue("allowed_mcp_servers_and_groups", val)
          }
          value={form.getFieldValue("allowed_mcp_servers_and_groups") || { servers: [], accessGroups: [] }}
          accessToken={accessToken ?? ""}
          placeholder={t("agents.create.entitlements.selectMcp")}
        />
      </Form.Item>
      <Form.Item name="mcp_tool_permissions" initialValue={{}} hidden>
        <Input type="hidden" />
      </Form.Item>
      <Form.Item
        noStyle
        shouldUpdate={(prev, curr) =>
          prev.allowed_mcp_servers_and_groups !== curr.allowed_mcp_servers_and_groups ||
          prev.mcp_tool_permissions !== curr.mcp_tool_permissions
        }
      >
        {() => (
          <div className="mt-4">
            <MCPToolPermissions
              accessToken={accessToken ?? ""}
              selectedServers={form.getFieldValue("allowed_mcp_servers_and_groups")?.servers ?? []}
              toolPermissions={form.getFieldValue("mcp_tool_permissions") ?? {}}
              onChange={(toolPerms: Record<string, string[]>) =>
                form.setFieldsValue({ mcp_tool_permissions: toolPerms })
              }
            />
          </div>
        )}
      </Form.Item>
    </div>
  );

  const renderObservabilityStep = () => (
    <div className="space-y-6">
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">{t("agents.create.governance.tracing")}</h4>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-gray-700">{t("agents.create.governance.inbound")}</span>
              <p className="text-xs text-gray-500 mt-1">{t("agents.create.governance.inboundHint")}</p>
            </div>
            <Switch checked={requireTraceIdInbound} onChange={setRequireTraceIdInbound} />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-gray-700">{t("agents.create.governance.outbound")}</span>
              <p className="text-xs text-gray-500 mt-1">{t("agents.create.governance.outboundHint")}</p>
            </div>
            <Switch
              checked={requireTraceIdOutbound}
              onChange={(checked) => {
                setRequireTraceIdOutbound(checked);
                if (!checked) {
                  setMaxIterations(null);
                  setMaxBudgetPerSession(null);
                }
              }}
            />
          </div>
        </div>
      </div>

      <Divider className="my-0" />

      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">{t("agents.create.governance.budgets")}</h4>
        <div className="space-y-4">
          {!requireTraceIdOutbound && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-800">
              {t("agents.create.governance.enableTracing")}
            </div>
          )}

          <div className="text-sm font-medium text-gray-700">{t("agents.create.governance.sessionBudgets")}</div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-gray-600 block mb-1">{t("agents.create.governance.maxIterations")}</label>
              <InputNumber
                className="w-full"
                min={1}
                placeholder={t("agents.create.placeholders.iterations")}
                disabled={!requireTraceIdOutbound}
                value={maxIterations}
                onChange={(val) => setMaxIterations(val)}
              />
              <p className="text-xs text-gray-400 mt-1">{t("agents.create.governance.maxIterationsHint")}</p>
            </div>
            <div>
              <label className="text-sm text-gray-600 block mb-1">{t("agents.create.governance.maxBudget")}</label>
              <InputNumber
                className="w-full"
                min={0.01}
                step={0.5}
                placeholder={t("agents.create.placeholders.budget")}
                disabled={!requireTraceIdOutbound}
                value={maxBudgetPerSession}
                onChange={(val) => setMaxBudgetPerSession(val)}
              />
              <p className="text-xs text-gray-400 mt-1">{t("agents.create.governance.maxBudgetHint")}</p>
            </div>
          </div>

          <Divider className="my-2" />

          <div className="text-sm font-medium text-gray-700">{t("agents.create.governance.agentLimits")}</div>
          <p className="text-xs text-gray-500">{t("agents.create.governance.agentLimitsHint")}</p>
          <div className="grid grid-cols-2 gap-4">
            <Form.Item label={t("agents.create.governance.tpm")} name="tpm_limit" className="mb-0">
              <InputNumber
                className="w-full"
                min={0}
                placeholder={t("agents.create.placeholders.tpm")}
                disabled={!requireTraceIdOutbound}
              />
            </Form.Item>
            <Form.Item label={t("agents.create.governance.rpm")} name="rpm_limit" className="mb-0">
              <InputNumber
                className="w-full"
                min={0}
                placeholder={t("agents.create.placeholders.rpm")}
                disabled={!requireTraceIdOutbound}
              />
            </Form.Item>
          </div>

          <div className="text-sm font-medium text-gray-700 mt-4">{t("agents.create.governance.sessionLimits")}</div>
          <p className="text-xs text-gray-500">{t("agents.create.governance.sessionLimitsHint")}</p>
          <div className="grid grid-cols-2 gap-4">
            <Form.Item label={t("agents.create.governance.sessionTpm")} name="session_tpm_limit" className="mb-0">
              <InputNumber
                className="w-full"
                min={0}
                placeholder={t("agents.create.placeholders.sessionTpm")}
                disabled={!requireTraceIdOutbound}
              />
            </Form.Item>
            <Form.Item label={t("agents.create.governance.sessionRpm")} name="session_rpm_limit" className="mb-0">
              <InputNumber
                className="w-full"
                min={0}
                placeholder={t("agents.create.placeholders.sessionRpm")}
                disabled={!requireTraceIdOutbound}
              />
            </Form.Item>
          </div>
        </div>
      </div>

      <Divider className="my-0" />

      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">{t("agents.create.governance.guardrails")}</h4>
        <p className="text-xs text-gray-500 mb-3">{t("agents.create.governance.guardrailsHint")}</p>
        <Form.Item name="guardrails" initialValue={[]}>
          <GuardrailSelector
            accessToken={accessToken ?? ""}
            value={form.getFieldValue("guardrails") ?? []}
            onChange={(selected: string[]) => form.setFieldsValue({ guardrails: selected })}
          />
        </Form.Item>
      </div>
    </div>
  );

  const handleAgentTypeChange = (value: string) => {
    setAgentType(value);
    form.resetFields();
    // Discovery selections are tied to a specific agent type's URL shape;
    // switching types invalidates them.
    setAppliedDiscoveredSelection(null);
  };

  // Apply a discovered agent card to the form so the rest of Step 1 (skills,
  // capabilities, name, description, URL) reflects what the user picked. The
  // proxy re-applies its own merge at registration; we only seed defaults here.
  //
  // AntD's `setFieldsValue` silently ignores keys whose Form.Item isn't
  // registered, so this is safe across all agent types — A2A forms pick up
  // every field below; LangGraph and other dynamic forms only pick up the
  // shared ones (`agent_name`, `description`, plus any credential field whose
  // key looks URL-ish).
  const handleApplyDiscoveredCard = (selection: DiscoveredAgentCardSelection | null) => {
    setAppliedDiscoveredSelection(selection);
    if (!selection) return;
    const { selected_card, upstream_url } = selection;
    const skills = (selected_card.skills ?? []).map((s) => ({
      id: s.id ?? "",
      name: s.name ?? "",
      description: s.description ?? "",
      tags: s.tags ?? [],
      examples: s.examples ?? [],
    }));

    const currentAgentName = form.getFieldValue("agent_name");
    const seededAgentName = currentAgentName || selected_card.name || selected_card.provider?.organization || "";

    const fieldsToSet: Record<string, any> = {
      agent_name: seededAgentName,
      name: selected_card.name,
      description: selected_card.description,
      url: upstream_url,
      version: selected_card.version,
      protocolVersion: selected_card.protocolVersion ?? "1.0",
      streaming: Boolean(selected_card.capabilities?.streaming),
      skills,
      iconUrl: selected_card.iconUrl,
      documentationUrl: selected_card.documentationUrl,
    };

    // For dynamic agent types (e.g. LangGraph), the URL lives in a
    // type-specific credential field. Match on common naming variants so the
    // user doesn't have to re-paste the URL they already typed above.
    const urlCredentialKeys = (selectedAgentTypeInfo?.credential_fields ?? [])
      .map((f) => f.key)
      .filter((key) => /(^|_)(url|api_base|endpoint)$/i.test(key));
    for (const key of urlCredentialKeys) {
      fieldsToSet[key] = upstream_url;
    }

    form.setFieldsValue(fieldsToSet);

    if (!newKeyName && seededAgentName) {
      setNewKeyName(`${seededAgentName}-key`);
    }
  };

  const isCustomAgent = agentType === CUSTOM_AGENT_TYPE;
  const selectedLogo = isCustomAgent
    ? null
    : selectedAgentTypeInfo?.logo_url || agentTypeMetadata.find((a) => a.agent_type === "a2a")?.logo_url;

  const renderConfigureStep = () => (
    <>
      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">{t("agents.create.configure.type")}</span>}
        required
        tooltip={t("agents.create.configure.typeHint")}
      >
        <Select
          value={agentType}
          onChange={handleAgentTypeChange}
          size="large"
          style={{ width: "100%" }}
          optionLabelProp="label"
          dropdownRender={(menu) => (
            <>
              {menu}
              <Divider style={{ margin: "4px 0" }} />
              <div className="px-2 py-1">
                <div className="text-xs text-gray-400 font-medium mb-1 uppercase tracking-wide px-2">
                  {t("agents.create.configure.notListed")}
                </div>
                <div
                  className={`flex items-center gap-3 px-2 py-2 rounded cursor-pointer transition-colors ${
                    agentType === CUSTOM_AGENT_TYPE ? "bg-amber-50" : "hover:bg-amber-50"
                  }`}
                  onClick={() => handleAgentTypeChange(CUSTOM_AGENT_TYPE)}
                >
                  <AppstoreOutlined className="text-amber-600 text-lg" />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-amber-700">{t("agents.create.configure.custom")}</span>
                      <Tag color="orange" style={{ fontSize: 10, padding: "0 4px" }}>
                        {t("agents.create.configure.generic")}
                      </Tag>
                    </div>
                    <div className="text-xs text-amber-600">{t("agents.create.configure.customHint")}</div>
                  </div>
                </div>
              </div>
            </>
          )}
        >
          {agentTypeMetadata.map((info) => (
            <Select.Option
              key={info.agent_type}
              value={info.agent_type}
              label={
                <div className="flex items-center gap-2">
                  <Logo src={info.logo_url} label={info.agent_type_display_name} className="w-4 h-4 object-contain" />
                  <span>{info.agent_type_display_name}</span>
                </div>
              }
            >
              <div className="flex items-center gap-3 py-1">
                <Logo src={info.logo_url} label={info.agent_type_display_name} className="w-5 h-5 object-contain" />
                <div>
                  <div className="font-medium">{info.agent_type_display_name}</div>
                  {info.description && <div className="text-xs text-gray-500">{info.description}</div>}
                </div>
              </div>
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      <div className="mt-4">
        {agentType === CUSTOM_AGENT_TYPE ? (
          <div className="space-y-4">
            <Form.Item
              label={t("agents.form.agentName")}
              name="agent_name"
              rules={[{ required: true, message: t("agents.create.configure.agentNameRequired") }]}
            >
              <Input placeholder={t("agents.create.placeholders.customAgent")} />
            </Form.Item>
            <Form.Item label={t("agents.create.configure.description")} name="description">
              <Input.TextArea placeholder={t("agents.create.configure.descriptionPlaceholder")} rows={3} />
            </Form.Item>
          </div>
        ) : agentType === "a2a" ? (
          <AgentFormFields showAgentName={true} />
        ) : selectedAgentTypeInfo?.use_a2a_form_fields ? (
          <>
            <AgentFormFields showAgentName={true} />
            {selectedAgentTypeInfo.credential_fields.length > 0 && (
              <div className="mt-4 p-4 border border-gray-200 rounded-lg">
                <h4 className="text-sm font-medium text-gray-700 mb-3">
                  {t("agents.create.configure.settings", { name: selectedAgentTypeInfo.agent_type_display_name })}
                </h4>
                {selectedAgentTypeInfo.credential_fields.map((field) => (
                  <Form.Item
                    key={field.key}
                    label={field.label}
                    name={field.key}
                    rules={
                      field.required
                        ? [{ required: true, message: t("agents.form.requiredField", { field: field.label }) }]
                        : undefined
                    }
                    tooltip={field.tooltip}
                    initialValue={field.default_value}
                  >
                    {field.field_type === "password" ? (
                      <Input.Password placeholder={field.placeholder || ""} />
                    ) : (
                      <Input placeholder={field.placeholder || ""} />
                    )}
                  </Form.Item>
                ))}
              </div>
            )}
          </>
        ) : selectedAgentTypeInfo ? (
          <DynamicAgentFormFields agentTypeInfo={selectedAgentTypeInfo} />
        ) : null}

        {/* Discovery sits at the bottom so its URL can be derived from the
            credential fields the user typed above. The plan (URL + mode +
            params) is computed from the agent type — LangGraph hits a
            different shape than pure A2A. Custom agents have no upstream to
            discover, so we skip them. */}
        {agentType !== CUSTOM_AGENT_TYPE && (
          <div className="mt-4">
            <AgentCardDiscovery
              accessToken={accessToken}
              onApply={handleApplyDiscoveredCard}
              discoveryRequest={discoveryRequest}
            />
          </div>
        )}
      </div>
    </>
  );

  const renderAssignKeyStep = () => {
    const agentName = form.getFieldValue("agent_name") || t("agents.create.management.fallbackName");
    return (
      <div>
        {/* Agent name chip */}
        <div className="flex justify-center mb-6">
          <Tag icon={<RobotOutlined />} color="purple" className="px-3 py-1 text-sm">
            {agentName}
          </Tag>
        </div>

        <Form.Item
          label={<span className="text-sm font-medium text-gray-700">{t("agents.create.management.team")}</span>}
          name="team_id"
          tooltip={t("agents.create.management.teamHint")}
        >
          <TeamDropdown />
        </Form.Item>

        <Divider className="my-4" />

        <div className="space-y-3">
          {/* Option: Create new key */}
          <div
            className={`p-4 border-2 rounded-lg cursor-pointer transition-colors ${
              keyAssignOption === "create_new"
                ? "border-indigo-600 bg-indigo-50"
                : "border-gray-200 bg-white hover:border-gray-300"
            }`}
            onClick={() => setKeyAssignOption("create_new")}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3 flex-1">
                <Radio
                  value="create_new"
                  checked={keyAssignOption === "create_new"}
                  onChange={() => setKeyAssignOption("create_new")}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <KeyOutlined className="text-indigo-600" />
                    <span className="font-medium text-gray-900">{t("agents.create.management.createKey")}</span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">{t("agents.create.management.createKeyHint")}</p>
                  {keyAssignOption === "create_new" && (
                    <div className="mt-3 space-y-3" onClick={(e) => e.stopPropagation()}>
                      <div>
                        <label className="text-sm text-gray-600 block mb-1">
                          {t("agents.create.management.keyName")}
                        </label>
                        <Input
                          value={newKeyName}
                          onChange={(e) => setNewKeyName(e.target.value)}
                          placeholder={t("agents.create.placeholders.keyName")}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
              <Tag color="green">{t("agents.create.management.recommended")}</Tag>
            </div>
          </div>

          {/* Option: Assign existing key */}
          <div
            className={`p-4 border-2 rounded-lg cursor-pointer transition-colors ${
              keyAssignOption === "existing_key"
                ? "border-indigo-600 bg-indigo-50"
                : "border-gray-200 bg-white hover:border-gray-300"
            }`}
            onClick={() => setKeyAssignOption("existing_key")}
          >
            <div className="flex items-start gap-3">
              <Radio
                value="existing_key"
                checked={keyAssignOption === "existing_key"}
                onChange={() => setKeyAssignOption("existing_key")}
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <KeyOutlined className="text-gray-500" />
                  <span className="font-medium text-gray-900">{t("agents.create.management.existingKey")}</span>
                </div>
                <p className="text-sm text-gray-500 mt-1">{t("agents.create.management.existingKeyHint")}</p>
                {keyAssignOption === "existing_key" && (
                  <div className="mt-3" onClick={(e) => e.stopPropagation()}>
                    <Select
                      showSearch
                      style={{ width: "100%" }}
                      placeholder={t("agents.create.management.searchKey")}
                      loading={loadingKeys}
                      value={selectedExistingKey}
                      onChange={(value) => setSelectedExistingKey(value)}
                      filterOption={(input, option) =>
                        ((option?.label as string) ?? "").toLowerCase().includes(input.toLowerCase())
                      }
                      options={existingKeys.map((k) => ({
                        label: k.key_alias || k.token?.slice(0, 12) + "…",
                        value: k.token,
                      }))}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="text-center mt-4">
          <button
            type="button"
            className="text-sm text-gray-500 underline hover:text-gray-700"
            onClick={() => setKeyAssignOption("skip")}
          >
            {t("agents.create.management.skip")}
          </button>
        </div>
      </div>
    );
  };

  const renderReadyStep = () => (
    <div className="text-center py-6">
      <CheckCircleFilled className="text-5xl text-green-500 mb-4" style={{ fontSize: 48 }} />
      <h3 className="text-xl font-semibold text-gray-900 mb-2">{t("agents.create.ready.title")}</h3>
      <div className="flex justify-center mb-4">
        <Tag icon={<RobotOutlined />} color="purple" className="px-3 py-1 text-sm">
          {createdAgentName}
        </Tag>
      </div>
      {createdKeyValue && (
        <div className="mt-4 text-left max-w-md mx-auto">
          <CreatedKeyDisplay apiKey={createdKeyValue} />
        </div>
      )}
      {assignedKeyAlias && (
        <p className="text-sm text-gray-600 mt-2">{t("agents.create.ready.keyAssigned", { key: assignedKeyAlias })}</p>
      )}
      {!createdKeyValue && !assignedKeyAlias && keyAssignOption === "skip" && (
        <p className="text-sm text-gray-500 mt-2">{t("agents.create.ready.noKey")}</p>
      )}
    </div>
  );

  return (
    <Modal
      title={
        <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
          {selectedLogo && currentStep < 1 && (
            <Logo src={selectedLogo} label={t("agents.title")} className="w-6 h-6 object-contain" />
          )}
          <h2 className="text-xl font-semibold text-gray-900">{t("agents.create.title")}</h2>
        </div>
      }
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={900}
      className="top-8"
      styles={{
        body: { padding: "24px" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
    >
      <div className="mt-4">
        {/* Step indicator */}
        <Steps current={currentStep} size="small" className="mb-8">
          <Step title={t("agents.create.steps.configure")} />
          <Step title={t("agents.create.steps.entitlements")} />
          <Step title={t("agents.create.steps.governance")} />
          <Step title={t("agents.create.steps.management")} />
          <Step title={t("agents.create.steps.ready")} />
        </Steps>

        <Form
          form={form}
          layout="vertical"
          initialValues={
            agentType === "a2a"
              ? {
                  ...getDefaultFormValues(),
                  allowed_mcp_servers_and_groups: { servers: [], accessGroups: [] },
                  mcp_tool_permissions: {},
                  entitlement_models: [],
                  entitlement_agents: [],
                  guardrails: [],
                }
              : {
                  allowed_mcp_servers_and_groups: { servers: [], accessGroups: [] },
                  mcp_tool_permissions: {},
                  entitlement_models: [],
                  entitlement_agents: [],
                  guardrails: [],
                }
          }
          className="space-y-4"
        >
          {currentStep === 0 && renderConfigureStep()}
          {currentStep === 1 && renderEntitlementsStep()}
          {currentStep === 2 && renderObservabilityStep()}
          {currentStep === 3 && renderAssignKeyStep()}
          {currentStep === 4 && renderReadyStep()}
        </Form>

        {/* Footer navigation */}
        <div className="flex items-center justify-between pt-6 border-t border-gray-100 mt-6">
          <div>
            {currentStep > 0 && currentStep < 4 && (
              <button
                type="button"
                onClick={handleBack}
                className="text-sm text-gray-600 border border-gray-300 rounded-sm px-4 py-2 hover:bg-gray-50"
              >
                {t("agents.create.navigation.back")}
              </button>
            )}
          </div>
          <div className="flex gap-3">
            {currentStep < 4 && (
              <Button variant="secondary" onClick={handleClose}>
                {t("agents.create.navigation.cancel")}
              </Button>
            )}
            {currentStep === 0 && (
              <Button variant="primary" onClick={handleNext}>
                {t("agents.create.navigation.next")}
              </Button>
            )}
            {currentStep === 1 && (
              <Button variant="primary" onClick={handleNext}>
                {t("agents.create.navigation.next")}
              </Button>
            )}
            {currentStep === 2 && (
              <Button variant="primary" onClick={handleNext}>
                {t("agents.create.navigation.next")}
              </Button>
            )}
            {currentStep === 3 && (
              <Button variant="primary" loading={isSubmitting} onClick={handleCreateAgent}>
                {isSubmitting ? t("agents.create.navigation.creating") : t("agents.create.navigation.create")}
              </Button>
            )}
            {currentStep === 4 && (
              <Button variant="primary" onClick={handleClose}>
                {t("agents.create.navigation.done")}
              </Button>
            )}
          </div>
        </div>
      </div>
    </Modal>
  );
};

export default AddAgentForm;
