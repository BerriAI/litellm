import { AgentIdentityFields } from "./AgentIdentityFields";
import { withAgentIdentity } from "./agent_identity";
import React, { useState, useEffect } from "react";
import { FormProvider, useForm, useWatch } from "react-hook-form";
import { toast } from "@/lib/toast";
import { Logo } from "@/components/molecules/logo/Logo";
import { Bot, Check, CircleCheck, Key, LayoutGrid } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import CreatedKeyDisplay from "@/components/shared/CreatedKeyDisplay";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectSeparator, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { SearchSelect } from "@/components/shared/SearchSelect";
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
import { PasswordInput } from "@/components/shared/PasswordInput";
import { AGENT_FORM_CONFIG, getDefaultFormValues, buildAgentDataFromForm } from "./agent_config";
import {
  AgentFormField,
  AgentFormValues,
  AgentMultiSelect,
  AgentNumberInput,
  AgentRequestPayload,
  AgentTagsInput,
  McpServerSelection,
  labelWithHint,
  useCollapsiblePanels,
} from "./AgentFormKit";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "@/components/mcp_server_management/MCPToolPermissions";
import AccessGroupSelector from "@/components/common_components/AccessGroupSelector";
import GuardrailSelector from "@/components/guardrails/GuardrailSelector";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import i18n from "@/i18n/i18n";

const CUSTOM_AGENT_TYPE = "custom";

const STEP_TITLES = [
  "agents.addForm.steps.configure",
  "agents.addForm.steps.entitlements",
  "agents.addForm.steps.governance",
  "agents.addForm.steps.agentManagement",
  "agents.addForm.steps.ready",
] as const;

const stepMarkerClass = (index: number, current: number): string => {
  if (index < current) return "border-primary text-primary";
  if (index === current) return "border-primary bg-primary text-primary-foreground";
  return "border-border text-muted-foreground";
};

const stepTitleClass = (index: number, current: number): string => {
  if (index === current) return "font-medium text-foreground";
  if (index < current) return "text-foreground";
  return "text-muted-foreground";
};

const AgentTypeLabel: React.FC<{ agentType: string; info: AgentCreateInfo | undefined }> = ({ agentType, info }) => {
  const { t } = useTranslation();
  if (agentType === CUSTOM_AGENT_TYPE) {
    return (
      <span className="flex items-center gap-2">
        <LayoutGrid className="size-4 text-warning" />
        <span>{t("agents.addForm.agentType.custom")}</span>
      </span>
    );
  }
  if (!info) return <>{agentType}</>;
  return (
    <span className="flex items-center gap-2">
      <Logo src={info.logo_url} label={info.agent_type_display_name} className="h-4 w-4 object-contain" />
      <span>{info.agent_type_display_name}</span>
    </span>
  );
};

const StepProgress: React.FC<{ current: number }> = ({ current }) => (
  <ol aria-label={i18n.t("agents.addForm.steps.ariaLabel")} className="mb-8 flex items-center">
    {STEP_TITLES.map((title, index) => (
      <li
        key={title}
        aria-current={index === current ? "step" : undefined}
        className="flex flex-1 items-center gap-2 last:flex-none"
      >
        <span
          aria-hidden="true"
          className={`flex size-6 shrink-0 items-center justify-center rounded-full border text-xs ${stepMarkerClass(index, current)}`}
        >
          {index < current ? <Check className="size-3.5" /> : index + 1}
        </span>
        <span className={`text-xs whitespace-nowrap ${stepTitleClass(index, current)}`}>{i18n.t(title)}</span>
        {index < STEP_TITLES.length - 1 && <span aria-hidden="true" className="mx-2 h-px flex-1 bg-border" />}
      </li>
    ))}
  </ol>
);

const SHARED_INITIAL_VALUES: AgentFormValues = {
  allowed_mcp_servers_and_groups: { servers: [], accessGroups: [] },
  mcp_tool_permissions: {},
  entitlement_models: [],
  entitlement_agents: [],
  access_group_ids: [],
  guardrails: [],
};

const buildInitialValues = (agentType: string): AgentFormValues =>
  agentType === "a2a" ? { ...getDefaultFormValues(), ...SHARED_INITIAL_VALUES } : { ...SHARED_INITIAL_VALUES };

interface AddAgentFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
  teams?: Team[] | null;
}

const AddAgentForm: React.FC<AddAgentFormProps> = ({ visible, onClose, accessToken, onSuccess, teams }) => {
  const { t } = useTranslation();
  const { userId, userRole } = useAuthorized();
  const form = useForm<AgentFormValues>({ defaultValues: buildInitialValues("a2a") });
  const panels = useCollapsiblePanels([AGENT_FORM_CONFIG.basic.key]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [agentType, setAgentType] = useState<string>("a2a");
  const [agentTypeMetadata, setAgentTypeMetadata] = useState<AgentCreateInfo[]>([]);

  // Step 3: key assignment state
  const [keyAssignOption, setKeyAssignOption] = useState<"create_new" | "existing_key" | "skip">("create_new");
  const [newKeyName, setNewKeyName] = useState<string>("");
  const [newKeyModels, setNewKeyModels] = useState<string[]>([]);
  const [existingKeys, setExistingKeys] = useState<{ token: string; key_alias?: string }[]>([]);
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

  const [appliedDiscoveredSelection, setAppliedDiscoveredSelection] = useState<DiscoveredAgentCardSelection | null>(
    null,
  );

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
        setAvailableAgents(
          agents.map((a: { agent_id: string; agent_name: string }) => ({
            agent_id: a.agent_id,
            agent_name: a.agent_name,
          })),
        );
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
  const watchedFormValues = useWatch({ control: form.control });
  const mcpSelection = useWatch({ control: form.control, name: "allowed_mcp_servers_and_groups" });
  const mcpToolPermissions = useWatch({ control: form.control, name: "mcp_tool_permissions" });

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
    if (currentStep === 0) {
      const isValid = await form.trigger();
      if (!isValid) return;
      const agentName = form.getValues("agent_name");
      if (agentName && !newKeyName) {
        setNewKeyName(`${agentName}-key`);
      }
    }
    setCurrentStep((s) => s + 1);
  };

  const handleBack = () => {
    setCurrentStep((s) => Math.max(0, s - 1));
  };

  const buildAgentData = (values: AgentFormValues): AgentRequestPayload | null => {
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

  const handleCreateAgent = async () => {
    if (!accessToken) {
      toast.error(t("agents.addForm.toasts.noAccessToken"));
      return;
    }

    if (keyAssignOption === "existing_key" && !selectedExistingKey) {
      toast.error(t("agents.addForm.toasts.selectExistingKey"));
      return;
    }

    setIsSubmitting(true);
    try {
      const isValid = await form.trigger();
      if (!isValid) {
        setIsSubmitting(false);
        return;
      }
      const values = form.getValues();
      const built = buildAgentData(values);
      if (!built) {
        toast.error(t("agents.addForm.toasts.buildFailed"));
        setIsSubmitting(false);
        return;
      }
      const agentData = withAgentIdentity(built, values);

      // Build object_permission from MCP Tools step (allowed_mcp_servers_and_groups, mcp_tool_permissions)
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

      // Wire trace-id flags and budget controls into agent litellm_params (before create call)
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
      } else if (keyAssignOption === "existing_key" && selectedExistingKey) {
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
      toast.error(
        errorMessage
          ? t("agents.addForm.toasts.createFailedWithError", { error: errorMessage })
          : t("agents.addForm.toasts.createFailed"),
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    form.reset(buildInitialValues(agentType));
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
      <p className="text-sm text-muted-foreground">{t("agents.addForm.entitlements.description")}</p>

      <FieldGroup>
        <AgentFormField
          name="entitlement_models"
          label={labelWithHint(
            t("agents.addForm.entitlements.models.label"),
            t("agents.addForm.entitlements.models.hint"),
          )}
        >
          {({ id, value, onChange }) => (
            <AgentTagsInput
              id={id}
              value={Array.isArray(value) ? (value as string[]) : []}
              onValueChange={onChange}
              placeholder={
                loadingModels
                  ? t("agents.addForm.entitlements.models.loading")
                  : t("agents.addForm.entitlements.models.placeholder")
              }
              options={availableModels.map((m) => ({ label: getModelDisplayName(m), value: m }))}
            />
          )}
        </AgentFormField>

        <AgentFormField
          name="entitlement_agents"
          label={labelWithHint(
            t("agents.addForm.entitlements.agents.label"),
            t("agents.addForm.entitlements.agents.hint"),
          )}
        >
          {({ id, value, onChange }) => (
            <AgentMultiSelect
              id={id}
              value={Array.isArray(value) ? (value as string[]) : []}
              onValueChange={onChange}
              placeholder={
                loadingAgents
                  ? t("agents.addForm.entitlements.agents.loading")
                  : t("agents.addForm.entitlements.agents.placeholder")
              }
              options={availableAgents.map((a) => ({ label: a.agent_name, value: a.agent_id }))}
            />
          )}
        </AgentFormField>

        <AgentFormField
          name="access_group_ids"
          label={labelWithHint(
            t("agents.addForm.entitlements.accessGroups.label"),
            t("agents.addForm.entitlements.accessGroups.hint"),
          )}
        >
          {({ value, onChange }) => (
            <AccessGroupSelector
              value={Array.isArray(value) ? (value as string[]) : []}
              onChange={onChange}
              placeholder={t("agents.addForm.entitlements.accessGroups.placeholder")}
            />
          )}
        </AgentFormField>

        <Separator className="my-2" />

        <AgentFormField
          name="allowed_mcp_servers_and_groups"
          label={labelWithHint(
            t("agents.addForm.entitlements.mcpServers.label"),
            t("agents.addForm.entitlements.mcpServers.hint"),
          )}
        >
          {({ value, onChange }) => (
            <MCPServerSelector
              onChange={onChange}
              value={{
                servers: (value as McpServerSelection | undefined)?.servers ?? [],
                accessGroups: (value as McpServerSelection | undefined)?.accessGroups ?? [],
              }}
              accessToken={accessToken ?? ""}
              placeholder={t("agents.addForm.entitlements.mcpServers.placeholder")}
            />
          )}
        </AgentFormField>
      </FieldGroup>

      <div className="mt-4">
        <MCPToolPermissions
          accessToken={accessToken ?? ""}
          selectedServers={mcpSelection?.servers ?? []}
          selectedAccessGroups={mcpSelection?.accessGroups ?? []}
          selectedToolsets={mcpSelection?.toolsets ?? []}
          toolPermissions={mcpToolPermissions ?? {}}
          onChange={(toolPerms: Record<string, string[]>) => form.setValue("mcp_tool_permissions", toolPerms)}
        />
      </div>
    </div>
  );

  const rateLimitField = (name: keyof AgentFormValues & string, label: string, placeholder: string) => (
    <AgentFormField name={name} label={label} className="gap-1">
      {({ value, onChange, ref, ...control }) => (
        <AgentNumberInput
          {...control}
          value={value}
          onChange={onChange}
          inputRef={ref}
          min={0}
          placeholder={placeholder}
          disabled={!requireTraceIdOutbound}
        />
      )}
    </AgentFormField>
  );

  const renderObservabilityStep = () => (
    <div className="space-y-6">
      <div>
        <h4 className="mb-3 text-sm font-medium text-foreground">{t("agents.addForm.governance.tracing.title")}</h4>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-foreground">
                {t("agents.addForm.governance.tracing.inbound.label", { header: "x-litellm-trace-id" })}
              </span>
              <p className="mt-1 text-xs text-muted-foreground">
                {t("agents.addForm.governance.tracing.inbound.description")}
              </p>
            </div>
            <Switch checked={requireTraceIdInbound} onCheckedChange={setRequireTraceIdInbound} />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-foreground">
                {t("agents.addForm.governance.tracing.outbound.label", { header: "x-litellm-trace-id" })}
              </span>
              <p className="mt-1 text-xs text-muted-foreground">
                {t("agents.addForm.governance.tracing.outbound.description", { header: "x-litellm-trace-id" })}
              </p>
            </div>
            <Switch
              checked={requireTraceIdOutbound}
              onCheckedChange={(checked) => {
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

      <Separator />

      <div>
        <h4 className="mb-3 text-sm font-medium text-foreground">{t("agents.addForm.governance.budgets.title")}</h4>
        <div className="space-y-4">
          {!requireTraceIdOutbound && (
            <div className="rounded-lg border border-warning/20 bg-warning/10 p-3 text-sm text-warning">
              {t("agents.addForm.governance.budgets.requiresOutbound", { header: "x-litellm-trace-id" })}
            </div>
          )}

          <div className="text-sm font-medium text-foreground">
            {t("agents.addForm.governance.budgets.session.title")}
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Field className="gap-1">
              <FieldLabel htmlFor="agent-max-iterations">
                {t("agents.addForm.governance.budgets.session.maxIterations.label")}
              </FieldLabel>
              <Input
                id="agent-max-iterations"
                type="number"
                step="any"
                placeholder={t("agents.addForm.governance.budgets.session.maxIterations.placeholder")}
                disabled={!requireTraceIdOutbound}
                value={maxIterations ?? ""}
                onChange={(event) =>
                  setMaxIterations(Number.isNaN(event.target.valueAsNumber) ? null : event.target.valueAsNumber)
                }
                onBlur={() => setMaxIterations((current) => (current !== null && current < 1 ? 1 : current))}
              />
              <p className="mt-1 text-xs text-muted-foreground">
                {t("agents.addForm.governance.budgets.session.maxIterations.hint")}
              </p>
            </Field>
            <Field className="gap-1">
              <FieldLabel htmlFor="agent-max-budget-per-session">
                {t("agents.addForm.governance.budgets.session.maxBudget.label")}
              </FieldLabel>
              <Input
                id="agent-max-budget-per-session"
                type="number"
                step="any"
                placeholder={t("agents.addForm.governance.budgets.session.maxBudget.placeholder")}
                disabled={!requireTraceIdOutbound}
                value={maxBudgetPerSession ?? ""}
                onChange={(event) =>
                  setMaxBudgetPerSession(Number.isNaN(event.target.valueAsNumber) ? null : event.target.valueAsNumber)
                }
                onBlur={() =>
                  setMaxBudgetPerSession((current) => (current !== null && current < 0.01 ? 0.01 : current))
                }
              />
              <p className="mt-1 text-xs text-muted-foreground">
                {t("agents.addForm.governance.budgets.session.maxBudget.hint")}
              </p>
            </Field>
          </div>

          <Separator className="my-2" />

          <div className="text-sm font-medium text-foreground">
            {t("agents.addForm.governance.rateLimits.agent.title")}
          </div>
          <p className="text-xs text-muted-foreground">{t("agents.addForm.governance.rateLimits.agent.description")}</p>
          <div className="grid grid-cols-2 gap-4">
            {rateLimitField(
              "tpm_limit",
              t("agents.addForm.governance.rateLimits.agent.tpm.label"),
              t("agents.addForm.governance.rateLimits.agent.tpm.placeholder"),
            )}
            {rateLimitField(
              "rpm_limit",
              t("agents.addForm.governance.rateLimits.agent.rpm.label"),
              t("agents.addForm.governance.rateLimits.agent.rpm.placeholder"),
            )}
          </div>

          <div className="mt-4 text-sm font-medium text-foreground">
            {t("agents.addForm.governance.rateLimits.session.title")}
          </div>
          <p className="text-xs text-muted-foreground">
            {t("agents.addForm.governance.rateLimits.session.description", { header: "x-litellm-trace-id" })}
          </p>
          <div className="grid grid-cols-2 gap-4">
            {rateLimitField(
              "session_tpm_limit",
              t("agents.addForm.governance.rateLimits.session.tpm.label"),
              t("agents.addForm.governance.rateLimits.session.tpm.placeholder"),
            )}
            {rateLimitField(
              "session_rpm_limit",
              t("agents.addForm.governance.rateLimits.session.rpm.label"),
              t("agents.addForm.governance.rateLimits.session.rpm.placeholder"),
            )}
          </div>
        </div>
      </div>

      <Separator />

      <div>
        <h4 className="mb-3 text-sm font-medium text-foreground">{t("agents.addForm.governance.guardrails.title")}</h4>
        <p className="mb-3 text-xs text-muted-foreground">{t("agents.addForm.governance.guardrails.description")}</p>
        <AgentFormField name="guardrails">
          {({ value, onChange }) => (
            <GuardrailSelector
              accessToken={accessToken ?? ""}
              value={Array.isArray(value) ? (value as string[]) : []}
              onChange={onChange}
            />
          )}
        </AgentFormField>
      </div>
    </div>
  );

  const handleAgentTypeChange = (value: string) => {
    setAgentType(value);
    form.reset(buildInitialValues(agentType));
    // Discovery selections are tied to a specific agent type's URL shape;
    // switching types invalidates them.
    setAppliedDiscoveredSelection(null);
  };

  // Apply a discovered agent card to the form so the rest of Step 1 (skills,
  // capabilities, name, description, URL) reflects what the user picked. The
  // proxy re-applies its own merge at registration; we only seed defaults here.
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

    const currentAgentName = form.getValues("agent_name");
    const seededAgentName = currentAgentName || selected_card.name || selected_card.provider?.organization || "";

    const urlCredentialKeys = (selectedAgentTypeInfo?.credential_fields ?? [])
      .map((f) => f.key)
      .filter((key) => /(^|_)(url|api_base|endpoint)$/i.test(key));

    const fieldsToSet: AgentFormValues = {
      agent_name: seededAgentName,
      name: selected_card.name ?? undefined,
      description: selected_card.description ?? undefined,
      url: upstream_url,
      version: selected_card.version ?? undefined,
      protocolVersion: selected_card.protocolVersion ?? "1.0",
      streaming: Boolean(selected_card.capabilities?.streaming),
      skills,
      iconUrl: selected_card.iconUrl ?? undefined,
      documentationUrl: selected_card.documentationUrl ?? undefined,
      ...Object.fromEntries(urlCredentialKeys.map((key) => [key, upstream_url])),
    };

    for (const [key, value] of Object.entries(fieldsToSet)) {
      form.setValue(key, value);
    }

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
      <Field className="gap-1">
        <FieldLabel htmlFor="agent-type">
          {labelWithHint(t("agents.addForm.configure.agentType.label"), t("agents.addForm.configure.agentType.hint"))}
        </FieldLabel>
        <Select value={agentType} onValueChange={(value) => value !== null && handleAgentTypeChange(value)}>
          <SelectTrigger id="agent-type" className="h-10 w-full">
            <SelectValue>{() => <AgentTypeLabel agentType={agentType} info={selectedAgentTypeInfo} />}</SelectValue>
          </SelectTrigger>
          <SelectContent className="p-1">
            {agentTypeMetadata.map((info) => (
              <SelectItem key={info.agent_type} value={info.agent_type}>
                <span className="flex items-center gap-3 py-1">
                  <Logo src={info.logo_url} label={info.agent_type_display_name} className="h-5 w-5 object-contain" />
                  <span className="block">
                    <span className="block font-medium">{info.agent_type_display_name}</span>
                    {info.description && (
                      <span className="block text-xs text-muted-foreground">{info.description}</span>
                    )}
                  </span>
                </span>
              </SelectItem>
            ))}
            <SelectSeparator />
            <div className="mb-1 px-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
              {t("agents.addForm.configure.agentType.notListed")}
            </div>
            <SelectItem value={CUSTOM_AGENT_TYPE} className="focus:bg-warning/10">
              <span className="flex items-center gap-3">
                <LayoutGrid className="size-4.5 shrink-0 text-warning" />
                <span className="block">
                  <span className="flex items-center gap-2">
                    <span className="font-medium text-warning">{t("agents.addForm.agentType.custom")}</span>
                    <StatusBadge
                      tone="warning"
                      label={t("agents.addForm.configure.agentType.genericBadge")}
                      className="h-4 px-1 text-[10px]"
                    />
                  </span>
                  <span className="block text-xs whitespace-normal text-warning">
                    {t("agents.addForm.configure.agentType.customDescription")}
                  </span>
                </span>
              </span>
            </SelectItem>
          </SelectContent>
        </Select>
      </Field>

      <AgentIdentityFields accessToken={accessToken} />

      <div className="mt-4">
        {agentType === CUSTOM_AGENT_TYPE ? (
          <FieldGroup>
            <AgentFormField
              name="agent_name"
              label={t("agents.addForm.configure.custom.agentName.label")}
              rules={{ required: t("agents.addForm.configure.custom.agentName.required") }}
            >
              {({ value, onChange, ref, ...control }) => (
                <Input
                  {...control}
                  ref={ref}
                  placeholder={t("agents.addForm.configure.custom.agentName.placeholder")}
                  value={typeof value === "string" ? value : ""}
                  onChange={onChange}
                />
              )}
            </AgentFormField>
            <AgentFormField name="description" label={t("agents.addForm.configure.custom.description.label")}>
              {({ value, onChange, ref, ...control }) => (
                <Textarea
                  {...control}
                  ref={ref}
                  rows={3}
                  placeholder={t("agents.addForm.configure.custom.description.placeholder")}
                  value={typeof value === "string" ? value : ""}
                  onChange={onChange}
                />
              )}
            </AgentFormField>
          </FieldGroup>
        ) : agentType === "a2a" ? (
          <AgentFormFields showAgentName={true} panels={panels} />
        ) : selectedAgentTypeInfo?.use_a2a_form_fields ? (
          <>
            <AgentFormFields showAgentName={true} panels={panels} />
            {selectedAgentTypeInfo.credential_fields.length > 0 && (
              <div className="mt-4 rounded-lg border border-border p-4">
                <h4 className="mb-3 text-sm font-medium text-foreground">
                  {t("agents.addForm.configure.typeSettings", { type: selectedAgentTypeInfo.agent_type_display_name })}
                </h4>
                <FieldGroup>
                  {selectedAgentTypeInfo.credential_fields.map((field) => (
                    <AgentFormField
                      key={field.key}
                      name={field.key}
                      label={field.tooltip ? labelWithHint(field.label, field.tooltip) : field.label}
                      defaultValue={field.default_value ?? undefined}
                      rules={
                        field.required
                          ? { required: t("agents.addForm.configure.enterField", { field: field.label }) }
                          : undefined
                      }
                    >
                      {({ value, onChange, ref, ...control }) =>
                        field.field_type === "password" ? (
                          <PasswordInput
                            {...control}
                            value={typeof value === "string" ? value : ""}
                            onChange={onChange}
                            ref={ref}
                            placeholder={field.placeholder || ""}
                          />
                        ) : (
                          <Input
                            {...control}
                            ref={ref}
                            placeholder={field.placeholder || ""}
                            value={typeof value === "string" ? value : ""}
                            onChange={onChange}
                          />
                        )
                      }
                    </AgentFormField>
                  ))}
                </FieldGroup>
              </div>
            )}
          </>
        ) : selectedAgentTypeInfo ? (
          <DynamicAgentFormFields agentTypeInfo={selectedAgentTypeInfo} panels={panels} />
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
    const agentName = form.getValues("agent_name") || t("agents.addForm.assignKey.agentNameFallback");
    return (
      <div>
        {/* Agent name chip */}
        <div className="mb-6 flex justify-center">
          <Badge className="h-auto gap-1.5 bg-purple-100 px-3 py-1 text-sm text-purple-700 dark:bg-purple-950 dark:text-purple-300">
            <Bot className="size-3.5" />
            {agentName}
          </Badge>
        </div>

        <AgentFormField
          name="team_id"
          label={labelWithHint(t("agents.addForm.assignKey.team.label"), t("agents.addForm.assignKey.team.hint"))}
        >
          {({ value, onChange }) => (
            <TeamDropdown value={typeof value === "string" ? value : undefined} onChange={onChange} />
          )}
        </AgentFormField>

        <Separator className="my-4" />

        {form.getValues("identity_provider") === "microsoft_entra" && (
          <p className="mb-4 text-sm text-muted-foreground">{t("agents.addForm.assignKey.entraNotice")}</p>
        )}
        <RadioGroup
          value={keyAssignOption}
          onValueChange={(value) => setKeyAssignOption(value as "create_new" | "existing_key" | "skip")}
          className="space-y-3"
        >
          {/* Option: Create new key */}
          <div
            className={`cursor-pointer rounded-lg border-2 p-4 transition-colors ${
              keyAssignOption === "create_new"
                ? "border-info bg-info/10"
                : "border-border bg-background hover:border-muted-foreground/40"
            }`}
            onClick={() => setKeyAssignOption("create_new")}
          >
            <div className="flex items-start justify-between">
              <div className="flex flex-1 items-start gap-3">
                <RadioGroupItem value="create_new" aria-label={t("agents.addForm.assignKey.createNew.title")} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Key className="size-4 text-info" />
                    <span className="font-medium text-foreground">{t("agents.addForm.assignKey.createNew.title")}</span>
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {t("agents.addForm.assignKey.createNew.description")}
                  </p>
                  {keyAssignOption === "create_new" && (
                    <div className="mt-3 space-y-3" onClick={(e) => e.stopPropagation()}>
                      <Field className="gap-1">
                        <FieldLabel htmlFor="agent-new-key-name">
                          {t("agents.addForm.assignKey.createNew.keyName.label")}
                        </FieldLabel>
                        <Input
                          id="agent-new-key-name"
                          value={newKeyName}
                          onChange={(e) => setNewKeyName(e.target.value)}
                          placeholder={t("agents.addForm.assignKey.createNew.keyName.placeholder")}
                        />
                      </Field>
                    </div>
                  )}
                </div>
              </div>
              <StatusBadge tone="success" label={t("agents.addForm.assignKey.createNew.recommended")} />
            </div>
          </div>

          {/* Option: Assign existing key */}
          <div
            className={`cursor-pointer rounded-lg border-2 p-4 transition-colors ${
              keyAssignOption === "existing_key"
                ? "border-info bg-info/10"
                : "border-border bg-background hover:border-muted-foreground/40"
            }`}
            onClick={() => setKeyAssignOption("existing_key")}
          >
            <div className="flex items-start gap-3">
              <RadioGroupItem value="existing_key" aria-label={t("agents.addForm.assignKey.existing.title")} />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Key className="size-4 text-muted-foreground" />
                  <span className="font-medium text-foreground">{t("agents.addForm.assignKey.existing.title")}</span>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">
                  {t("agents.addForm.assignKey.existing.description")}
                </p>
                {keyAssignOption === "existing_key" && (
                  <div className="mt-3" onClick={(e) => e.stopPropagation()}>
                    <SearchSelect
                      inputId="agent-existing-key"
                      placeholder={
                        loadingKeys
                          ? t("agents.addForm.assignKey.existing.loading")
                          : t("agents.addForm.assignKey.existing.placeholder")
                      }
                      value={selectedExistingKey}
                      onValueChange={setSelectedExistingKey}
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
        </RadioGroup>

        <div className="mt-4 text-center">
          <button
            type="button"
            className="text-sm text-muted-foreground underline hover:text-foreground"
            onClick={() => setKeyAssignOption("skip")}
          >
            {form.getValues("identity_provider") === "microsoft_entra"
              ? t("agents.addForm.assignKey.skip.entra")
              : t("agents.addForm.assignKey.skip.later")}
          </button>
        </div>
      </div>
    );
  };

  const renderReadyStep = () => (
    <div className="py-6 text-center">
      <CircleCheck className="mb-4 size-12 text-success" />
      <h3 className="mb-2 text-xl font-semibold text-foreground">{t("agents.addForm.ready.title")}</h3>
      <div className="mb-4 flex justify-center">
        <Badge className="h-auto gap-1.5 bg-purple-100 px-3 py-1 text-sm text-purple-700 dark:bg-purple-950 dark:text-purple-300">
          <Bot className="size-3.5" />
          {createdAgentName}
        </Badge>
      </div>
      {createdKeyValue && (
        <div className="mx-auto mt-4 max-w-md text-left">
          <CreatedKeyDisplay apiKey={createdKeyValue} />
        </div>
      )}
      {assignedKeyAlias && (
        <p className="mt-2 text-sm text-muted-foreground">
          <Trans
            i18nKey="agents.addForm.ready.keyAssigned"
            components={{ alias: <span className="font-medium">{assignedKeyAlias}</span> }}
          />
        </p>
      )}
      {!createdKeyValue && !assignedKeyAlias && keyAssignOption === "skip" && (
        <p className="mt-2 text-sm text-muted-foreground">
          {form.getValues("identity_provider") === "microsoft_entra"
            ? t("agents.addForm.ready.entraConfigured")
            : t("agents.addForm.ready.noKey")}
        </p>
      )}
    </div>
  );

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[900px]">
        <DialogHeader>
          <div className="flex items-center space-x-3 border-b border-border pb-4">
            {selectedLogo && currentStep < 1 && (
              <Logo
                src={selectedLogo}
                label={t("agents.addForm.dialog.logoLabel")}
                className="h-6 w-6 object-contain"
              />
            )}
            <DialogTitle className="text-xl font-semibold text-foreground">
              {t("agents.addForm.dialog.title")}
            </DialogTitle>
          </div>
        </DialogHeader>
        <TooltipProvider>
          <div className="mt-4">
            <StepProgress current={currentStep} />

            <FormProvider {...form}>
              <form onSubmit={(event) => event.preventDefault()} className="space-y-4">
                {currentStep === 0 && renderConfigureStep()}
                {currentStep === 1 && renderEntitlementsStep()}
                {currentStep === 2 && renderObservabilityStep()}
                {currentStep === 3 && renderAssignKeyStep()}
                {currentStep === 4 && renderReadyStep()}
              </form>
            </FormProvider>

            <div className="mt-6 flex items-center justify-between border-t border-border pt-6">
              <div>
                {currentStep > 0 && currentStep < 4 && (
                  <Button type="button" variant="outline" onClick={handleBack}>
                    {t("agents.addForm.dialog.back")}
                  </Button>
                )}
              </div>
              <div className="flex gap-3">
                {currentStep < 4 && (
                  <Button variant="secondary" onClick={handleClose}>
                    {t("agents.addForm.dialog.cancel")}
                  </Button>
                )}
                {currentStep < 3 && <Button onClick={handleNext}>{t("agents.addForm.dialog.next")}</Button>}
                {currentStep === 3 && (
                  <Button disabled={isSubmitting} aria-busy={isSubmitting} onClick={handleCreateAgent}>
                    {isSubmitting && <UiLoadingSpinner className="size-4" />}
                    {isSubmitting ? t("agents.addForm.dialog.creating") : t("agents.addForm.dialog.create")}
                  </Button>
                )}
                {currentStep === 4 && <Button onClick={handleClose}>{t("agents.addForm.dialog.done")}</Button>}
              </div>
            </div>
          </div>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddAgentForm;
