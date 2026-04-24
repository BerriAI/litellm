import React, { useState, useEffect } from "react";
import {
  Controller,
  FormProvider,
  useForm,
  UseFormReturn,
} from "react-hook-form";
import MessageManager from "@/components/molecules/message_manager";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Check,
  CheckCircle2,
  Info,
  Key as KeyIcon,
  LayoutGrid,
  Bot,
  X,
} from "lucide-react";
import CreatedKeyDisplay from "../shared/CreatedKeyDisplay";
import {
  createAgentCall,
  getAgentCreateMetadata,
  getAgentsList,
  keyCreateForAgentCall,
  keyListCall,
  keyUpdateCall,
  modelAvailableCall,
  AgentCreateInfo,
} from "../networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { Team } from "../key_team_helpers/key_list";
import TeamDropdown from "../common_components/team_dropdown";
import AgentFormFields from "./agent_form_fields";
import DynamicAgentFormFields, { buildDynamicAgentData } from "./dynamic_agent_form_fields";
import { getDefaultFormValues, buildAgentDataFromForm } from "./agent_config";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import GuardrailSelector from "../guardrails/GuardrailSelector";

const CUSTOM_AGENT_TYPE = "custom";

interface AddAgentFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
  teams?: Team[] | null;
}

const STEP_LABELS = [
  "Configure",
  "Entitlements",
  "Governance",
  "Agent Management",
  "Ready",
];

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="ml-1 inline h-3 w-3 text-muted-foreground" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

function Stepper({ current, steps }: { current: number; steps: string[] }) {
  return (
    <ol className="flex items-center gap-2 mb-8">
      {steps.map((label, i) => {
        const active = i === current;
        const completed = i < current;
        return (
          <li
            key={label}
            className="flex items-center gap-2 flex-1 min-w-0"
          >
            <div
              className={
                "flex h-6 w-6 items-center justify-center rounded-full border text-xs font-medium " +
                (completed
                  ? "bg-primary text-primary-foreground border-primary"
                  : active
                    ? "border-primary text-primary"
                    : "border-border text-muted-foreground")
              }
            >
              {completed ? <Check className="h-3 w-3" /> : i + 1}
            </div>
            <span
              className={
                "text-sm truncate " +
                (active || completed
                  ? "text-foreground"
                  : "text-muted-foreground")
              }
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <div className="h-px flex-1 bg-border" />
            )}
          </li>
        );
      })}
    </ol>
  );
}

/**
 * Shadcn-Select-backed multi-select with chip list under the trigger. Shares
 * the pattern with AccessGroupBaseForm.tsx.
 */
function ChipMultiSelect({
  value,
  onChange,
  options,
  placeholder,
  emptyText = "No options",
  loading = false,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  emptyText?: string;
  loading?: boolean;
}) {
  const selected = value ?? [];
  const remaining = options.filter((o) => !selected.includes(o.value));
  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={loading ? "Loading…" : placeholder} />
        </SelectTrigger>
        <SelectContent>
          {loading ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              Loading…
            </div>
          ) : remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              {emptyText}
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

/**
 * Free-form tag entry (models entitlement accepts arbitrary strings).
 * Variant that also accepts an options list for suggestions.
 */
function FreeformTagsWithOptions({
  value,
  onChange,
  options,
  placeholder,
  loading = false,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  loading?: boolean;
}) {
  const [draft, setDraft] = useState("");
  const selected = value ?? [];
  const remaining = options.filter((o) => !selected.includes(o.value));
  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          placeholder={placeholder}
          value={draft}
          onChange={(e) => {
            const v = e.target.value;
            if (v.includes(",")) {
              const parts = v
                .split(",")
                .map((s) => s.trim())
                .filter((s) => s && !selected.includes(s));
              if (parts.length > 0) onChange([...selected, ...parts]);
              setDraft("");
            } else {
              setDraft(v);
            }
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              const v = draft.trim();
              if (v && !selected.includes(v)) {
                onChange([...selected, v]);
              }
              setDraft("");
            }
          }}
          onBlur={() => {
            const v = draft.trim();
            if (v && !selected.includes(v)) {
              onChange([...selected, v]);
            }
            setDraft("");
          }}
        />
        <Select
          value=""
          onValueChange={(v) => {
            if (v) onChange([...selected, v]);
          }}
        >
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Pick…" />
          </SelectTrigger>
          <SelectContent>
            {loading ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                Loading…
              </div>
            ) : remaining.length === 0 ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                No options
              </div>
            ) : (
              remaining.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
      </div>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

type FormValues = Record<string, any>;

const getInitialFormValues = () => ({
  ...getDefaultFormValues(),
  allowed_mcp_servers_and_groups: { servers: [], accessGroups: [] },
  mcp_tool_permissions: {},
  entitlement_models: [],
  entitlement_agents: [],
  guardrails: [],
});

const AddAgentForm: React.FC<AddAgentFormProps> = ({
  visible,
  onClose,
  accessToken,
  onSuccess,
}) => {
  const { userId, userRole } = useAuthorized();
  const form: UseFormReturn<FormValues> = useForm<FormValues>({
    defaultValues: getInitialFormValues(),
  });
  const { register, control, watch, trigger, getValues, reset, formState } =
    form;
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [agentType, setAgentType] = useState<string>("a2a");
  const [agentTypeMetadata, setAgentTypeMetadata] = useState<AgentCreateInfo[]>([]);
  const [, setLoadingMetadata] = useState(false);

  const [keyAssignOption, setKeyAssignOption] = useState<"create_new" | "existing_key" | "skip">("create_new");
  const [newKeyName, setNewKeyName] = useState<string>("");
  const [newKeyModels] = useState<string[]>([]);
  const [existingKeys, setExistingKeys] = useState<any[]>([]);
  const [selectedExistingKey, setSelectedExistingKey] = useState<string | null>(null);
  const [loadingKeys, setLoadingKeys] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [availableAgents, setAvailableAgents] = useState<
    { agent_id: string; agent_name: string }[]
  >([]);
  const [loadingAgents, setLoadingAgents] = useState(false);

  const [createdAgentName, setCreatedAgentName] = useState<string>("");
  const [createdKeyValue, setCreatedKeyValue] = useState<string | null>(null);
  const [assignedKeyAlias, setAssignedKeyAlias] = useState<string | null>(null);

  const [requireTraceIdInbound, setRequireTraceIdInbound] = useState(false);
  const [requireTraceIdOutbound, setRequireTraceIdOutbound] = useState(false);
  const [maxIterations, setMaxIterations] = useState<number | null>(null);
  const [maxBudgetPerSession, setMaxBudgetPerSession] = useState<number | null>(null);

  useEffect(() => {
    const fetchMetadata = async () => {
      setLoadingMetadata(true);
      try {
        const metadata = await getAgentCreateMetadata();
        setAgentTypeMetadata(metadata);
      } catch (error) {
        console.error("Error fetching agent metadata:", error);
      } finally {
        setLoadingMetadata(false);
      }
    };
    fetchMetadata();
  }, []);

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
  }, [currentStep, accessToken, existingKeys.length]);

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
    return () => { cancelled = true; };
  }, [currentStep, accessToken]);

  const selectedAgentTypeInfo = agentTypeMetadata.find(
    (info) => info.agent_type === agentType,
  );

  const handleNext = async () => {
    if (currentStep === 0) {
      const ok = await trigger(["agent_name"]);
      if (!ok) return;
      const agentName = getValues("agent_name");
      if (agentName && !newKeyName) {
        setNewKeyName(`${agentName}-key`);
      }
    }
    setCurrentStep((s) => s + 1);
  };

  const handleBack = () => {
    setCurrentStep((s) => Math.max(0, s - 1));
  };

  const buildAgentData = (values: FormValues): any => {
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
    } else if (agentType === "a2a") {
      return buildAgentDataFromForm(values);
    } else if (selectedAgentTypeInfo?.use_a2a_form_fields) {
      const agentData = buildAgentDataFromForm(values);
      if (selectedAgentTypeInfo.litellm_params_template) {
        agentData.litellm_params = {
          ...agentData.litellm_params,
          ...selectedAgentTypeInfo.litellm_params_template,
        };
      }
      for (const field of selectedAgentTypeInfo.credential_fields) {
        const value = values[field.key];
        if (value && field.include_in_litellm_params !== false) {
          if (!agentData.litellm_params) agentData.litellm_params = {};
          agentData.litellm_params[field.key] = value;
        }
      }
      return agentData;
    } else if (selectedAgentTypeInfo) {
      return buildDynamicAgentData(values, selectedAgentTypeInfo);
    }
    return null;
  };

  const handleCreateAgent = async () => {
    if (!accessToken) {
      MessageManager.error("No access token available");
      return;
    }

    setIsSubmitting(true);
    try {
      const ok = await trigger();
      if (!ok) {
        setIsSubmitting(false);
        return;
      }
      const values = { ...getValues() };
      const agentData = buildAgentData(values);
      if (!agentData) {
        MessageManager.error("Failed to build agent data");
        setIsSubmitting(false);
        return;
      }

      const mcpServersAndGroups = values.allowed_mcp_servers_and_groups;
      const mcpToolPermissions = values.mcp_tool_permissions || {};
      const entitlementModels = values.entitlement_models || [];
      const entitlementAgents = values.entitlement_agents || [];
      const hasObjectPermission =
        (mcpServersAndGroups?.servers?.length > 0 || mcpServersAndGroups?.accessGroups?.length > 0) ||
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
          MessageManager.error("Please select an existing key to assign");
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
      MessageManager.error(errorMessage ? `Failed to create agent: ${errorMessage}` : "Failed to create agent");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    reset(getInitialFormValues());
    setAgentType("a2a");
    setCurrentStep(0);
    setKeyAssignOption("create_new");
    setNewKeyName("");
    setSelectedExistingKey(null);
    setCreatedAgentName("");
    setCreatedKeyValue(null);
    setAssignedKeyAlias(null);
    setRequireTraceIdInbound(false);
    setRequireTraceIdOutbound(false);
    setMaxIterations(null);
    setMaxBudgetPerSession(null);
    onClose();
  };

  const handleAgentTypeChange = (value: string) => {
    setAgentType(value);
    reset(getInitialFormValues());
  };

  const isCustomAgent = agentType === CUSTOM_AGENT_TYPE;
  const selectedLogo = isCustomAgent
    ? null
    : selectedAgentTypeInfo?.logo_url ||
      agentTypeMetadata.find((a) => a.agent_type === "a2a")?.logo_url;

  const agentNameError = (formState.errors as any)?.agent_name;

  const renderConfigureStep = () => (
    <>
      <div className="space-y-2">
        <Label>
          Agent Type <span className="text-destructive">*</span>
          <InfoTip>Select the type of agent you want to create</InfoTip>
        </Label>
        <Select value={agentType} onValueChange={handleAgentTypeChange}>
          <SelectTrigger>
            <SelectValue placeholder="Select an agent type" />
          </SelectTrigger>
          <SelectContent>
            {agentTypeMetadata.map((info) => (
              <SelectItem key={info.agent_type} value={info.agent_type}>
                <div className="flex items-center gap-3 py-1">
                  <img
                    src={info.logo_url || ""}
                    alt=""
                    className="w-4 h-4 object-contain"
                  />
                  <div>
                    <div className="font-medium">
                      {info.agent_type_display_name}
                    </div>
                    {info.description && (
                      <div className="text-xs text-muted-foreground">
                        {info.description}
                      </div>
                    )}
                  </div>
                </div>
              </SelectItem>
            ))}
            <div className="px-2 py-1 border-t border-border mt-1">
              <div className="text-xs text-muted-foreground font-medium mb-1 uppercase tracking-wide px-2">
                Not listed?
              </div>
              <SelectItem value={CUSTOM_AGENT_TYPE}>
                <div className="flex items-center gap-3">
                  <LayoutGrid className="h-4 w-4 text-amber-600" />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-amber-700">
                        Custom / Other
                      </span>
                      <Badge variant="outline" className="text-[10px] px-1 py-0">
                        GENERIC
                      </Badge>
                    </div>
                    <div className="text-xs text-amber-600">
                      For agents that don&apos;t follow a standard protocol —
                      just needs a virtual key
                    </div>
                  </div>
                </div>
              </SelectItem>
            </div>
          </SelectContent>
        </Select>
      </div>

      <div className="mt-4">
        {agentType === CUSTOM_AGENT_TYPE ? (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="custom_agent_name">
                Agent Name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="custom_agent_name"
                placeholder="e.g. my-custom-agent"
                aria-invalid={!!agentNameError}
                {...register("agent_name", {
                  required: "Please enter an agent name",
                })}
              />
              {agentNameError && (
                <p className="text-sm text-destructive">
                  {agentNameError.message as string}
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="custom_description">Description</Label>
              <Textarea
                id="custom_description"
                placeholder="Describe what this agent does…"
                rows={3}
                {...register("description")}
              />
            </div>
          </div>
        ) : agentType === "a2a" ? (
          <AgentFormFields showAgentName={true} />
        ) : selectedAgentTypeInfo?.use_a2a_form_fields ? (
          <>
            <AgentFormFields showAgentName={true} />
            {selectedAgentTypeInfo.credential_fields.length > 0 && (
              <div className="mt-4 p-4 border border-border rounded-lg">
                <h4 className="text-sm font-medium text-foreground mb-3">
                  {selectedAgentTypeInfo.agent_type_display_name} Settings
                </h4>
                <div className="space-y-4">
                  {selectedAgentTypeInfo.credential_fields.map((field) => {
                    const fieldError = (formState.errors as any)?.[field.key];
                    const inputId = `credential-${field.key}`;
                    return (
                      <div key={field.key} className="space-y-2">
                        <Label htmlFor={inputId}>
                          {field.label}
                          {field.required && (
                            <span className="text-destructive"> *</span>
                          )}
                          {field.tooltip ? (
                            <InfoTip>{field.tooltip}</InfoTip>
                          ) : null}
                        </Label>
                        <Input
                          id={inputId}
                          type={
                            field.field_type === "password" ? "password" : undefined
                          }
                          placeholder={field.placeholder || ""}
                          aria-invalid={!!fieldError}
                          defaultValue={field.default_value ?? undefined}
                          {...register(
                            field.key,
                            field.required
                              ? { required: `Please enter ${field.label}` }
                              : undefined,
                          )}
                        />
                        {fieldError && (
                          <p className="text-sm text-destructive">
                            {fieldError.message as string}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        ) : selectedAgentTypeInfo ? (
          <DynamicAgentFormFields agentTypeInfo={selectedAgentTypeInfo} />
        ) : null}
      </div>
    </>
  );

  const renderEntitlementsStep = () => (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Configure which models, agents, and MCP tools this agent is allowed to
        use. Leave fields empty to allow all (subject to key/team permissions).
      </p>

      <div className="space-y-2">
        <Label>
          Allowed Models
          <InfoTip>
            Restrict which models this agent can call. Leave empty to allow
            all.
          </InfoTip>
        </Label>
        <Controller
          control={control}
          name="entitlement_models"
          render={({ field }) => (
            <FreeformTagsWithOptions
              value={field.value ?? []}
              onChange={field.onChange}
              options={availableModels.map((m) => ({
                label: getModelDisplayName(m),
                value: m,
              }))}
              placeholder={
                loadingModels
                  ? "Loading models..."
                  : "Type or pick models (leave empty for all)"
              }
              loading={loadingModels}
            />
          )}
        />
      </div>

      <div className="space-y-2">
        <Label>
          Allowed Agents (Sub-Agents)
          <InfoTip>
            Restrict which other agents this agent can invoke as sub-agents.
            Leave empty to allow all.
          </InfoTip>
        </Label>
        <Controller
          control={control}
          name="entitlement_agents"
          render={({ field }) => (
            <ChipMultiSelect
              value={field.value ?? []}
              onChange={field.onChange}
              options={availableAgents.map((a) => ({
                label: a.agent_name,
                value: a.agent_id,
              }))}
              placeholder={
                loadingAgents
                  ? "Loading agents..."
                  : "Select agents (leave empty for all)"
              }
              loading={loadingAgents}
            />
          )}
        />
      </div>

      <Separator className="my-2" />

      <div className="space-y-2">
        <Label>
          Allowed MCP Servers
          <InfoTip>
            Select which MCP servers or access groups this agent can access
          </InfoTip>
        </Label>
        <Controller
          control={control}
          name="allowed_mcp_servers_and_groups"
          render={({ field }) => (
            <MCPServerSelector
              onChange={(val: { servers?: string[]; accessGroups?: string[] }) =>
                field.onChange(val)
              }
              value={field.value || { servers: [], accessGroups: [] }}
              accessToken={accessToken ?? ""}
              placeholder="Select MCP servers or access groups (optional)"
            />
          )}
        />
      </div>

      <Controller
        control={control}
        name="mcp_tool_permissions"
        render={({ field }) => {
          const selectedServers =
            watch("allowed_mcp_servers_and_groups")?.servers ?? [];
          return (
            <div className="mt-4">
              <MCPToolPermissions
                accessToken={accessToken ?? ""}
                selectedServers={selectedServers}
                toolPermissions={field.value ?? {}}
                onChange={(toolPerms: Record<string, string[]>) =>
                  field.onChange(toolPerms)
                }
              />
            </div>
          );
        }}
      />
    </div>
  );

  const renderObservabilityStep = () => (
    <div className="space-y-6">
      <div>
        <h4 className="text-sm font-medium text-foreground mb-3">Tracing</h4>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-foreground">
                Require x-litellm-trace-id on calls TO this agent
              </span>
              <p className="text-xs text-muted-foreground mt-1">
                Only accept this agent being invoked with a trace-id (e.g.
                when used as a sub-agent).
              </p>
            </div>
            <Switch
              checked={requireTraceIdInbound}
              onCheckedChange={setRequireTraceIdInbound}
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-foreground">
                Require x-litellm-trace-id on calls BY this agent
              </span>
              <p className="text-xs text-muted-foreground mt-1">
                Requires LLM/MCP calls made by this agent to include
                x-litellm-trace-id for session tracking.
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

      <Separator className="my-0" />

      <div>
        <h4 className="text-sm font-medium text-foreground mb-3">
          Budgets &amp; Rate Limits
        </h4>
        <div className="space-y-4">
          {!requireTraceIdOutbound && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-800">
              Enable &quot;Require x-litellm-trace-id on calls BY this
              agent&quot; in Tracing to configure budgets and rate limits.
            </div>
          )}

          <div className="text-sm font-medium text-foreground">
            Session Budgets
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-muted-foreground block mb-1">
                Max Iterations
              </label>
              <Input
                type="number"
                min={1}
                placeholder="e.g. 25"
                disabled={!requireTraceIdOutbound}
                value={maxIterations ?? ""}
                onChange={(e) =>
                  setMaxIterations(
                    e.target.value === "" ? null : Number(e.target.value),
                  )
                }
              />
              <p className="text-xs text-muted-foreground mt-1">
                Hard cap on LLM calls per session
              </p>
            </div>
            <div>
              <label className="text-sm text-muted-foreground block mb-1">
                Max Budget Per Session ($)
              </label>
              <Input
                type="number"
                min={0.01}
                step={0.5}
                placeholder="e.g. 5.00"
                disabled={!requireTraceIdOutbound}
                value={maxBudgetPerSession ?? ""}
                onChange={(e) =>
                  setMaxBudgetPerSession(
                    e.target.value === "" ? null : Number(e.target.value),
                  )
                }
              />
              <p className="text-xs text-muted-foreground mt-1">
                Max spend per trace before returning 429
              </p>
            </div>
          </div>

          <Separator className="my-2" />

          <div className="text-sm font-medium text-foreground">
            Agent Rate Limits
          </div>
          <p className="text-xs text-muted-foreground">
            Global rate limits applied across all callers of this agent.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="tpm_limit">TPM Limit</Label>
              <Input
                id="tpm_limit"
                type="number"
                min={0}
                placeholder="e.g. 100000"
                disabled={!requireTraceIdOutbound}
                {...register("tpm_limit", { valueAsNumber: true })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="rpm_limit">RPM Limit</Label>
              <Input
                id="rpm_limit"
                type="number"
                min={0}
                placeholder="e.g. 100"
                disabled={!requireTraceIdOutbound}
                {...register("rpm_limit", { valueAsNumber: true })}
              />
            </div>
          </div>

          <div className="text-sm font-medium text-foreground mt-4">
            Per-Session Rate Limits
          </div>
          <p className="text-xs text-muted-foreground">
            Rate limits per session (x-litellm-trace-id). Each session gets
            its own counters.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="session_tpm_limit">Session TPM Limit</Label>
              <Input
                id="session_tpm_limit"
                type="number"
                min={0}
                placeholder="e.g. 10000"
                disabled={!requireTraceIdOutbound}
                {...register("session_tpm_limit", { valueAsNumber: true })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="session_rpm_limit">Session RPM Limit</Label>
              <Input
                id="session_rpm_limit"
                type="number"
                min={0}
                placeholder="e.g. 20"
                disabled={!requireTraceIdOutbound}
                {...register("session_rpm_limit", { valueAsNumber: true })}
              />
            </div>
          </div>
        </div>
      </div>

      <Separator className="my-0" />

      <div>
        <h4 className="text-sm font-medium text-foreground mb-3">
          Guardrails
        </h4>
        <p className="text-xs text-muted-foreground mb-3">
          Apply guardrails to this agent. Selected guardrails will run on all
          calls made by this agent.
        </p>
        <Controller
          control={control}
          name="guardrails"
          render={({ field }) => (
            <GuardrailSelector
              accessToken={accessToken ?? ""}
              value={field.value ?? []}
              onChange={(selected: string[]) => field.onChange(selected)}
            />
          )}
        />
      </div>
    </div>
  );

  const renderAssignKeyStep = () => {
    const agentName = watch("agent_name") || "your-agent";
    return (
      <div>
        <div className="flex justify-center mb-6">
          <Badge
            variant="secondary"
            className="px-3 py-1 text-sm gap-1 bg-purple-50 text-purple-700 border border-purple-200"
          >
            <Bot className="h-4 w-4" />
            {agentName}
          </Badge>
        </div>

        <div className="space-y-2 mb-4">
          <Label>
            Assign to Team
            <InfoTip>
              Optionally assign this agent to a team. The agent and its key
              will belong to the selected team.
            </InfoTip>
          </Label>
          <Controller
            control={control}
            name="team_id"
            render={({ field }) => (
              <TeamDropdown
                value={field.value as string | undefined}
                onChange={field.onChange}
              />
            )}
          />
        </div>

        <Separator className="my-4" />

        <RadioGroup
          value={keyAssignOption}
          onValueChange={(v) =>
            setKeyAssignOption(v as "create_new" | "existing_key" | "skip")
          }
        >
          <div className="space-y-3">
            <div
              className={
                "p-4 border-2 rounded-lg cursor-pointer transition-colors " +
                (keyAssignOption === "create_new"
                  ? "border-primary bg-accent"
                  : "border-border bg-card hover:border-muted-foreground/30")
              }
              onClick={() => setKeyAssignOption("create_new")}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  <RadioGroupItem
                    value="create_new"
                    id="radio-create-new"
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <KeyIcon className="h-4 w-4 text-primary" />
                      <span className="font-medium text-foreground">
                        Create a new key for this agent
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      A dedicated key scoped to this agent.
                    </p>
                    {keyAssignOption === "create_new" && (
                      <div
                        className="mt-3 space-y-3"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <div>
                          <label className="text-sm text-muted-foreground block mb-1">
                            Key Name
                          </label>
                          <Input
                            value={newKeyName}
                            onChange={(e) => setNewKeyName(e.target.value)}
                            placeholder="e.g. my-agent-key"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                <Badge
                  variant="secondary"
                  className="bg-green-50 text-green-700 border border-green-200"
                >
                  Recommended
                </Badge>
              </div>
            </div>

            <div
              className={
                "p-4 border-2 rounded-lg cursor-pointer transition-colors " +
                (keyAssignOption === "existing_key"
                  ? "border-primary bg-accent"
                  : "border-border bg-card hover:border-muted-foreground/30")
              }
              onClick={() => setKeyAssignOption("existing_key")}
            >
              <div className="flex items-start gap-3">
                <RadioGroupItem
                  value="existing_key"
                  id="radio-existing-key"
                  className="mt-1"
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <KeyIcon className="h-4 w-4 text-muted-foreground" />
                    <span className="font-medium text-foreground">
                      Assign an existing key
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">
                    Re-assign a key you already have to this agent.
                  </p>
                  {keyAssignOption === "existing_key" && (
                    <div
                      className="mt-3"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Select
                        value={selectedExistingKey ?? ""}
                        onValueChange={(v) => setSelectedExistingKey(v)}
                      >
                        <SelectTrigger>
                          <SelectValue
                            placeholder={
                              loadingKeys
                                ? "Loading keys…"
                                : "Search by key name…"
                            }
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {existingKeys.map((k) => (
                            <SelectItem key={k.token} value={k.token}>
                              {k.key_alias || k.token?.slice(0, 12) + "…"}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </RadioGroup>

        <div className="text-center mt-4">
          <button
            type="button"
            className="text-sm text-muted-foreground underline hover:text-foreground"
            onClick={() => setKeyAssignOption("skip")}
          >
            Skip for now — I&apos;ll assign a key later
          </button>
        </div>
      </div>
    );
  };

  const renderReadyStep = () => (
    <div className="text-center py-6">
      <CheckCircle2
        className="mx-auto mb-4 text-green-500"
        style={{ width: 48, height: 48 }}
      />
      <h3 className="text-xl font-semibold text-foreground mb-2">
        Agent Created!
      </h3>
      <div className="flex justify-center mb-4">
        <Badge
          variant="secondary"
          className="px-3 py-1 text-sm gap-1 bg-purple-50 text-purple-700 border border-purple-200"
        >
          <Bot className="h-4 w-4" />
          {createdAgentName}
        </Badge>
      </div>
      {createdKeyValue && (
        <div className="mt-4 text-left max-w-md mx-auto">
          <CreatedKeyDisplay apiKey={createdKeyValue} />
        </div>
      )}
      {assignedKeyAlias && (
        <p className="text-sm text-muted-foreground mt-2">
          Key <span className="font-medium">{assignedKeyAlias}</span> has been
          assigned to this agent.
        </p>
      )}
      {!createdKeyValue && !assignedKeyAlias && keyAssignOption === "skip" && (
        <p className="text-sm text-muted-foreground mt-2">
          No key assigned. You can create one from the Virtual Keys page.
        </p>
      )}
    </div>
  );

  return (
    <Dialog
      open={visible}
      onOpenChange={(open) => {
        if (!open) handleClose();
      }}
    >
      <DialogContent className="max-w-[900px] top-8 translate-y-0 max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            <div className="flex items-center space-x-3 pb-4 border-b border-border">
              {selectedLogo && currentStep < 1 && (
                <img
                  src={selectedLogo}
                  alt="Agent"
                  className="w-6 h-6 object-contain"
                />
              )}
              <span className="text-xl font-semibold text-foreground">
                Add New Agent
              </span>
            </div>
          </DialogTitle>
        </DialogHeader>
        <div className="mt-4">
          <Stepper current={currentStep} steps={STEP_LABELS} />

          <FormProvider {...form}>
            <form onSubmit={(e) => e.preventDefault()} className="space-y-4">
              {currentStep === 0 && renderConfigureStep()}
              {currentStep === 1 && renderEntitlementsStep()}
              {currentStep === 2 && renderObservabilityStep()}
              {currentStep === 3 && renderAssignKeyStep()}
              {currentStep === 4 && renderReadyStep()}
            </form>
          </FormProvider>

          <div className="flex items-center justify-between pt-6 border-t border-border mt-6">
            <div>
              {currentStep > 0 && currentStep < 4 && (
                <Button type="button" variant="outline" onClick={handleBack}>
                  ← Back
                </Button>
              )}
            </div>
            <div className="flex gap-3">
              {currentStep < 4 && (
                <Button variant="outline" onClick={handleClose}>
                  Cancel
                </Button>
              )}
              {currentStep === 0 && (
                <Button onClick={handleNext}>Next →</Button>
              )}
              {currentStep === 1 && (
                <Button onClick={handleNext}>Next →</Button>
              )}
              {currentStep === 2 && (
                <Button onClick={handleNext}>Next →</Button>
              )}
              {currentStep === 3 && (
                <Button disabled={isSubmitting} onClick={handleCreateAgent}>
                  {isSubmitting ? "Creating..." : "Create Agent →"}
                </Button>
              )}
              {currentStep === 4 && <Button onClick={handleClose}>Done</Button>}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default AddAgentForm;
