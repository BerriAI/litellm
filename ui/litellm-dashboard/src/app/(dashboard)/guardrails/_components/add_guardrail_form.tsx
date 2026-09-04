import React, { useEffect, useMemo, useState } from "react";
import { useForm, type UseFormReturn } from "react-hook-form";
import { toast } from "@/lib/toast";
import {
  createGuardrailCall,
  getGuardrailProviderSpecificParams,
  getGuardrailUISettings,
  modelAvailableCall,
} from "@/components/networking";
import ContentFilterConfiguration from "./content_filter/ContentFilterConfiguration";
import { type CompetitorIntentConfig } from "./content_filter/CompetitorIntentConfiguration";
import {
  choiceToSkipSystemForCreate,
  choiceToSkipToolForCreate,
  getGuardrailLogo,
  getGuardrailProviders,
  getSupportedModesForProvider,
  guardrail_provider_map,
  populateGuardrailProviderMap,
  populateGuardrailProviders,
  shouldRenderContentFilterConfigSettings,
  shouldRenderLLMJudgeFields,
  shouldRenderPIIConfigSettings,
  toModeArray,
} from "./guardrail_info_helpers";
import { Logo } from "@/components/molecules/logo/Logo";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { FieldGroup } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { TooltipProvider } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import {
  asStringArray,
  asText,
  GuardrailField,
  labelWithHint,
  readRecord,
  requiredRule,
  type GuardrailCriterion,
  type GuardrailFormValues,
  SkipMessageSelect,
} from "./GuardrailFormField";
import GuardrailOptionalParams from "./guardrail_optional_params";
import GuardrailProviderFields from "./guardrail_provider_fields";
import LLMJudgeFields from "./llm_judge/LLMJudgeFields";
import PiiConfiguration from "./pii_configuration";
import ToolPermissionRulesEditor, { ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";

// Define human-friendly descriptions for each mode
const modeDescriptions = {
  pre_call: "Before LLM Call - Runs before the LLM call and checks the input (Recommended)",
  during_call: "During LLM Call - Runs in parallel with the LLM call, with response held until check completes",
  post_call: "After LLM Call - Runs after the LLM call and checks only the output",
  logging_only: "Logging Only - Only runs on logging callbacks without affecting the LLM call",
  pre_mcp_call: "Before MCP Tool Call - Runs before MCP tool execution and validates tool calls",
  during_mcp_call: "During MCP Tool Call - Runs in parallel with MCP tool execution for monitoring",
  post_mcp_call: "After MCP Tool Call - Runs after MCP tool execution and checks the tool result",
};

interface GuardrailPreset {
  provider: string;
  categoryName?: string;
  guardrailNameSuggestion: string;
  mode: string;
  defaultOn: boolean;
}

interface AddGuardrailFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
  preset?: GuardrailPreset;
}

interface GuardrailSettings {
  supported_entities: string[];
  supported_actions: string[];
  supported_modes: string[];
  supported_modes_by_provider?: Record<string, string[]>;
  pii_entity_categories: Array<{
    category: string;
    entities: string[];
  }>;
  content_filter_settings?: {
    prebuilt_patterns: Array<{
      name: string;
      display_name: string;
      category: string;
      description: string;
    }>;
    pattern_categories: string[];
    supported_actions: string[];
    content_categories?: Array<{
      name: string;
      display_name: string;
      description: string;
      default_action: string;
    }>;
  };
}

interface ContentFilterPattern {
  id: string;
  type: "prebuilt" | "custom";
  name: string;
  display_name?: string;
  pattern?: string;
  action: "BLOCK" | "MASK";
}

interface ContentFilterBlockedWord {
  id: string;
  keyword: string;
  action: "BLOCK" | "MASK";
  description?: string;
}

interface SelectedContentCategory {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

const createEmptyToolPermissionConfig = (): ToolPermissionConfig => ({
  rules: [],
  default_action: "deny",
  on_disallowed_action: "block",
  violation_message_template: "",
});

const getStepIndicatorClass = (isDone: boolean, isCurrent: boolean): string => {
  if (isDone) return "bg-info text-info-foreground";
  if (isCurrent) return "bg-background text-info border-2 border-info";
  return "bg-muted text-muted-foreground border border-border";
};

const getStepTitleClass = (isDone: boolean, isCurrent: boolean): string => {
  if (isCurrent) return "font-semibold text-foreground";
  if (isDone) return "font-medium text-info";
  return "font-medium text-muted-foreground";
};

type SkipMessageChoice = "inherit" | "yes" | "no";

const INITIAL_VALUES: GuardrailFormValues = {
  mode: "pre_call",
  default_on: false,
  skip_system_message_choice: "inherit",
  skip_tool_message_choice: "inherit",
};

const ALWAYS_ON_ITEMS = [
  { label: "Yes", value: true },
  { label: "No", value: false },
];

const DEFAULT_MODES = ["pre_call", "during_call", "post_call", "logging_only"];

const CALL_TYPE_ITEMS = [{ label: "/v1/realtime", value: "realtime" }];

const applyValues = (form: UseFormReturn<GuardrailFormValues>, values: Record<string, unknown>) => {
  Object.entries(values).forEach(([name, value]) => form.setValue(name, value));
};

const asSkipChoice = (value: unknown): SkipMessageChoice | undefined =>
  value === "inherit" || value === "yes" || value === "no" ? value : undefined;

// Mapping of provider -> list of param descriptors
interface ProviderParam {
  param: string;
  description: string;
  required: boolean;
  default_value?: string;
  options?: string[];
  type?: string;
  fields?: { [key: string]: ProviderParam };
  dict_key_options?: string[];
  dict_value_type?: string;
}

interface ProviderParamsResponse {
  [provider: string]: { [key: string]: ProviderParam };
}

const AddGuardrailForm: React.FC<AddGuardrailFormProps> = ({ visible, onClose, accessToken, onSuccess, preset }) => {
  const form = useForm<GuardrailFormValues>({ defaultValues: INITIAL_VALUES });
  const [loading, setLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [guardrailSettings, setGuardrailSettings] = useState<GuardrailSettings | null>(null);
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [selectedActions, setSelectedActions] = useState<{ [key: string]: string }>({});
  const [currentStep, setCurrentStep] = useState(0);
  const [providerParams, setProviderParams] = useState<ProviderParamsResponse | null>(null);

  // Content Filter state
  const [selectedPatterns, setSelectedPatterns] = useState<ContentFilterPattern[]>([]);
  const [blockedWords, setBlockedWords] = useState<ContentFilterBlockedWord[]>([]);
  const [selectedContentCategories, setSelectedContentCategories] = useState<SelectedContentCategory[]>([]);
  const [pendingCategorySelection, setPendingCategorySelection] = useState<string>("");
  const [competitorIntentEnabled, setCompetitorIntentEnabled] = useState(false);
  const [competitorIntentConfig, setCompetitorIntentConfig] = useState<CompetitorIntentConfig | null>(null);

  // Endpoint Settings state (step 5)
  const [selectedEndpointType, setSelectedEndpointType] = useState<string>("");
  const [endSessionAfterNFails, setEndSessionAfterNFails] = useState<number | undefined>(undefined);
  const [onViolation, setOnViolation] = useState<"warn" | "end_session">("warn");
  const [realtimeViolationMessage, setRealtimeViolationMessage] = useState<string>("");
  const [endpointSettingsOpen, setEndpointSettingsOpen] = useState<boolean>(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  const [toolPermissionConfig, setToolPermissionConfig] = useState<ToolPermissionConfig>(
    createEmptyToolPermissionConfig,
  );

  const isToolPermissionProvider = useMemo(() => {
    if (!selectedProvider) {
      return false;
    }
    const providerValue = guardrail_provider_map[selectedProvider];
    return (providerValue || "").toLowerCase() === "tool_permission";
  }, [selectedProvider]);

  // Fetch guardrail UI settings + provider params on mount / accessToken change
  useEffect(() => {
    if (!accessToken) return;

    const fetchData = async () => {
      try {
        // Parallel requests for speed
        const [uiSettings, providerParamsResp, modelsResp] = await Promise.all([
          getGuardrailUISettings(accessToken),
          getGuardrailProviderSpecificParams(accessToken),
          modelAvailableCall(accessToken, "", "").catch(() => null),
        ]);

        setGuardrailSettings(uiSettings);
        setProviderParams(providerParamsResp);
        if (modelsResp?.data) {
          setAvailableModels(modelsResp.data.map((m: { id: string }) => m.id));
        }

        // Populate dynamic providers from API response
        populateGuardrailProviders(providerParamsResp);
        populateGuardrailProviderMap(providerParamsResp);
      } catch (error) {
        console.error("Error fetching guardrail data:", error);
        toast.fromError("Failed to load guardrail configuration");
      }
    };

    fetchData();
  }, [accessToken]);

  // Apply preset when settings are loaded and form becomes visible
  useEffect(() => {
    if (!preset || !visible || !guardrailSettings) return;

    // Set provider
    setSelectedProvider(preset.provider);
    const baseValues: Record<string, unknown> = {
      provider: preset.provider,
      guardrail_name: preset.guardrailNameSuggestion,
      mode: preset.mode,
      default_on: preset.defaultOn,
      skip_system_message_choice: "inherit",
      skip_tool_message_choice: "inherit",
    };
    if (preset.provider === "BlockCodeExecution") {
      baseValues.confidence_threshold = 0.5;
    }
    applyValues(form, baseValues);

    // Pre-select content category if specified
    if (preset.categoryName && guardrailSettings.content_filter_settings?.content_categories) {
      const category = guardrailSettings.content_filter_settings.content_categories.find(
        (c) => c.name === preset.categoryName,
      );
      if (category) {
        setSelectedContentCategories([
          {
            id: `category-${Date.now()}`,
            category: category.name,
            display_name: category.display_name,
            action: category.default_action as "BLOCK" | "MASK",
            severity_threshold: "medium",
          },
        ]);
      }
    }
  }, [preset, visible, guardrailSettings, form]);

  const handleProviderChange = (value: string) => {
    setSelectedProvider(value);
    // Reset form fields that are provider-specific
    const resetValues: Record<string, unknown> = {
      config: undefined,
      presidio_analyzer_api_base: undefined,
      presidio_anonymizer_api_base: undefined,
    };
    if (value === "BlockCodeExecution") {
      resetValues.confidence_threshold = 0.5;
    }

    // Drop selected modes the new provider does not support
    const newProviderKey = guardrail_provider_map[value]?.toLowerCase();
    const newProviderModes =
      newProviderKey && guardrailSettings?.supported_modes_by_provider
        ? guardrailSettings.supported_modes_by_provider[newProviderKey]
        : undefined;
    if (newProviderModes) {
      const selectedModes = toModeArray(form.getValues("mode"));
      const keptModes = selectedModes.filter((m) => newProviderModes.includes(m));
      if (keptModes.length !== selectedModes.length) {
        resetValues.mode = keptModes.length > 0 ? keptModes : undefined;
      }
    }

    applyValues(form, resetValues);

    // Reset PII selections when changing provider
    setSelectedEntities([]);
    setSelectedActions({});

    // Reset Content Filter selections
    setSelectedPatterns([]);
    setBlockedWords([]);
    setSelectedContentCategories([]);
    setPendingCategorySelection("");
    setCompetitorIntentEnabled(false);
    setCompetitorIntentConfig(null);

    setToolPermissionConfig(createEmptyToolPermissionConfig());

    // Default LLM-as-a-Judge to post_call mode
    if (value === "LlmAsAJudge") {
      form.setValue("mode", "post_call");
    }
  };

  const handleEntitySelect = (entity: string) => {
    setSelectedEntities((prev) => {
      if (prev.includes(entity)) {
        return prev.filter((e) => e !== entity);
      } else {
        return [...prev, entity];
      }
    });
  };

  const handleActionSelect = (entity: string, action: string) => {
    setSelectedActions((prev) => ({
      ...prev,
      [entity]: action,
    }));
  };

  const nextStep = async () => {
    // Validate current step fields
    if (currentStep === 0) {
      const presidioFields =
        selectedProvider === "PresidioPII" ? ["presidio_analyzer_api_base", "presidio_anonymizer_api_base"] : [];
      const isValid = await form.trigger(["guardrail_name", "provider", "mode", "default_on", ...presidioFields]);
      if (!isValid) {
        return;
      }
    }

    // Validate configuration steps
    if (currentStep === 1) {
      if (shouldRenderPIIConfigSettings(selectedProvider) && selectedEntities.length === 0) {
        toast.fromError("Please select at least one PII entity to continue");
        return;
      }
    }

    setCurrentStep(currentStep + 1);
  };

  const prevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  const resetForm = () => {
    form.reset(INITIAL_VALUES);
    setSelectedProvider(null);
    setSelectedEntities([]);
    setSelectedActions({});
    setSelectedPatterns([]);
    setBlockedWords([]);
    setSelectedContentCategories([]);
    setPendingCategorySelection("");
    setToolPermissionConfig(createEmptyToolPermissionConfig());
    setSelectedEndpointType("");
    setEndSessionAfterNFails(undefined);
    setOnViolation("warn");
    setRealtimeViolationMessage("");
    setEndpointSettingsOpen(false);
    setCurrentStep(0);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      // First validate currently visible fields
      if (!(await form.trigger())) {
        toast.fromError("Failed to create guardrail: please fix the highlighted fields");
        return;
      }

      // After validation, fetch *all* form values (including those from previous steps)
      const values = form.getValues();
      const providerKey = asText(values.provider);

      // Get the guardrail provider value from the map
      const guardrailProvider = guardrail_provider_map[providerKey];

      // Prepare the guardrail data with proper typings
      const guardrailData: {
        guardrail_name: string;
        litellm_params: {
          guardrail: string;
          [key: string]: unknown; // Allow dynamic properties
        };
        guardrail_info: Record<string, unknown>;
      } = {
        guardrail_name: asText(values.guardrail_name),
        litellm_params: {
          guardrail: guardrailProvider,
          mode: values.mode,
          default_on: values.default_on,
        },
        guardrail_info: {},
      };

      const skipForCreate = choiceToSkipSystemForCreate(asSkipChoice(values.skip_system_message_choice));
      if (skipForCreate !== undefined) {
        guardrailData.litellm_params.skip_system_message_in_guardrail = skipForCreate;
      }

      const skipToolForCreate = choiceToSkipToolForCreate(asSkipChoice(values.skip_tool_message_choice));
      if (skipToolForCreate !== undefined) {
        guardrailData.litellm_params.skip_tool_message_in_guardrail = skipToolForCreate;
      }

      // For Presidio PII, add the entity and action configurations
      if (providerKey === "PresidioPII" && selectedEntities.length > 0) {
        const piiEntitiesConfig: { [key: string]: string } = {};
        selectedEntities.forEach((entity) => {
          piiEntitiesConfig[entity] = selectedActions[entity] || "MASK"; // Default to MASK if no action selected
        });

        guardrailData.litellm_params.pii_entities_config = piiEntitiesConfig;

        // Add Presidio API bases if provided
        if (values.presidio_analyzer_api_base) {
          guardrailData.litellm_params.presidio_analyzer_api_base = values.presidio_analyzer_api_base;
        }
        if (values.presidio_anonymizer_api_base) {
          guardrailData.litellm_params.presidio_anonymizer_api_base = values.presidio_anonymizer_api_base;
        }
      }

      // For Content Filter, add patterns, blocked words, categories, and optionally competitor intent
      if (shouldRenderContentFilterConfigSettings(providerKey)) {
        // Validate that at least one content filter setting is configured
        const hasCompetitorIntent = competitorIntentEnabled && (competitorIntentConfig?.brand_self?.length ?? 0) > 0;
        const hasContentFilterSelections =
          selectedPatterns.length > 0 || blockedWords.length > 0 || selectedContentCategories.length > 0;
        if (!hasContentFilterSelections && !hasCompetitorIntent) {
          toast.fromError(
            "Please configure at least one content filter setting (category, pattern, keyword, or competitor intent)",
          );
          setLoading(false);
          return;
        }

        if (selectedPatterns.length > 0) {
          guardrailData.litellm_params.patterns = selectedPatterns.map((p) => ({
            pattern_type: p.type === "prebuilt" ? "prebuilt" : "regex",
            pattern_name: p.type === "prebuilt" ? p.name : undefined,
            pattern: p.type === "custom" ? p.pattern : undefined,
            name: p.name,
            action: p.action,
          }));
        }
        if (blockedWords.length > 0) {
          guardrailData.litellm_params.blocked_words = blockedWords.map((w) => ({
            keyword: w.keyword,
            action: w.action,
            description: w.description,
          }));
        }
        if (selectedContentCategories.length > 0) {
          guardrailData.litellm_params.categories = selectedContentCategories.map((c) => ({
            category: c.category,
            enabled: true,
            action: c.action,
            severity_threshold: c.severity_threshold || "medium",
          }));
        }
        if (hasCompetitorIntent && competitorIntentConfig) {
          guardrailData.litellm_params.competitor_intent_config = {
            competitor_intent_type: competitorIntentConfig.competitor_intent_type ?? "airline",
            brand_self: competitorIntentConfig.brand_self,
            locations:
              (competitorIntentConfig.locations?.length ?? 0) > 0 ? competitorIntentConfig.locations : undefined,
            competitors:
              competitorIntentConfig.competitor_intent_type === "generic" &&
              (competitorIntentConfig.competitors?.length ?? 0) > 0
                ? competitorIntentConfig.competitors
                : undefined,
            policy: competitorIntentConfig.policy,
            threshold_high: competitorIntentConfig.threshold_high,
            threshold_medium: competitorIntentConfig.threshold_medium,
            threshold_low: competitorIntentConfig.threshold_low,
          };
        }
      }
      // Add config values to the guardrail_info if provided
      else if (values.config) {
        try {
          const configObj = JSON.parse(asText(values.config));
          // For some guardrails, the config values need to be in litellm_params
          guardrailData.guardrail_info = configObj;
        } catch (error) {
          toast.fromError("Invalid JSON in configuration");
          setLoading(false);
          return;
        }
      }

      if (guardrailProvider === "llm_as_a_judge") {
        const criteria: GuardrailCriterion[] = values.criteria ?? [];
        if (criteria.length === 0) {
          toast.fromError("Add at least one evaluation criterion");
          setLoading(false);
          return;
        }
        const weightTotal = criteria.reduce((sum, c) => sum + (Number(c?.weight) || 0), 0);
        if (weightTotal !== 100) {
          toast.fromError(`Criterion weights must sum to 100% (currently ${weightTotal}%)`);
          setLoading(false);
          return;
        }
        guardrailData.litellm_params.judge_model = values.judge_model;
        guardrailData.litellm_params.overall_threshold = values.overall_threshold ?? 80;
        guardrailData.litellm_params.on_failure = values.on_failure ?? "block";
        guardrailData.litellm_params.criteria = criteria.map((c) => ({
          name: c.name,
          weight: Number(c.weight),
          description: c.description || "",
        }));
      }

      if (guardrailProvider === "tool_permission") {
        if (toolPermissionConfig.rules.length === 0) {
          toast.fromError("Add at least one tool permission rule");
          setLoading(false);
          return;
        }
        guardrailData.litellm_params.rules = toolPermissionConfig.rules;
        guardrailData.litellm_params.default_action = toolPermissionConfig.default_action;
        guardrailData.litellm_params.on_disallowed_action = toolPermissionConfig.on_disallowed_action;
        if (toolPermissionConfig.violation_message_template) {
          guardrailData.litellm_params.violation_message_template = toolPermissionConfig.violation_message_template;
        }
      }

      // Endpoint Settings (realtime) — content filter only
      if (shouldRenderContentFilterConfigSettings(providerKey)) {
        if (endSessionAfterNFails !== undefined && endSessionAfterNFails > 0) {
          guardrailData.litellm_params.end_session_after_n_fails = endSessionAfterNFails;
        }
        if (onViolation && selectedEndpointType === "realtime") {
          guardrailData.litellm_params.on_violation = onViolation;
        }
        if (realtimeViolationMessage.trim()) {
          guardrailData.litellm_params.realtime_violation_message = realtimeViolationMessage.trim();
        }
      }

      /******************************
       * Add provider-specific params
       * ----------------------------------
       * The backend exposes exactly which extra parameters a provider
       * accepts via `/guardrails/ui/provider_specific_params`.
       * Instead of copying every unknown form field, we fetch the list for
       * the selected provider and ONLY pass those recognised params.
       ******************************/

      // Use pre-fetched provider params to copy recognised params
      // Skip for providers that handle their own litellm_params (llm_as_a_judge, tool_permission, content filter, PII)
      if (providerParams && selectedProvider && guardrailProvider !== "llm_as_a_judge") {
        const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
        const providerSpecificParams = providerParams[providerKey] || {};

        const allowedParams = new Set<string>();

        // Add root-level parameters (like api_key, api_base, api_version)
        Object.keys(providerSpecificParams).forEach((paramName) => {
          if (paramName !== "optional_params") {
            allowedParams.add(paramName);
          }
        });

        // Add nested parameters from optional_params.fields
        if (providerSpecificParams.optional_params && providerSpecificParams.optional_params.fields) {
          Object.keys(providerSpecificParams.optional_params.fields).forEach((paramName) => {
            allowedParams.add(paramName);
          });
        }

        allowedParams.forEach((paramName) => {
          // Check for both direct parameter name and nested optional_params object
          const directValue = values[paramName];
          const paramValue =
            directValue === undefined || directValue === null || directValue === ""
              ? readRecord(values.optional_params, paramName)
              : directValue;

          if (paramValue !== undefined && paramValue !== null && paramValue !== "") {
            guardrailData.litellm_params[paramName] = paramValue;
          }
        });
      }

      if (!accessToken) {
        throw new Error("No access token available");
      }

      await createGuardrailCall(accessToken, guardrailData);

      toast.success("Guardrail created successfully");

      // Reset form and close modal
      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create guardrail:", error);
      toast.fromError("Failed to create guardrail: " + (error instanceof Error ? error.message : String(error)));
    } finally {
      setLoading(false);
    }
  };

  const renderBasicInfo = () => {
    const showProviderFields =
      !isToolPermissionProvider &&
      !shouldRenderContentFilterConfigSettings(selectedProvider) &&
      !shouldRenderLLMJudgeFields(selectedProvider);
    const providerLabels: Record<string, string> = getGuardrailProviders();
    const providerKeys = Object.keys(providerLabels);
    const supportedModes = getSupportedModesForProvider(guardrailSettings, selectedProvider) ?? DEFAULT_MODES;
    return (
      <FieldGroup>
        <GuardrailField
          control={form.control}
          name="guardrail_name"
          label="Guardrail Name"
          rules={requiredRule("Please enter a guardrail name")}
        >
          {({ ref, value, ...field }) => (
            <Input {...field} ref={ref} value={asText(value)} placeholder="Enter a name for this guardrail" />
          )}
        </GuardrailField>

        <GuardrailField
          control={form.control}
          name="provider"
          label="Guardrail Provider"
          rules={requiredRule("Please select a provider")}
        >
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Combobox
              items={providerKeys}
              itemToStringLabel={(key: string) => providerLabels[key] ?? key}
              value={asText(value) || null}
              onValueChange={(key: string | null) => {
                onChange(key ?? "");
                if (key) {
                  handleProviderChange(key);
                }
              }}
            >
              <ComboboxInput
                id={id}
                aria-invalid={ariaInvalid}
                aria-describedby={ariaDescribedBy}
                placeholder="Select a guardrail provider"
                className="w-full"
              />
              <ComboboxContent>
                <ComboboxEmpty>No matching providers</ComboboxEmpty>
                <ComboboxList>
                  {(key: string) => (
                    <ComboboxItem key={key} value={key}>
                      <span className="flex items-center">
                        <Logo
                          src={getGuardrailLogo(providerLabels[key])}
                          label={providerLabels[key]}
                          className="mr-2 h-5 w-5 shrink-0 object-contain"
                        />
                        <span>{providerLabels[key]}</span>
                      </span>
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          )}
        </GuardrailField>

        <GuardrailField
          control={form.control}
          name="mode"
          label={labelWithHint("Mode", "How the guardrail should be applied")}
          rules={requiredRule("Please select a mode")}
        >
          {({ id, value, onChange }) => (
            <MultiSelect
              id={id}
              options={supportedModes.map((mode) => ({
                label: mode,
                value: mode,
                description: modeDescriptions[mode as keyof typeof modeDescriptions],
              }))}
              value={asStringArray(value)}
              onValueChange={onChange}
              placeholder=""
            />
          )}
        </GuardrailField>

        <GuardrailField
          control={form.control}
          name="default_on"
          label={labelWithHint("Always On", "If enabled, this guardrail will be applied to all requests by default.")}
        >
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Select
              items={ALWAYS_ON_ITEMS}
              value={typeof value === "boolean" ? value : null}
              onValueChange={(next: boolean | null) => onChange(next)}
            >
              <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy} className="w-full">
                <SelectValue placeholder="Select an option" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={true}>Yes</SelectItem>
                <SelectItem value={false}>No</SelectItem>
              </SelectContent>
            </Select>
          )}
        </GuardrailField>

        <GuardrailField
          control={form.control}
          name="skip_system_message_choice"
          label={labelWithHint(
            "Skip system messages in guardrail",
            "Unified guardrails only: omit role: system from guardrail evaluation input (OpenAI chat + Anthropic messages). The model still receives full messages. Use global default follows litellm_settings.skip_system_message_in_guardrail.",
          )}
        >
          {(fieldControl) => <SkipMessageSelect control={fieldControl} />}
        </GuardrailField>

        <GuardrailField
          control={form.control}
          name="skip_tool_message_choice"
          label={labelWithHint(
            "Skip tool messages in guardrail",
            "Unified guardrails only: omit role: tool from guardrail evaluation input (OpenAI chat + Anthropic messages). The model still receives full messages. Use global default follows litellm_settings.skip_tool_message_in_guardrail.",
          )}
        >
          {(fieldControl) => <SkipMessageSelect control={fieldControl} />}
        </GuardrailField>

        {/* Use the GuardrailProviderFields component to render provider-specific fields */}
        {showProviderFields && (
          <GuardrailProviderFields
            selectedProvider={selectedProvider}
            control={form.control}
            accessToken={accessToken}
            providerParams={providerParams}
          />
        )}
      </FieldGroup>
    );
  };

  const renderPiiConfiguration = () => {
    if (!guardrailSettings || selectedProvider !== "PresidioPII") return null;

    return (
      <PiiConfiguration
        entities={guardrailSettings.supported_entities}
        actions={guardrailSettings.supported_actions}
        selectedEntities={selectedEntities}
        selectedActions={selectedActions}
        onEntitySelect={handleEntitySelect}
        onActionSelect={handleActionSelect}
        entityCategories={guardrailSettings.pii_entity_categories}
      />
    );
  };

  const renderContentFilterConfiguration = (step: "patterns" | "keywords" | "categories") => {
    if (!guardrailSettings || !shouldRenderContentFilterConfigSettings(selectedProvider)) return null;

    const contentFilterSettings = guardrailSettings.content_filter_settings;
    if (!contentFilterSettings) return null;

    return (
      <ContentFilterConfiguration
        prebuiltPatterns={contentFilterSettings.prebuilt_patterns || []}
        categories={contentFilterSettings.pattern_categories || []}
        selectedPatterns={selectedPatterns}
        blockedWords={blockedWords}
        onPatternAdd={(pattern) => setSelectedPatterns([...selectedPatterns, pattern])}
        onPatternRemove={(id) => setSelectedPatterns(selectedPatterns.filter((p) => p.id !== id))}
        onPatternActionChange={(id, action) => {
          setSelectedPatterns(selectedPatterns.map((p) => (p.id === id ? { ...p, action } : p)));
        }}
        onBlockedWordAdd={(word) => setBlockedWords([...blockedWords, word])}
        onBlockedWordRemove={(id) => setBlockedWords(blockedWords.filter((w) => w.id !== id))}
        onBlockedWordUpdate={(id, field, value) => {
          setBlockedWords(blockedWords.map((w) => (w.id === id ? { ...w, [field]: value } : w)));
        }}
        contentCategories={contentFilterSettings.content_categories || []}
        selectedContentCategories={selectedContentCategories}
        onContentCategoryAdd={(category) => setSelectedContentCategories([...selectedContentCategories, category])}
        onContentCategoryRemove={(id) =>
          setSelectedContentCategories(selectedContentCategories.filter((c) => c.id !== id))
        }
        onContentCategoryUpdate={(id, field, value) => {
          setSelectedContentCategories(
            selectedContentCategories.map((c) => (c.id === id ? { ...c, [field]: value } : c)),
          );
        }}
        pendingCategorySelection={pendingCategorySelection}
        onPendingCategorySelectionChange={setPendingCategorySelection}
        accessToken={accessToken}
        showStep={step}
        competitorIntentEnabled={competitorIntentEnabled}
        competitorIntentConfig={competitorIntentConfig}
        onCompetitorIntentChange={(enabled, config) => {
          setCompetitorIntentEnabled(enabled);
          setCompetitorIntentConfig(config);
        }}
      />
    );
  };

  const renderOptionalParams = () => {
    if (!selectedProvider) return null;

    if (isToolPermissionProvider) {
      return <ToolPermissionRulesEditor value={toolPermissionConfig} onChange={setToolPermissionConfig} />;
    }

    if (!providerParams) {
      return null;
    }

    const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
    const providerFields = providerParams && providerParams[providerKey];

    if (!providerFields || !providerFields.optional_params) return null;

    return (
      <GuardrailOptionalParams
        optionalParams={providerFields.optional_params}
        parentFieldKey="optional_params"
        control={form.control}
      />
    );
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return renderBasicInfo();
      case 1:
        if (shouldRenderPIIConfigSettings(selectedProvider)) {
          return renderPiiConfiguration();
        }
        if (shouldRenderContentFilterConfigSettings(selectedProvider)) {
          return renderContentFilterConfiguration("categories");
        }
        if (shouldRenderLLMJudgeFields(selectedProvider)) {
          return <LLMJudgeFields availableModels={availableModels} control={form.control} />;
        }
        return renderOptionalParams();
      case 2:
        if (shouldRenderContentFilterConfigSettings(selectedProvider)) {
          return renderContentFilterConfiguration("patterns");
        }
        return null;
      case 3:
        if (shouldRenderContentFilterConfigSettings(selectedProvider)) {
          return renderContentFilterConfiguration("keywords");
        }
        return null;
      case 4:
        return renderEndpointSettings();
      default:
        return null;
    }
  };

  const renderEndpointSettings = () => {
    return (
      <div className="space-y-6">
        <div>
          <p className="text-sm text-muted-foreground">
            Configure settings for a specific call type. Most guardrails don't need this — skip it unless you're using a
            specific endpoint like <code>/v1/realtime</code>.
          </p>
        </div>

        <div>
          <label htmlFor="guardrail-call-type" className="mb-1 block text-sm font-medium text-foreground">
            Call type
          </label>
          <Select
            items={CALL_TYPE_ITEMS}
            value={selectedEndpointType || null}
            onValueChange={(next: string | null) => {
              setSelectedEndpointType(next ?? "");
              setEndpointSettingsOpen(false);
            }}
          >
            <SelectTrigger id="guardrail-call-type" className="w-65">
              <SelectValue placeholder="Select a call type" />
            </SelectTrigger>
            <SelectContent>
              {CALL_TYPE_ITEMS.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="mt-1 text-xs text-muted-foreground">More call types coming soon.</p>
        </div>

        {selectedEndpointType === "realtime" && (
          <div className="overflow-hidden rounded-lg border border-border">
            <button
              type="button"
              onClick={() => setEndpointSettingsOpen((o) => !o)}
              className="flex w-full items-center justify-between bg-muted px-4 py-3 text-sm font-medium text-foreground hover:bg-muted/70"
            >
              <span>/v1/realtime settings</span>
              <svg
                className={`w-4 h-4 text-muted-foreground transition-transform ${endpointSettingsOpen ? "rotate-180" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {endpointSettingsOpen && (
              <div className="space-y-5 border-t border-border px-4 py-4">
                <div>
                  <label
                    htmlFor="guardrail-end-session-after"
                    className="mb-1 block text-sm font-medium text-foreground"
                  >
                    End session after X violations
                  </label>
                  <p className="mb-2 text-xs text-muted-foreground">
                    Automatically close the session after this many guardrail violations. Leave empty to never
                    auto-close.
                  </p>
                  <Input
                    id="guardrail-end-session-after"
                    type="number"
                    min={1}
                    placeholder="e.g. 3"
                    value={endSessionAfterNFails ?? ""}
                    onChange={(e) =>
                      setEndSessionAfterNFails(e.target.value ? parseInt(e.target.value, 10) : undefined)
                    }
                    className="w-32"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm font-medium text-foreground">On violation</label>
                  <div className="space-y-2">
                    {(["warn", "end_session"] as const).map((opt) => (
                      <label key={opt} className="flex items-start gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="on_violation"
                          value={opt}
                          checked={onViolation === opt}
                          onChange={() => setOnViolation(opt)}
                          className="mt-0.5"
                        />
                        <div>
                          <span className="text-sm font-medium text-foreground">
                            {opt === "warn" ? "Warn" : "End session"}
                          </span>
                          <p className="m-0 text-xs text-muted-foreground">
                            {opt === "warn"
                              ? "Bot speaks the message, session continues"
                              : "Bot speaks the message, connection closes immediately"}
                          </p>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label
                    htmlFor="guardrail-realtime-message"
                    className="mb-1 block text-sm font-medium text-foreground"
                  >
                    Message the user hears
                  </label>
                  <p className="mb-2 text-xs text-muted-foreground">
                    What the bot says aloud when this guardrail fires. Falls back to the default violation message if
                    empty.
                  </p>
                  <Textarea
                    id="guardrail-realtime-message"
                    rows={3}
                    placeholder="e.g. I'm not able to continue this conversation. Please contact us at 1-800-774-2678."
                    value={realtimeViolationMessage}
                    onChange={(e) => setRealtimeViolationMessage(e.target.value)}
                    className="w-full resize-none"
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const getStepConfigs = () => {
    if (shouldRenderContentFilterConfigSettings(selectedProvider)) {
      return [
        { title: "Basic Info", optional: false },
        { title: "Topics", optional: false },
        { title: "Patterns", optional: false },
        { title: "Keywords", optional: false },
        { title: "Endpoint Settings (Optional)", optional: true },
      ];
    }
    if (shouldRenderPIIConfigSettings(selectedProvider)) {
      return [
        { title: "Basic Info", optional: false },
        { title: "PII Configuration", optional: false },
      ];
    }
    return [
      { title: "Basic Info", optional: false },
      { title: "Provider Configuration", optional: false },
    ];
  };

  const stepConfigs = getStepConfigs();

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleClose()} disablePointerDismissal>
      <DialogContent
        className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 gap-0 overflow-hidden p-0 sm:max-w-[1000px]"
        showCloseButton={false}
      >
        <TooltipProvider>
          <div className="flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border px-6 py-4">
              <DialogTitle className="m-0 text-base font-semibold text-foreground">Create guardrail</DialogTitle>
              <button
                type="button"
                onClick={handleClose}
                className="cursor-pointer border-none bg-transparent p-1 text-base leading-none text-muted-foreground hover:text-foreground"
              >
                &#x2715;
              </button>
            </div>

            {/* Scrollable content - inline vertical stepper */}
            <div className="max-h-[calc(80vh-120px)] overflow-auto px-6 py-4">
              <form onSubmit={(event) => event.preventDefault()}>
                {stepConfigs.map((step, index) => {
                  const isDone = index < currentStep;
                  const isCurrent = index === currentStep;
                  const isLast = index === stepConfigs.length - 1;
                  return (
                    <div key={index} className={`relative flex gap-4 ${isLast ? "" : "pb-2"}`}>
                      {/* Vertical line + step indicator */}
                      <div className="flex w-6 shrink-0 flex-col items-center">
                        <div
                          className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-medium ${getStepIndicatorClass(isDone, isCurrent)}`}
                        >
                          {isDone ? "\u2713" : index + 1}
                        </div>
                        {!isLast && <div className={`min-h-4 w-px flex-1 ${isDone ? "bg-info" : "bg-border"}`} />}
                      </div>

                      {/* Step content */}
                      <div className={`min-w-0 flex-1 ${isLast ? "" : "pb-4"}`}>
                        {/* Step header - clickable for completed steps */}
                        <div
                          className={`flex min-h-6 items-center gap-2 ${isDone ? "cursor-pointer" : ""}`}
                          onClick={() => {
                            if (isDone) setCurrentStep(index);
                          }}
                        >
                          <span className={`text-sm ${getStepTitleClass(isDone, isCurrent)}`}>{step.title}</span>
                          {step.optional && !isCurrent && (
                            <span className="text-[11px] text-muted-foreground">optional</span>
                          )}
                          {isDone && <span className="text-[11px] text-info hover:underline">Edit</span>}
                        </div>

                        {/* Expanded form content for current step */}
                        {isCurrent && <div className="mt-3">{renderStepContent()}</div>}
                      </div>
                    </div>
                  );
                })}
              </form>
            </div>

            {/* Bottom bar */}
            <div className="flex items-center justify-end space-x-3 border-t border-border px-6 py-3">
              <Button type="button" variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              {currentStep > 0 && (
                <Button type="button" variant="outline" onClick={prevStep}>
                  Previous
                </Button>
              )}
              {currentStep < stepConfigs.length - 1 ? (
                <Button type="button" onClick={nextStep}>
                  Next
                </Button>
              ) : (
                <Button type="button" onClick={handleSubmit} disabled={loading}>
                  {loading && <UiLoadingSpinner className="size-4" />}
                  Create Guardrail
                </Button>
              )}
            </div>
          </div>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddGuardrailForm;
