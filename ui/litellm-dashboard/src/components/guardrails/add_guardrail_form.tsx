import { Form, Input, Modal, Select, Tag, Typography, Button } from "antd";
import type { TFunction } from "i18next";
import React, { useEffect, useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import NotificationsManager from "../molecules/notifications_manager";
import {
  createGuardrailCall,
  getGuardrailProviderSpecificParams,
  getGuardrailUISettings,
  modelAvailableCall,
} from "../networking";
import ContentFilterConfiguration from "./content_filter/ContentFilterConfiguration";
import {
  choiceToSkipSystemForCreate,
  choiceToSkipToolForCreate,
  getGuardrailProviders,
  guardrail_provider_map,
  guardrailLogoMap,
  populateGuardrailProviderMap,
  populateGuardrailProviders,
  shouldRenderContentFilterConfigSettings,
  shouldRenderLLMJudgeFields,
  shouldRenderPIIConfigSettings,
} from "./guardrail_info_helpers";
import { resolveLogoSrc } from "@/lib/assetPaths";
import GuardrailOptionalParams from "./guardrail_optional_params";
import GuardrailProviderFields from "./guardrail_provider_fields";
import LLMJudgeFields from "./llm_judge/LLMJudgeFields";
import PiiConfiguration from "./pii_configuration";
import ToolPermissionRulesEditor, { ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";

const { Title, Text, Link } = Typography;
const { Option } = Select;

const getModeDescriptions = (t: TFunction) => ({
  pre_call: t("guardrails.addGuardrailForm.modeDesc.pre_call"),
  during_call: t("guardrails.addGuardrailForm.modeDesc.during_call"),
  post_call: t("guardrails.addGuardrailForm.modeDesc.post_call"),
  logging_only: t("guardrails.addGuardrailForm.modeDesc.logging_only"),
  pre_mcp_call: t("guardrails.addGuardrailForm.modeDesc.pre_mcp_call"),
  during_mcp_call: t("guardrails.addGuardrailForm.modeDesc.during_mcp_call"),
});

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

interface LiteLLMParams {
  guardrail: string;
  mode: string;
  default_on: boolean;
  [key: string]: any; // Allow additional properties for specific guardrails
}

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
  const { t } = useTranslation();
  const modeDescriptions = useMemo(() => getModeDescriptions(t), [t]);
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [guardrailSettings, setGuardrailSettings] = useState<GuardrailSettings | null>(null);
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [selectedActions, setSelectedActions] = useState<{ [key: string]: string }>({});
  const [currentStep, setCurrentStep] = useState(0);
  const [providerParams, setProviderParams] = useState<ProviderParamsResponse | null>(null);

  // Azure Text Moderation state
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [globalSeverityThreshold, setGlobalSeverityThreshold] = useState<number>(2);
  const [categorySpecificThresholds, setCategorySpecificThresholds] = useState<{ [key: string]: number }>({});

  // Content Filter state
  const [selectedPatterns, setSelectedPatterns] = useState<any[]>([]);
  const [blockedWords, setBlockedWords] = useState<any[]>([]);
  const [selectedContentCategories, setSelectedContentCategories] = useState<any[]>([]);
  const [pendingCategorySelection, setPendingCategorySelection] = useState<string>("");
  const [competitorIntentEnabled, setCompetitorIntentEnabled] = useState(false);
  const [competitorIntentConfig, setCompetitorIntentConfig] = useState<any>(null);

  // Endpoint Settings state (step 5)
  const [selectedEndpointType, setSelectedEndpointType] = useState<string>("");
  const [endSessionAfterNFails, setEndSessionAfterNFails] = useState<number | undefined>(undefined);
  const [onViolation, setOnViolation] = useState<"warn" | "end_session">("warn");
  const [realtimeViolationMessage, setRealtimeViolationMessage] = useState<string>("");
  const [endpointSettingsOpen, setEndpointSettingsOpen] = useState<boolean>(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  const [toolPermissionConfig, setToolPermissionConfig] = useState<ToolPermissionConfig>({
    rules: [],
    default_action: "deny",
    on_disallowed_action: "block",
    violation_message_template: "",
  });

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
          setAvailableModels(modelsResp.data.map((m: any) => m.id));
        }

        // Populate dynamic providers from API response
        populateGuardrailProviders(providerParamsResp);
        populateGuardrailProviderMap(providerParamsResp);
      } catch (error) {
        console.error("Error fetching guardrail data:", error);
        NotificationsManager.fromBackend(t("guardrails.addGuardrailForm.failedToLoadConfig"));
      }
    };

    fetchData();
  }, [accessToken]);

  // Apply preset when settings are loaded and form becomes visible
  useEffect(() => {
    if (!preset || !visible || !guardrailSettings) return;

    // Set provider
    setSelectedProvider(preset.provider);
    const baseValues: Record<string, any> = {
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
    form.setFieldsValue(baseValues);

    // Pre-select content category if specified
    if (preset.categoryName && guardrailSettings.content_filter_settings?.content_categories) {
      const category = guardrailSettings.content_filter_settings.content_categories.find(
        (c: any) => c.name === preset.categoryName,
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
  }, [preset, visible, guardrailSettings]);

  const handleProviderChange = (value: string) => {
    setSelectedProvider(value);
    // Reset form fields that are provider-specific
    const resetValues: Record<string, any> = {
      config: undefined,
      presidio_analyzer_api_base: undefined,
      presidio_anonymizer_api_base: undefined,
    };
    if (value === "BlockCodeExecution") {
      resetValues.confidence_threshold = 0.5;
    }
    form.setFieldsValue(resetValues);

    // Reset PII selections when changing provider
    setSelectedEntities([]);
    setSelectedActions({});

    // Reset Azure Text Moderation selections when changing provider
    setSelectedCategories([]);
    setGlobalSeverityThreshold(2);
    setCategorySpecificThresholds({});

    // Reset Content Filter selections
    setSelectedPatterns([]);
    setBlockedWords([]);
    setSelectedContentCategories([]);
    setPendingCategorySelection("");
    setCompetitorIntentEnabled(false);
    setCompetitorIntentConfig(null);

    setToolPermissionConfig({
      rules: [],
      default_action: "deny",
      on_disallowed_action: "block",
      violation_message_template: "",
    });

    // Default LLM-as-a-Judge to post_call mode
    if (value === "LlmAsAJudge") {
      form.setFieldsValue({ mode: "post_call" });
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

  // Azure Text Moderation handlers
  const handleCategorySelect = (category: string) => {
    setSelectedCategories((prev) =>
      prev.includes(category) ? prev.filter((c) => c !== category) : [...prev, category],
    );
  };

  const handleGlobalSeverityChange = (threshold: number) => {
    setGlobalSeverityThreshold(threshold);
  };

  const handleCategorySeverityChange = (category: string, threshold: number) => {
    setCategorySpecificThresholds((prev) => ({
      ...prev,
      [category]: threshold,
    }));
  };

  const nextStep = async () => {
    try {
      // Validate current step fields
      if (currentStep === 0) {
        await form.validateFields(["guardrail_name", "provider", "mode", "default_on"]);
        // Also validate provider-specific fields if applicable
        if (selectedProvider) {
          // This will automatically validate any required fields for the selected provider
          const fieldsToValidate = ["guardrail_name", "provider", "mode", "default_on"];

          if (selectedProvider === "PresidioPII") {
            fieldsToValidate.push("presidio_analyzer_api_base", "presidio_anonymizer_api_base");
          }
          await form.validateFields(fieldsToValidate);
        }
      }

      // Validate configuration steps
      if (currentStep === 1) {
        if (shouldRenderPIIConfigSettings(selectedProvider) && selectedEntities.length === 0) {
          NotificationsManager.fromBackend(t("guardrails.addGuardrailForm.selectAtLeastOnePii"));
          return;
        }
      }

      setCurrentStep(currentStep + 1);
    } catch (error) {
      console.error("Form validation failed:", error);
    }
  };

  const prevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  const handleAddAndContinue = (competitorIntentOnly?: boolean) => {
    // Competitor intent only: just advance to next step (no category to add)
    if (competitorIntentOnly) {
      setCurrentStep(currentStep + 1);
      return;
    }

    if (!pendingCategorySelection || !guardrailSettings) return;

    const contentFilterSettings = guardrailSettings.content_filter_settings;
    if (!contentFilterSettings) return;

    const category = contentFilterSettings.content_categories?.find((c) => c.name === pendingCategorySelection);
    if (!category) return;

    // Check if already added
    if (selectedContentCategories.some((c) => c.category === pendingCategorySelection)) {
      setPendingCategorySelection("");
      setCurrentStep(currentStep + 1);
      return;
    }

    // Add the category
    setSelectedContentCategories([
      ...selectedContentCategories,
      {
        id: `category-${Date.now()}`,
        category: category.name,
        display_name: category.display_name,
        action: category.default_action as "BLOCK" | "MASK",
        severity_threshold: "medium",
      },
    ]);

    // Clear pending selection and advance to next step
    setPendingCategorySelection("");
    setCurrentStep(currentStep + 1);
  };

  const resetForm = () => {
    form.resetFields();
    setSelectedProvider(null);
    setSelectedEntities([]);
    setSelectedActions({});
    setSelectedCategories([]);
    setGlobalSeverityThreshold(2);
    setCategorySpecificThresholds({});
    setSelectedPatterns([]);
    setBlockedWords([]);
    setSelectedContentCategories([]);
    setPendingCategorySelection("");
    setToolPermissionConfig({
      rules: [],
      default_action: "deny",
      on_disallowed_action: "block",
      violation_message_template: "",
    });
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
      await form.validateFields();

      // After validation, fetch *all* form values (including those from previous steps)
      const values = form.getFieldsValue(true);

      // Get the guardrail provider value from the map
      const guardrailProvider = guardrail_provider_map[values.provider];

      // Prepare the guardrail data with proper typings
      const guardrailData: {
        guardrail_name: string;
        litellm_params: {
          guardrail: string;
          mode: string;
          default_on: boolean;
          [key: string]: any; // Allow dynamic properties
        };
        guardrail_info: any;
      } = {
        guardrail_name: values.guardrail_name,
        litellm_params: {
          guardrail: guardrailProvider,
          mode: values.mode,
          default_on: values.default_on,
        },
        guardrail_info: {},
      };

      const skipForCreate = choiceToSkipSystemForCreate(values.skip_system_message_choice);
      if (skipForCreate !== undefined) {
        guardrailData.litellm_params.skip_system_message_in_guardrail = skipForCreate;
      }

      const skipToolForCreate = choiceToSkipToolForCreate(values.skip_tool_message_choice);
      if (skipToolForCreate !== undefined) {
        guardrailData.litellm_params.skip_tool_message_in_guardrail = skipToolForCreate;
      }

      // For Presidio PII, add the entity and action configurations
      if (values.provider === "PresidioPII" && selectedEntities.length > 0) {
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
      if (shouldRenderContentFilterConfigSettings(values.provider)) {
        // Validate that at least one content filter setting is configured
        const hasCompetitorIntent = competitorIntentEnabled && competitorIntentConfig?.brand_self?.length > 0;
        if (
          selectedPatterns.length === 0 &&
          blockedWords.length === 0 &&
          selectedContentCategories.length === 0 &&
          !hasCompetitorIntent
        ) {
          NotificationsManager.fromBackend(t("guardrails.addGuardrailForm.configureAtLeastOneFilter"));
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
        if (competitorIntentEnabled && competitorIntentConfig?.brand_self?.length > 0) {
          guardrailData.litellm_params.competitor_intent_config = {
            competitor_intent_type: competitorIntentConfig.competitor_intent_type ?? "airline",
            brand_self: competitorIntentConfig.brand_self,
            locations: competitorIntentConfig.locations?.length > 0 ? competitorIntentConfig.locations : undefined,
            competitors:
              competitorIntentConfig.competitor_intent_type === "generic" &&
              competitorIntentConfig.competitors?.length > 0
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
          const configObj = JSON.parse(values.config);
          // For some guardrails, the config values need to be in litellm_params
          guardrailData.guardrail_info = configObj;
        } catch (error) {
          NotificationsManager.fromBackend(t("guardrails.addGuardrailForm.invalidJsonConfig"));
          setLoading(false);
          return;
        }
      }

      if (guardrailProvider === "llm_as_a_judge") {
        const criteria: any[] = values.criteria || [];
        if (criteria.length === 0) {
          NotificationsManager.fromBackend(t("guardrails.addGuardrailForm.addAtLeastOneCriterion"));
          setLoading(false);
          return;
        }
        const weightTotal = criteria.reduce((sum: number, c: any) => sum + (Number(c?.weight) || 0), 0);
        if (weightTotal !== 100) {
          NotificationsManager.fromBackend(t("guardrails.addGuardrailForm.weightsMustSum100", { weightTotal }));
          setLoading(false);
          return;
        }
        guardrailData.litellm_params.judge_model = values.judge_model;
        guardrailData.litellm_params.overall_threshold = values.overall_threshold ?? 80;
        guardrailData.litellm_params.on_failure = values.on_failure ?? "block";
        guardrailData.litellm_params.criteria = criteria.map((c: any) => ({
          name: c.name,
          weight: Number(c.weight),
          description: c.description || "",
        }));
      }

      if (guardrailProvider === "tool_permission") {
        if (toolPermissionConfig.rules.length === 0) {
          NotificationsManager.fromBackend(t("guardrails.addGuardrailForm.addAtLeastOneRule"));
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
      if (shouldRenderContentFilterConfigSettings(values.provider)) {
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

      console.log("values: ", JSON.stringify(values));

      // Use pre-fetched provider params to copy recognised params
      // Skip for providers that handle their own litellm_params (llm_as_a_judge, tool_permission, content filter, PII)
      if (providerParams && selectedProvider && guardrailProvider !== "llm_as_a_judge") {
        const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
        console.log("providerKey: ", providerKey);
        const providerSpecificParams = providerParams[providerKey] || {};

        const allowedParams = new Set<string>();

        console.log("providerSpecificParams: ", JSON.stringify(providerSpecificParams));

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

        console.log("allowedParams: ", allowedParams);
        allowedParams.forEach((paramName) => {
          // Check for both direct parameter name and nested optional_params object
          let paramValue = values[paramName];
          if (paramValue === undefined || paramValue === null || paramValue === "") {
            paramValue = values.optional_params?.[paramName];
          }

          if (paramValue !== undefined && paramValue !== null && paramValue !== "") {
            guardrailData.litellm_params[paramName] = paramValue;
          }
        });
      }

      if (!accessToken) {
        throw new Error("No access token available");
      }

      console.log("Sending guardrail data:", JSON.stringify(guardrailData));
      await createGuardrailCall(accessToken, guardrailData);

      NotificationsManager.success(t("guardrails.addGuardrailForm.createSuccess"));

      // Reset form and close modal
      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create guardrail:", error);
      NotificationsManager.fromBackend(
        t("guardrails.addGuardrailForm.createFailed", {
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setLoading(false);
    }
  };

  const renderBasicInfo = () => {
    return (
      <>
        <Form.Item
          name="guardrail_name"
          label={t("guardrails.addGuardrailForm.guardrailNameLabel")}
          rules={[{ required: true, message: t("guardrails.addGuardrailForm.guardrailNameRequired") }]}
        >
          <Input placeholder={t("guardrails.addGuardrailForm.guardrailNamePlaceholder")} />
        </Form.Item>

        <Form.Item
          name="provider"
          label={t("guardrails.addGuardrailForm.providerLabel")}
          rules={[{ required: true, message: t("guardrails.addGuardrailForm.providerRequired") }]}
        >
          <Select
            placeholder={t("guardrails.addGuardrailForm.providerPlaceholder")}
            onChange={handleProviderChange}
            labelInValue={false}
            optionLabelProp="label"
            dropdownRender={(menu) => menu}
            showSearch={true}
          >
            {Object.entries(getGuardrailProviders()).map(([key, value]) => (
              <Option
                key={key}
                value={key}
                label={
                  <div style={{ display: "flex", alignItems: "center" }}>
                    {guardrailLogoMap[value] && (
                      <img
                        src={resolveLogoSrc(guardrailLogoMap[value])}
                        alt=""
                        style={{
                          height: "20px",
                          width: "20px",
                          marginRight: "8px",
                          objectFit: "contain",
                        }}
                        onError={(e) => {
                          // Hide broken image icon if image fails to load
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    )}
                    <span>{value}</span>
                  </div>
                }
              >
                <div style={{ display: "flex", alignItems: "center" }}>
                  {guardrailLogoMap[value] && (
                    <img
                      src={resolveLogoSrc(guardrailLogoMap[value])}
                      alt=""
                      style={{
                        height: "20px",
                        width: "20px",
                        marginRight: "8px",
                        objectFit: "contain",
                      }}
                      onError={(e) => {
                        // Hide broken image icon if image fails to load
                        e.currentTarget.style.display = "none";
                      }}
                    />
                  )}
                  <span>{value}</span>
                </div>
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          name="mode"
          label={t("guardrails.addGuardrailForm.modeLabel")}
          tooltip={t("guardrails.addGuardrailForm.modeTooltip")}
          rules={[{ required: true, message: t("guardrails.addGuardrailForm.modeRequired") }]}
        >
          <Select optionLabelProp="label" mode="multiple">
            {guardrailSettings?.supported_modes?.map((mode) => (
              <Option key={mode} value={mode} label={mode}>
                <div>
                  <div>
                    <strong>{mode}</strong>
                    {mode === "pre_call" && (
                      <Tag color="green" style={{ marginLeft: "8px" }}>
                        {t("guardrails.addGuardrailForm.recommended")}
                      </Tag>
                    )}
                  </div>
                  <div style={{ fontSize: "12px", color: "#888" }}>
                    {modeDescriptions[mode as keyof typeof modeDescriptions]}
                  </div>
                </div>
              </Option>
            )) || (
              <>
                <Option value="pre_call" label="pre_call">
                  <div>
                    <div>
                      <strong>pre_call</strong> <Tag color="green">{t("guardrails.addGuardrailForm.recommended")}</Tag>
                    </div>
                    <div style={{ fontSize: "12px", color: "#888" }}>{modeDescriptions.pre_call}</div>
                  </div>
                </Option>
                <Option value="during_call" label="during_call">
                  <div>
                    <div>
                      <strong>during_call</strong>
                    </div>
                    <div style={{ fontSize: "12px", color: "#888" }}>{modeDescriptions.during_call}</div>
                  </div>
                </Option>
                <Option value="post_call" label="post_call">
                  <div>
                    <div>
                      <strong>post_call</strong>
                    </div>
                    <div style={{ fontSize: "12px", color: "#888" }}>{modeDescriptions.post_call}</div>
                  </div>
                </Option>
                <Option value="logging_only" label="logging_only">
                  <div>
                    <div>
                      <strong>logging_only</strong>
                    </div>
                    <div style={{ fontSize: "12px", color: "#888" }}>{modeDescriptions.logging_only}</div>
                  </div>
                </Option>
              </>
            )}
          </Select>
        </Form.Item>

        <Form.Item
          name="default_on"
          label={t("guardrails.addGuardrailForm.alwaysOnLabel")}
          tooltip={t("guardrails.addGuardrailForm.alwaysOnTooltip")}
        >
          <Select>
            <Select.Option value={true}>{t("common.yes")}</Select.Option>
            <Select.Option value={false}>{t("common.no")}</Select.Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="skip_system_message_choice"
          label={t("guardrails.addGuardrailForm.skipSystemMsgLabel")}
          tooltip={t("guardrails.addGuardrailForm.skipSystemMsgTooltip")}
        >
          <Select>
            <Select.Option value="inherit">{t("guardrails.addGuardrailForm.useGlobalDefault")}</Select.Option>
            <Select.Option value="yes">{t("guardrails.addGuardrailForm.yesExcludeFromScan")}</Select.Option>
            <Select.Option value="no">{t("guardrails.addGuardrailForm.noAlwaysInclude")}</Select.Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="skip_tool_message_choice"
          label={t("guardrails.addGuardrailForm.skipToolMsgLabel")}
          tooltip={t("guardrails.addGuardrailForm.skipToolMsgTooltip")}
        >
          <Select>
            <Select.Option value="inherit">{t("guardrails.addGuardrailForm.useGlobalDefault")}</Select.Option>
            <Select.Option value="yes">{t("guardrails.addGuardrailForm.yesExcludeFromScan")}</Select.Option>
            <Select.Option value="no">{t("guardrails.addGuardrailForm.noAlwaysInclude")}</Select.Option>
          </Select>
        </Form.Item>

        {/* Use the GuardrailProviderFields component to render provider-specific fields */}
        {!isToolPermissionProvider &&
          !shouldRenderContentFilterConfigSettings(selectedProvider) &&
          !shouldRenderLLMJudgeFields(selectedProvider) && (
            <GuardrailProviderFields
              selectedProvider={selectedProvider}
              accessToken={accessToken}
              providerParams={providerParams}
            />
          )}
      </>
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

    console.log("guardrail_provider_map: ", guardrail_provider_map);
    console.log("selectedProvider: ", selectedProvider);
    const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
    const providerFields = providerParams && providerParams[providerKey];

    if (!providerFields || !providerFields.optional_params) return null;

    return <GuardrailOptionalParams optionalParams={providerFields.optional_params} parentFieldKey="optional_params" />;
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
          return <LLMJudgeFields availableModels={availableModels} form={form} />;
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

  const renderStepButtons = () => {
    const totalSteps = shouldRenderContentFilterConfigSettings(selectedProvider) ? 5 : 2;
    const isLastStep = currentStep === totalSteps - 1;
    const isCategoriesStep = shouldRenderContentFilterConfigSettings(selectedProvider) && currentStep === 1;
    const hasPendingCategory = pendingCategorySelection !== "";
    const hasCompetitorIntentConfigured =
      competitorIntentEnabled && (competitorIntentConfig?.brand_self?.length ?? 0) > 0;
    const canContinueFromCategoriesStep = hasPendingCategory || hasCompetitorIntentConfigured;

    return (
      <div className="flex justify-end space-x-2 mt-4">
        {currentStep > 0 && <Button onClick={prevStep}>{t("common.previous")}</Button>}
        {isCategoriesStep ? (
          <>
            <Button onClick={nextStep}>{t("guardrails.addGuardrailForm.skip")}</Button>
            <Button
              type="primary"
              onClick={() => handleAddAndContinue(hasCompetitorIntentConfigured)}
              disabled={!canContinueFromCategoriesStep}
            >
              {hasPendingCategory
                ? t("guardrails.addGuardrailForm.addAndContinue")
                : t("guardrails.addGuardrailForm.continueArrow")}
            </Button>
          </>
        ) : (
          <>
            {!isLastStep && (
              <Button type="primary" onClick={nextStep}>
                {t("common.next")}
              </Button>
            )}
            {isLastStep && (
              <Button type="primary" onClick={handleSubmit} loading={loading}>
                {t("guardrails.addGuardrailForm.createGuardrailButton")}
              </Button>
            )}
          </>
        )}
        <Button onClick={handleClose}>{t("common.cancel")}</Button>
      </div>
    );
  };

  const renderEndpointSettings = () => {
    return (
      <div className="space-y-6">
        <div>
          <p className="text-sm text-gray-500">
            <Trans i18nKey="guardrails.addGuardrailForm.endpointSettingsDesc" components={{ code: <code /> }} />
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {t("guardrails.addGuardrailForm.callTypeLabel")}
          </label>
          <Select
            placeholder={t("guardrails.addGuardrailForm.callTypePlaceholder")}
            value={selectedEndpointType || undefined}
            onChange={(v) => {
              setSelectedEndpointType(v);
              setEndpointSettingsOpen(false);
            }}
            style={{ width: 260 }}
            allowClear
            options={[{ value: "realtime", label: "/v1/realtime" }]}
          />
          <p className="text-xs text-gray-400 mt-1">{t("guardrails.addGuardrailForm.moreCallTypesSoon")}</p>
        </div>

        {selectedEndpointType === "realtime" && (
          <div className="border border-gray-200 rounded-lg overflow-hidden">
            <button
              type="button"
              onClick={() => setEndpointSettingsOpen((o) => !o)}
              className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 text-sm font-medium text-gray-700"
            >
              <span>{t("guardrails.addGuardrailForm.realtimeSettingsTitle")}</span>
              <svg
                className={`w-4 h-4 text-gray-500 transition-transform ${endpointSettingsOpen ? "rotate-180" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {endpointSettingsOpen && (
              <div className="space-y-5 px-4 py-4 border-t border-gray-200">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t("guardrails.addGuardrailForm.endSessionLabel")}
                  </label>
                  <p className="text-xs text-gray-400 mb-2">{t("guardrails.addGuardrailForm.endSessionDesc")}</p>
                  <input
                    type="number"
                    min={1}
                    placeholder="e.g. 3"
                    value={endSessionAfterNFails ?? ""}
                    onChange={(e) =>
                      setEndSessionAfterNFails(e.target.value ? parseInt(e.target.value, 10) : undefined)
                    }
                    className="border border-gray-300 rounded px-3 py-1.5 text-sm w-32"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {t("guardrails.addGuardrailForm.onViolationLabel")}
                  </label>
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
                          <span className="text-sm font-medium text-gray-800">
                            {opt === "warn"
                              ? t("guardrails.addGuardrailForm.violationWarn")
                              : t("guardrails.addGuardrailForm.violationEndSession")}
                          </span>
                          <p className="text-xs text-gray-400 m-0">
                            {opt === "warn"
                              ? t("guardrails.addGuardrailForm.violationWarnDesc")
                              : t("guardrails.addGuardrailForm.violationEndSessionDesc")}
                          </p>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t("guardrails.addGuardrailForm.violationMessageLabel")}
                  </label>
                  <p className="text-xs text-gray-400 mb-2">{t("guardrails.addGuardrailForm.violationMessageDesc")}</p>
                  <textarea
                    rows={3}
                    placeholder={t("guardrails.addGuardrailForm.violationMessagePlaceholder")}
                    value={realtimeViolationMessage}
                    onChange={(e) => setRealtimeViolationMessage(e.target.value)}
                    className="border border-gray-300 rounded px-3 py-2 text-sm w-full resize-none"
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
        { title: t("guardrails.addGuardrailForm.stepBasicInfo"), optional: false },
        { title: t("guardrails.addGuardrailForm.stepTopics"), optional: false },
        { title: t("guardrails.addGuardrailForm.stepPatterns"), optional: false },
        { title: t("guardrails.addGuardrailForm.stepKeywords"), optional: false },
        { title: t("guardrails.addGuardrailForm.stepEndpointSettings"), optional: true },
      ];
    }
    if (shouldRenderPIIConfigSettings(selectedProvider)) {
      return [
        { title: t("guardrails.addGuardrailForm.stepBasicInfo"), optional: false },
        { title: t("guardrails.addGuardrailForm.stepPiiConfig"), optional: false },
      ];
    }
    return [
      { title: t("guardrails.addGuardrailForm.stepBasicInfo"), optional: false },
      { title: t("guardrails.addGuardrailForm.stepProviderConfig"), optional: false },
    ];
  };

  const stepConfigs = getStepConfigs();

  return (
    <Modal
      title={null}
      open={visible}
      onCancel={handleClose}
      maskClosable={false}
      footer={null}
      width={1000}
      closable={false}
      className="top-8"
      styles={{
        body: { padding: 0 },
      }}
    >
      <div className="flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <h3 className="text-base font-semibold text-gray-900 m-0">{t("guardrails.addGuardrailForm.modalTitle")}</h3>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600 bg-transparent border-none cursor-pointer text-base leading-none p-1"
          >
            &#x2715;
          </button>
        </div>

        {/* Scrollable content - inline vertical stepper */}
        <div className="overflow-auto px-6 py-4" style={{ maxHeight: "calc(80vh - 120px)" }}>
          <Form
            form={form}
            layout="vertical"
            initialValues={{
              mode: "pre_call",
              default_on: false,
              skip_system_message_choice: "inherit",
              skip_tool_message_choice: "inherit",
            }}
          >
            {stepConfigs.map((step, index) => {
              const isDone = index < currentStep;
              const isCurrent = index === currentStep;
              const isLast = index === stepConfigs.length - 1;
              return (
                <div key={index} className="relative flex gap-4" style={{ paddingBottom: isLast ? 0 : 8 }}>
                  {/* Vertical line + step indicator */}
                  <div className="flex flex-col items-center flex-shrink-0" style={{ width: 24 }}>
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0"
                      style={{
                        background: isDone ? "#4f46e5" : isCurrent ? "#fff" : "#f8fafc",
                        color: isDone ? "#fff" : isCurrent ? "#4f46e5" : "#94a3b8",
                        border: isCurrent ? "2px solid #4f46e5" : isDone ? "none" : "1px solid #e2e8f0",
                      }}
                    >
                      {isDone ? "\u2713" : index + 1}
                    </div>
                    {!isLast && (
                      <div
                        className="flex-1"
                        style={{
                          width: 1,
                          background: isDone ? "#4f46e5" : "#e2e8f0",
                          minHeight: 16,
                        }}
                      />
                    )}
                  </div>

                  {/* Step content */}
                  <div className="flex-1 min-w-0" style={{ paddingBottom: isLast ? 0 : 16 }}>
                    {/* Step header - clickable for completed steps */}
                    <div
                      className={`flex items-center gap-2 ${isDone ? "cursor-pointer" : ""}`}
                      onClick={() => {
                        if (isDone) setCurrentStep(index);
                      }}
                      style={{ minHeight: 24 }}
                    >
                      <span
                        className="text-sm"
                        style={{
                          fontWeight: isCurrent ? 600 : 500,
                          color: isCurrent ? "#1e293b" : isDone ? "#4f46e5" : "#94a3b8",
                        }}
                      >
                        {step.title}
                      </span>
                      {step.optional && !isCurrent && (
                        <span className="text-[11px] text-slate-400">{t("common.optional")}</span>
                      )}
                      {isDone && (
                        <span className="text-[11px] text-indigo-500 hover:underline">{t("common.edit")}</span>
                      )}
                    </div>

                    {/* Expanded form content for current step */}
                    {isCurrent && <div className="mt-3">{renderStepContent()}</div>}
                  </div>
                </div>
              );
            })}
          </Form>
        </div>

        {/* Bottom bar */}
        <div className="flex items-center justify-end space-x-3 px-6 py-3 border-t border-gray-200">
          <Button onClick={handleClose}>{t("common.cancel")}</Button>
          {currentStep > 0 && <Button onClick={prevStep}>{t("common.previous")}</Button>}
          {currentStep < stepConfigs.length - 1 ? (
            <Button type="primary" onClick={nextStep}>
              {t("common.next")}
            </Button>
          ) : (
            <Button type="primary" onClick={handleSubmit} loading={loading}>
              {t("guardrails.addGuardrailForm.createGuardrailButton")}
            </Button>
          )}
        </div>
      </div>
    </Modal>
  );
};

export default AddGuardrailForm;
