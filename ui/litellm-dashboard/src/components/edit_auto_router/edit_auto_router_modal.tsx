import React, { useEffect, useState } from "react";
import { Form, Button, Select as AntdSelect, Tooltip } from "antd";
import { TextInput } from "@tremor/react";
import { modelAvailableCall, modelPatchUpdateCall } from "../networking";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import RouterConfigBuilder from "../add_model/RouterConfigBuilder";
import { normalizeTierModels } from "../add_model/complexity_router_tiers";
import { isComplexityRouter } from "../add_model/auto_router_strategies";
import {
  getKeywordTierRulesError,
  getSemanticConfigError,
  getTierLabelsError,
  hydrateTierLabels,
  normalizeClassifierLlmConfig,
  serializeTierLabels,
} from "../add_model/build_complexity_router_config";
import { KeywordTierRule } from "../add_model/KeywordTierRules";
import { DEFAULT_MATCH_THRESHOLD } from "../add_model/SemanticKeywordMatching";
import { hydrateKeywordTierRules, serializeKeywordTierRules } from "../add_model/complexity_router_keywords";
import ComplexityRouterConfig, {
  ComplexityRouterConfigValue,
  DEFAULT_ADAPTIVE_WEIGHTS,
  DEFAULT_SESSION_AFFINITY,
  DEFAULT_DEPLOYMENT_AFFINITY,
  DEFAULT_TIER_DISTANCE_PENALTY,
} from "../add_model/ComplexityRouterConfig";
import NotificationsManager from "../molecules/notifications_manager";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface EditAutoRouterModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: (updatedModel: any) => void;
  modelData: any;
  accessToken: string;
  userRole: string;
}

// Keys this modal rewrites from its own form state on save. Anything absent from this set is
// carried through untouched from the stored config, so a key only belongs here once the modal
// actually renders a control that can set it.
const MANAGED_COMPLEXITY_ROUTER_KEYS = new Set([
  "tiers",
  "tier_labels",
  "classifier_type",
  "classifier_llm_config",
  "classifier_context_window_size",
  "classifier_context_per_turn_chars",
  "classifier_context_include_assistant_turns",
  "classifier_fallback",
  "session_affinity",
  "deployment_affinity",
  "adaptive",
  "adaptive_weights",
  "tier_distance_penalty",
  "adaptive_eligible",
  "return_raw_model_name",
]);

// Managed only when the caller passes the corresponding state. A caller that does not render
// these controls must carry the stored values through untouched instead of dropping them.
const KEYWORD_MATCHING_KEYS = new Set([
  "keyword_tier_rules",
  "escalation_keywords",
  "semantic_keyword_matching",
  "embedding_model",
  "match_threshold",
]);

const toRecord = (value: unknown): Record<string, unknown> => {
  const parsed: unknown = typeof value === "string" ? JSON.parse(value) : value;
  return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>)
    : {};
};

export interface KeywordMatchingState {
  keywordTierRules: KeywordTierRule[];
  escalationKeywords: string[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  matchThreshold: number;
}

export const buildUpdatedComplexityRouterConfig = (
  storedConfig: unknown,
  value: ComplexityRouterConfigValue,
  customTechnicalKeywords?: string[],
  keywordMatching?: KeywordMatchingState,
): Record<string, unknown> => {
  const isManaged = (key: string): boolean => {
    if (MANAGED_COMPLEXITY_ROUTER_KEYS.has(key)) return true;
    if (keywordMatching !== undefined && KEYWORD_MATCHING_KEYS.has(key)) return true;
    return customTechnicalKeywords !== undefined && key === "custom_technical_keywords";
  };

  const preservedConfig = Object.fromEntries(Object.entries(toRecord(storedConfig)).filter(([key]) => !isManaged(key)));
  const adaptiveEligible = value.adaptive_eligible ?? "all";
  const storedKeywordRules = keywordMatching ? serializeKeywordTierRules(keywordMatching.keywordTierRules) : [];
  const serializedTierLabels = serializeTierLabels(value.tier_labels);

  return {
    ...preservedConfig,
    tiers: value.tiers,
    ...(serializedTierLabels && { tier_labels: serializedTierLabels }),
    classifier_type: value.classifier_type,
    ...(value.classifier_type === "llm" && value.classifier_llm_config
      ? { classifier_llm_config: normalizeClassifierLlmConfig(value.classifier_llm_config) }
      : {}),
    ...(value.classifier_type === "llm" &&
      value.classifier_fallback !== undefined && { classifier_fallback: value.classifier_fallback }),
    ...(value.classifier_type === "llm" &&
      value.classifier_context_window_size !== undefined && {
        classifier_context_window_size: value.classifier_context_window_size,
      }),
    ...(value.classifier_type === "llm" &&
      value.classifier_context_per_turn_chars !== undefined && {
        classifier_context_per_turn_chars: value.classifier_context_per_turn_chars,
      }),
    ...(value.classifier_type === "llm" &&
      value.classifier_context_include_assistant_turns !== undefined && {
        classifier_context_include_assistant_turns: value.classifier_context_include_assistant_turns,
      }),
    session_affinity: value.session_affinity ?? DEFAULT_SESSION_AFFINITY,
    deployment_affinity: value.deployment_affinity ?? DEFAULT_DEPLOYMENT_AFFINITY,
    ...(customTechnicalKeywords &&
      customTechnicalKeywords.length > 0 && {
        custom_technical_keywords: customTechnicalKeywords,
      }),
    ...(value.adaptive && {
      adaptive: true,
      adaptive_weights: value.adaptive_weights ?? DEFAULT_ADAPTIVE_WEIGHTS,
      ...(adaptiveEligible === "all" && {
        tier_distance_penalty: value.tier_distance_penalty ?? DEFAULT_TIER_DISTANCE_PENALTY,
      }),
      adaptive_eligible: adaptiveEligible,
    }),
    ...(value.return_raw_model_name && { return_raw_model_name: true }),
    ...(keywordMatching && {
      // Mirrors buildComplexityRouterConfig: the key only when there is a rule to write,
      // escalation keywords always, semantic trio only when on.
      ...(storedKeywordRules.length > 0 && { keyword_tier_rules: storedKeywordRules }),
      escalation_keywords: keywordMatching.escalationKeywords.map((k) => k.trim()).filter(Boolean),
      ...(keywordMatching.semanticMatchingEnabled && {
        semantic_keyword_matching: true,
        embedding_model: keywordMatching.embeddingModel,
        match_threshold: keywordMatching.matchThreshold,
      }),
    }),
  };
};

const EditAutoRouterModal: React.FC<EditAutoRouterModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
  userRole,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showValidationErrors, setShowValidationErrors] = useState<boolean>(false);
  const [routerConfig, setRouterConfig] = useState<any>(null);
  const [customTechnicalKeywords, setCustomTechnicalKeywords] = useState<string[]>([]);
  const [keywordTierRules, setKeywordTierRules] = useState<KeywordTierRule[]>([]);
  const [escalationKeywords, setEscalationKeywords] = useState<string[]>([]);
  const [semanticMatchingEnabled, setSemanticMatchingEnabled] = useState<boolean>(false);
  const [embeddingModel, setEmbeddingModel] = useState<string | undefined>(undefined);
  const [matchThreshold, setMatchThreshold] = useState<number>(DEFAULT_MATCH_THRESHOLD);
  const [complexityRouterConfig, setComplexityRouterConfig] = useState<ComplexityRouterConfigValue>({
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    classifier_type: "heuristic",
  });
  const isComplexityRouterModel = isComplexityRouter(modelData?.litellm_params);

  // Mirrors the create form: the button says why it is unavailable and disables on the same
  // answer. Tiers use this modal's own rule, which allows a partly filled router, so an edit that
  // is legal today stays legal.
  const submitBlockedReason = !isComplexityRouterModel
    ? null
    : (Object.values(complexityRouterConfig.tiers).every((models) => models.length === 0)
        ? "Please select at least one model for a complexity tier"
        : null) ??
      getTierLabelsError(complexityRouterConfig.tier_labels) ??
      getKeywordTierRulesError(keywordTierRules);

  useEffect(() => {
    if (isVisible && modelData) {
      initializeForm();
    }
  }, [isVisible, modelData]);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      if (!accessToken) return;
      try {
        const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
        setModelAccessGroups(response["data"].map((model: any) => model["id"]));
      } catch (error) {
        console.error("Error fetching model access groups:", error);
      }
    };

    const loadModels = async () => {
      if (!accessToken) return;
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    if (isVisible) {
      fetchModelAccessGroups();
      loadModels();
    }
  }, [isVisible, accessToken]);

  const initializeForm = () => {
    try {
      if (isComplexityRouterModel) {
        // Parse the complexity_router_config if it exists and is a string
        let parsedConfig = modelData.litellm_params?.complexity_router_config || {};
        if (typeof parsedConfig === "string") {
          parsedConfig = JSON.parse(parsedConfig);
        }

        const hydratedComplexityRouterConfig: ComplexityRouterConfigValue = {
          tiers: {
            SIMPLE: normalizeTierModels(parsedConfig.tiers?.SIMPLE),
            MEDIUM: normalizeTierModels(parsedConfig.tiers?.MEDIUM),
            COMPLEX: normalizeTierModels(parsedConfig.tiers?.COMPLEX),
            REASONING: normalizeTierModels(parsedConfig.tiers?.REASONING),
          },
          tier_labels: hydrateTierLabels(parsedConfig.tier_labels),
          classifier_type: parsedConfig.classifier_type || "heuristic",
          classifier_llm_config: parsedConfig.classifier_llm_config,
          classifier_context_window_size:
            typeof parsedConfig.classifier_context_window_size === "number"
              ? parsedConfig.classifier_context_window_size
              : undefined,
          classifier_context_per_turn_chars:
            typeof parsedConfig.classifier_context_per_turn_chars === "number"
              ? parsedConfig.classifier_context_per_turn_chars
              : undefined,
          classifier_context_include_assistant_turns:
            typeof parsedConfig.classifier_context_include_assistant_turns === "boolean"
              ? parsedConfig.classifier_context_include_assistant_turns
              : undefined,
          classifier_fallback:
            parsedConfig.classifier_fallback === "default_model" || parsedConfig.classifier_fallback === "heuristic"
              ? parsedConfig.classifier_fallback
              : undefined,
          session_affinity:
            typeof parsedConfig.session_affinity === "boolean"
              ? parsedConfig.session_affinity
              : DEFAULT_SESSION_AFFINITY,
          deployment_affinity:
            typeof parsedConfig.deployment_affinity === "boolean"
              ? parsedConfig.deployment_affinity
              : DEFAULT_DEPLOYMENT_AFFINITY,
          adaptive: parsedConfig.adaptive || false,
          adaptive_weights: parsedConfig.adaptive_weights,
          tier_distance_penalty: parsedConfig.tier_distance_penalty,
          adaptive_eligible: parsedConfig.adaptive_eligible || "all",
          return_raw_model_name: parsedConfig.return_raw_model_name || false,
        };
        setComplexityRouterConfig(hydratedComplexityRouterConfig);
        setCustomTechnicalKeywords(
          Array.isArray(parsedConfig.custom_technical_keywords) ? parsedConfig.custom_technical_keywords : [],
        );
        // Hydrated from the stored config, never from create-form defaults: these keys are now
        // rewritten on save, so seeding a default here would inject it into a config that never
        // had it.
        setKeywordTierRules(hydrateKeywordTierRules(parsedConfig.keyword_tier_rules));
        setEscalationKeywords(
          Array.isArray(parsedConfig.escalation_keywords)
            ? parsedConfig.escalation_keywords.filter((k: unknown): k is string => typeof k === "string")
            : [],
        );
        setSemanticMatchingEnabled(parsedConfig.semantic_keyword_matching === true);
        setEmbeddingModel(typeof parsedConfig.embedding_model === "string" ? parsedConfig.embedding_model : undefined);
        setMatchThreshold(
          typeof parsedConfig.match_threshold === "number" ? parsedConfig.match_threshold : DEFAULT_MATCH_THRESHOLD,
        );

        form.setFieldsValue({
          auto_router_name: modelData.model_name,
          model_access_group: modelData.model_info?.access_groups || [],
        });
        return;
      }

      // Parse the auto_router_config if it exists and is a string
      let parsedConfig = null;
      if (modelData.litellm_params?.auto_router_config) {
        if (typeof modelData.litellm_params.auto_router_config === "string") {
          parsedConfig = JSON.parse(modelData.litellm_params.auto_router_config);
        } else {
          parsedConfig = modelData.litellm_params.auto_router_config;
        }
      }

      setRouterConfig(parsedConfig);

      // Set form values
      form.setFieldsValue({
        auto_router_name: modelData.model_name,
        auto_router_default_model: modelData.litellm_params?.auto_router_default_model || "",
        auto_router_embedding_model: modelData.litellm_params?.auto_router_embedding_model || "",
        model_access_group: modelData.model_info?.access_groups || [],
      });
    } catch (error) {
      console.error("Error parsing auto router config:", error);
      NotificationsManager.fromBackend("Error loading auto router configuration");
    }
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();

      if (isComplexityRouterModel) {
        const { tiers, classifier_type, classifier_llm_config } = complexityRouterConfig;
        if (Object.values(tiers).every((models) => models.length === 0)) {
          setShowValidationErrors(true);
          NotificationsManager.fromBackend("Please select at least one model for a complexity tier");
          return;
        }
        if (classifier_type === "llm" && !classifier_llm_config?.model) {
          setShowValidationErrors(true);
          NotificationsManager.fromBackend("Please select a classifier model, or switch back to Heuristic");
          return;
        }
        // Same guards the create form applies (add_auto_router_tab.tsx). The backend rejects a
        // keyword rule with no keyword, and semantic_keyword_matching without an embedding model
        // or keyword rules (complexity_router/config.py), so without these a save fails as a raw
        // 400 instead of an inline message.
        const keywordRulesError = getKeywordTierRulesError(keywordTierRules);
        if (keywordRulesError) {
          setShowValidationErrors(true);
          NotificationsManager.fromBackend(keywordRulesError);
          return;
        }

        const semanticError = getSemanticConfigError({ semanticMatchingEnabled, embeddingModel, keywordTierRules });
        if (semanticError) {
          setShowValidationErrors(true);
          NotificationsManager.fromBackend(semanticError);
          return;
        }

        const defaultModel = tiers.MEDIUM[0] || tiers.SIMPLE[0] || tiers.COMPLEX[0] || tiers.REASONING[0];
        const updatedLitellmParams = {
          ...modelData.litellm_params,
          complexity_router_config: buildUpdatedComplexityRouterConfig(
            modelData.litellm_params?.complexity_router_config,
            complexityRouterConfig,
            customTechnicalKeywords,
            {
              keywordTierRules,
              escalationKeywords,
              semanticMatchingEnabled,
              embeddingModel,
              matchThreshold,
            },
          ),
          complexity_router_default_model: defaultModel,
        };
        const updatedModelInfo = {
          ...modelData.model_info,
          access_groups: values.model_access_group || [],
        };

        await modelPatchUpdateCall(
          accessToken,
          { model_name: values.auto_router_name, litellm_params: updatedLitellmParams, model_info: updatedModelInfo },
          modelData.model_info.id,
        );

        NotificationsManager.success("Auto router configuration updated successfully");
        onSuccess({
          ...modelData,
          model_name: values.auto_router_name,
          litellm_params: updatedLitellmParams,
          model_info: updatedModelInfo,
        });
        onCancel();
        return;
      }

      // Prepare the updated litellm_params
      const updatedLitellmParams = {
        ...modelData.litellm_params,
        auto_router_config: JSON.stringify(routerConfig),
        auto_router_default_model: values.auto_router_default_model,
        auto_router_embedding_model: values.auto_router_embedding_model || undefined,
      };

      // Prepare updated model_info
      const updatedModelInfo = {
        ...modelData.model_info,
        access_groups: values.model_access_group || [],
      };

      const updateData = {
        model_name: values.auto_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      await modelPatchUpdateCall(accessToken, updateData, modelData.model_info.id);

      const updatedModelData = {
        ...modelData,
        model_name: values.auto_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      NotificationsManager.success("Auto router configuration updated successfully");
      onSuccess(updatedModelData);
      onCancel();
    } catch (error) {
      console.error("Error updating auto router:", error);
      NotificationsManager.fromBackend("Failed to update auto router configuration");
    } finally {
      setLoading(false);
    }
  };

  const modelOptions = modelInfo.map((model) => ({
    value: model.model_group,
    label: model.model_group,
  }));

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle>Edit Auto Router Configuration</DialogTitle>
          <DialogDescription>
            Edit the auto router configuration including routing logic, default models, and access settings.
          </DialogDescription>
        </DialogHeader>

        <Form form={form} layout="vertical" className="space-y-4">
          {/* Auto Router Name */}
          <Form.Item
            label="Auto Router Name"
            name="auto_router_name"
            rules={[{ required: true, message: "Auto router name is required" }]}
          >
            <TextInput placeholder="e.g., auto_router_1, smart_routing" />
          </Form.Item>

          {isComplexityRouterModel ? (
            /* Complexity Router Configuration */
            <div className="w-full">
              <ComplexityRouterConfig
                showValidationErrors={showValidationErrors}
                modelInfo={modelInfo}
                value={complexityRouterConfig}
                onChange={(config) => {
                  setComplexityRouterConfig(config);
                }}
                customTechnicalKeywords={customTechnicalKeywords}
                onCustomTechnicalKeywordsChange={setCustomTechnicalKeywords}
                keywordTierRules={keywordTierRules}
                onKeywordTierRulesChange={setKeywordTierRules}
                semanticMatchingEnabled={semanticMatchingEnabled}
                onSemanticMatchingEnabledChange={setSemanticMatchingEnabled}
                embeddingModel={embeddingModel}
                onEmbeddingModelChange={setEmbeddingModel}
                matchThreshold={matchThreshold}
                onMatchThresholdChange={setMatchThreshold}
                escalationKeywords={escalationKeywords}
                onEscalationKeywordsChange={setEscalationKeywords}
              />
            </div>
          ) : (
            <>
              {/* Router Configuration Builder */}
              <div className="w-full">
                <RouterConfigBuilder
                  modelInfo={modelInfo}
                  value={routerConfig}
                  onChange={(config) => {
                    setRouterConfig(config);
                  }}
                />
              </div>

              {/* Default Model */}
              <Form.Item
                label="Default Model"
                name="auto_router_default_model"
                rules={[{ required: true, message: "Default model is required" }]}
              >
                <AntdSelect
                  placeholder="Select a default model"
                  options={[...modelOptions, { value: "custom", label: "Enter custom model name" }]}
                  showSearch={true}
                />
              </Form.Item>

              {/* Embedding Model */}
              <Form.Item
                label="Embedding Model"
                name="auto_router_embedding_model"
                rules={[{ required: true, message: "Embedding model is required" }]}
              >
                <AntdSelect
                  placeholder="Select an embedding model"
                  options={[...modelOptions, { value: "custom", label: "Enter custom model name" }]}
                  showSearch={true}
                />
              </Form.Item>
            </>
          )}

          {/* Model Access Groups - Admin only */}
          {userRole === "Admin" && (
            <Form.Item
              label="Model Access Groups"
              name="model_access_group"
              tooltip="Control who can access this auto router"
            >
              <AntdSelect
                mode="tags"
                showSearch
                placeholder="Select existing groups or type to create new ones"
                optionFilterProp="children"
                tokenSeparators={[","]}
                options={modelAccessGroups.map((group) => ({
                  value: group,
                  label: group,
                }))}
                maxTagCount="responsive"
                allowClear
              />
            </Form.Item>
          )}
        </Form>

        <DialogFooter>
          <Button onClick={onCancel}>Cancel</Button>
          <Tooltip title={submitBlockedReason}>
            <Button loading={loading} disabled={submitBlockedReason !== null} onClick={handleSubmit}>
              Save Changes
            </Button>
          </Tooltip>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditAutoRouterModal;
