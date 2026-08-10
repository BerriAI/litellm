import { InfoCircleOutlined } from "@ant-design/icons";
import { Select as AntdSelect, Card, Collapse, Divider, Input, Space, Switch, Tooltip, Typography } from "antd";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import AdaptiveRoutingConfig from "./AdaptiveRoutingConfig";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import EscalationKeywords from "./EscalationKeywords";
import KeywordTierRules, { KeywordTierRule } from "./KeywordTierRules";
import SemanticKeywordMatching from "./SemanticKeywordMatching";
import { useTranslation } from "react-i18next";

const { Text } = Typography;

export const DEFAULT_CLASSIFIER_TIMEOUT_MS = 3000;
export const DEFAULT_TIER_DISTANCE_PENALTY = 0.5;
export const DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE = 3;
export const DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS = 200;
export const DEFAULT_SESSION_AFFINITY = false;

export interface ComplexityTiers {
  SIMPLE: string[];
  MEDIUM: string[];
  COMPLEX: string[];
  REASONING: string[];
}

export interface ClassifierLLMConfig {
  model: string;
  timeout_ms: number;
  system_prompt?: string;
}

export type ClassifierType = "heuristic" | "llm";

export type ClassifierFallback = "heuristic" | "default_model";

export const DEFAULT_CLASSIFIER_FALLBACK: ClassifierFallback = "heuristic";

export interface AdaptiveRouterWeights {
  quality: number;
  cost: number;
}

export const DEFAULT_ADAPTIVE_WEIGHTS: AdaptiveRouterWeights = { quality: 0.3, cost: 0.7 };

export type AdaptiveEligible = "all" | "classified_tier";

export type ComplexityTierLabels = Partial<Record<keyof ComplexityTiers, string>>;

export interface ComplexityRouterConfigValue {
  tiers: ComplexityTiers;
  tier_labels?: ComplexityTierLabels;
  classifier_type: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
  classifier_context_window_size?: number;
  classifier_context_per_turn_chars?: number;
  classifier_context_include_assistant_turns?: boolean;
  classifier_fallback?: ClassifierFallback;
  session_affinity?: boolean;
  adaptive?: boolean;
  adaptive_weights?: AdaptiveRouterWeights;
  tier_distance_penalty?: number;
  adaptive_eligible?: AdaptiveEligible;
  return_raw_model_name?: boolean;
}

interface ComplexityRouterConfigProps {
  modelInfo: ModelGroup[];
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  customTechnicalKeywords?: string[];
  onCustomTechnicalKeywordsChange?: (keywords: string[]) => void;
  // Optional: the edit-auto-router modal doesn't yet support editing keyword tier
  // rules or semantic matching, so it renders this component without them.
  keywordTierRules?: KeywordTierRule[];
  onKeywordTierRulesChange?: (rules: KeywordTierRule[]) => void;
  semanticMatchingEnabled?: boolean;
  onSemanticMatchingEnabledChange?: (enabled: boolean) => void;
  embeddingModel?: string;
  onEmbeddingModelChange?: (model: string) => void;
  matchThreshold?: number;
  onMatchThresholdChange?: (threshold: number) => void;
  escalationKeywords?: string[];
  onEscalationKeywordsChange?: (keywords: string[]) => void;
  showValidationErrors?: boolean;
}

export const TIER_DESCRIPTIONS: Record<
  keyof ComplexityTiers,
  { label: string; description: string; examples: string }
> = {
  SIMPLE: {
    label: "Simple",
    description: "Basic questions, greetings, simple factual queries",
    examples: '"Hello!", "What is Python?", "Thanks!"',
  },
  MEDIUM: {
    label: "Medium",
    description: "Standard queries requiring some reasoning or explanation",
    examples: '"Explain how REST APIs work", "Debug this error"',
  },
  COMPLEX: {
    label: "Complex",
    description: "Technical, multi-part requests requiring deep knowledge",
    examples: '"Design a microservices architecture", "Implement a rate limiter"',
  },
  REASONING: {
    label: "Reasoning",
    description: "Chain-of-thought, analysis, explicit reasoning requests",
    examples: '"Think step by step...", "Analyze the pros and cons..."',
  },
};

export const TIER_KEYS = Object.keys(TIER_DESCRIPTIONS) as Array<keyof ComplexityTiers>;

export const effectiveTierLabel = (tier: keyof ComplexityTiers, tierLabels: ComplexityTierLabels | undefined): string =>
  tierLabels?.[tier]?.trim() || TIER_DESCRIPTIONS[tier].label;

const ComplexityRouterConfig: React.FC<ComplexityRouterConfigProps> = ({
  modelInfo,
  value,
  onChange,
  customTechnicalKeywords,
  onCustomTechnicalKeywordsChange,
  keywordTierRules = [],
  onKeywordTierRulesChange,
  semanticMatchingEnabled = false,
  onSemanticMatchingEnabledChange,
  embeddingModel,
  onEmbeddingModelChange = () => {},
  matchThreshold = 0.5,
  onMatchThresholdChange = () => {},
  escalationKeywords = [],
  onEscalationKeywordsChange,
  showValidationErrors = false,
}) => {
  const { t } = useTranslation("gateway");
  // The deployment's default model is derived from the tiers on submit, mirroring the order
  // add_auto_router_tab uses, so the fallback option is offered exactly when one will exist.
  const hasDefaultModel = Boolean(
    value.tiers.MEDIUM[0] || value.tiers.SIMPLE[0] || value.tiers.COMPLEX[0] || value.tiers.REASONING[0],
  );

  // Embedding models can't serve a chat-completion role, so they're excluded here.
  const modelOptions = modelInfo
    .filter((model) => model.mode !== "embedding")
    .map((model) => ({
      value: model.model_group,
      label: model.model_group,
    }));

  const handleTierChange = (tier: keyof ComplexityTiers, models: string[]) => {
    onChange({
      ...value,
      tiers: { ...value.tiers, [tier]: models },
    });
  };

  const handleTierLabelChange = (tier: keyof ComplexityTiers, label: string) => {
    onChange({
      ...value,
      tier_labels: { ...value.tier_labels, [tier]: label },
    });
  };

  return (
    <div className="w-full max-w-none">
      <Space align="center" style={{ marginBottom: 16 }}>
        <Typography.Title level={4} style={{ margin: 0 }}>
          {t("models.autoRouters.details.title")}
        </Typography.Title>
        <Tooltip title={t("models.autoRouters.details.titleTooltip")}>
          <InfoCircleOutlined className="text-gray-400" />
        </Tooltip>
      </Space>

      <Text type="secondary" style={{ display: "block", marginBottom: 24 }}>
        {t("models.autoRouters.details.description")}
      </Text>

      <Text type="secondary" style={{ display: "block", marginBottom: 16, fontSize: 12 }}>
        {t("models.autoRouters.details.renameHint")}{" "}
        {value.classifier_type === "llm" && t("models.autoRouters.details.llmRenameHint")}
      </Text>

      <Card>
        {TIER_KEYS.map((tier, index) => {
          const tierInfo = TIER_DESCRIPTIONS[tier];
          const tierKey = tier.toLowerCase();
          const defaultLabel = t(`models.autoRouters.details.tiers.${tierKey}.label`, {
            defaultValue: tierInfo.label,
          });
          const label = value.tier_labels?.[tier]?.trim() || defaultLabel;
          const description = t(`models.autoRouters.details.tiers.${tierKey}.description`, {
            defaultValue: tierInfo.description,
          });
          const examples = t(`models.autoRouters.details.tiers.${tierKey}.examples`, {
            defaultValue: tierInfo.examples,
          });
          const tierMissing = showValidationErrors && value.tiers[tier].length === 0;
          return (
            <div key={tier}>
              {index > 0 && <Divider style={{ margin: "16px 0" }} />}
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <Text strong style={{ fontSize: 16 }}>
                    {t("models.autoRouters.details.tierTitle", { label })}
                  </Text>
                  <Tooltip title={description}>
                    <InfoCircleOutlined className="text-gray-400" />
                  </Tooltip>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {t("models.autoRouters.details.tierPosition", {
                      current: index + 1,
                      total: TIER_KEYS.length,
                      key: tier,
                    })}
                  </Text>
                </div>
                <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
                  {t("models.autoRouters.details.examples", { examples })}
                </Text>
                <Input
                  value={value.tier_labels?.[tier] ?? ""}
                  onChange={(event) => handleTierLabelChange(tier, event.target.value)}
                  placeholder={t("models.autoRouters.details.displayName", { label: defaultLabel })}
                  aria-label={t("models.autoRouters.details.displayNameAria", { label: defaultLabel })}
                  style={{ marginBottom: 8 }}
                  allowClear
                />
                <AntdSelect
                  mode="multiple"
                  value={value.tiers[tier]}
                  onChange={(models) => handleTierChange(tier, models)}
                  placeholder={t("models.autoRouters.details.selectModels", { label: label.toLowerCase() })}
                  showSearch
                  style={{ width: "100%" }}
                  options={modelOptions}
                  status={tierMissing ? "error" : undefined}
                />
                {value.tiers[tier].length > 1 && (
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {t("models.autoRouters.details.multipleModels")}
                  </Text>
                )}
                {tierMissing && (
                  <Text type="danger" style={{ fontSize: 12 }}>
                    {t("models.autoRouters.details.tierRequired", { label })}
                  </Text>
                )}
              </div>
            </div>
          );
        })}
      </Card>

      <Divider />

      <Collapse
        ghost
        style={{ background: "#f9fafb", borderRadius: 8, border: "1px solid #e5e7eb" }}
        items={[
          {
            key: "classifier",
            label: (
              <Text strong style={{ color: "#374151" }}>
                {t("models.autoRouters.details.sections.classification")}
              </Text>
            ),
            children: (
              <ClassificationMethodConfig
                value={value}
                onChange={onChange}
                modelOptions={modelOptions}
                customTechnicalKeywords={customTechnicalKeywords}
                onCustomTechnicalKeywordsChange={onCustomTechnicalKeywordsChange}
                showValidationErrors={showValidationErrors}
                hasDefaultModel={hasDefaultModel}
              />
            ),
          },
          {
            key: "adaptive",
            label: (
              <Text strong style={{ color: "#374151" }}>
                {t("models.autoRouters.details.sections.adaptive")}
              </Text>
            ),
            children: <AdaptiveRoutingConfig value={value} onChange={onChange} />,
          },
          {
            key: "session-affinity",
            label: (
              <Text strong style={{ color: "#374151" }}>
                {t("models.autoRouters.details.sections.sessionAffinity")}
              </Text>
            ),
            children: (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <Switch
                    checked={value.session_affinity ?? DEFAULT_SESSION_AFFINITY}
                    onChange={(sessionAffinity) => onChange({ ...value, session_affinity: sessionAffinity })}
                    aria-label={t("models.autoRouters.details.sessionAffinity")}
                  />
                  <Text strong>{t("models.autoRouters.details.sessionAffinity")}</Text>
                </div>
                <Text type="secondary" style={{ display: "block", fontSize: 12 }}>
                  {t("models.autoRouters.details.sessionAffinityDescription")}
                </Text>
              </>
            ),
          },
          {
            key: "response",
            label: (
              <Text strong style={{ color: "#374151" }}>
                {t("models.autoRouters.details.sections.responseFormat")}
              </Text>
            ),
            children: (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <Switch
                    checked={value.return_raw_model_name ?? false}
                    onChange={(returnRawModelName) => onChange({ ...value, return_raw_model_name: returnRawModelName })}
                    aria-label={t("models.autoRouters.details.rawModelName")}
                  />
                  <Text strong>{t("models.autoRouters.details.rawModelName")}</Text>
                </div>
                <Text type="secondary" style={{ display: "block", fontSize: 12 }}>
                  {t("models.autoRouters.details.rawModelNameDescription")}
                </Text>
              </>
            ),
          },
          ...(onEscalationKeywordsChange
            ? [
                {
                  key: "escalation",
                  label: (
                    <Text strong style={{ color: "#374151" }}>
                      {t("models.autoRouters.details.sections.escalation")}
                    </Text>
                  ),
                  children: <EscalationKeywords keywords={escalationKeywords} onChange={onEscalationKeywordsChange} />,
                },
              ]
            : []),
          ...(onKeywordTierRulesChange || onSemanticMatchingEnabledChange
            ? [
                {
                  key: "keyword-semantic",
                  label: (
                    <Text strong style={{ color: "#374151" }}>
                      {t("models.autoRouters.details.sections.keywordSemantic")}
                    </Text>
                  ),
                  children: (
                    <>
                      {onKeywordTierRulesChange && (
                        <KeywordTierRules
                          rules={keywordTierRules}
                          onChange={onKeywordTierRulesChange}
                          tierLabels={value.tier_labels}
                        />
                      )}
                      {onKeywordTierRulesChange && onSemanticMatchingEnabledChange && (
                        <Divider style={{ margin: "16px 0" }} />
                      )}
                      {onSemanticMatchingEnabledChange && (
                        <SemanticKeywordMatching
                          enabled={semanticMatchingEnabled}
                          onEnabledChange={onSemanticMatchingEnabledChange}
                          embeddingModel={embeddingModel}
                          onEmbeddingModelChange={onEmbeddingModelChange}
                          matchThreshold={matchThreshold}
                          onMatchThresholdChange={onMatchThresholdChange}
                          modelInfo={modelInfo}
                          showValidationErrors={showValidationErrors}
                        />
                      )}
                    </>
                  ),
                },
              ]
            : []),
        ]}
      />
    </div>
  );
};

export default ComplexityRouterConfig;
