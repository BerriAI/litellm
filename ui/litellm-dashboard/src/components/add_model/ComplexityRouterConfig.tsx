import { InfoCircleOutlined } from "@ant-design/icons";
import { Select as AntdSelect, Card, Collapse, Divider, Space, Tooltip, Typography } from "antd";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import AdaptiveRoutingConfig from "./AdaptiveRoutingConfig";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import KeywordTierRules, { KeywordTierRule } from "./KeywordTierRules";
import SemanticKeywordMatching from "./SemanticKeywordMatching";

const { Text } = Typography;

export const DEFAULT_CLASSIFIER_TIMEOUT_MS = 3000;
export const DEFAULT_TIER_DISTANCE_PENALTY = 0.5;

// A tier maps to one or more models. With more than one, the backend randomly
// picks among them (or Thompson-samples within the pool when adaptive routing
// is on) instead of always calling the same model.
export interface ComplexityTiers {
  SIMPLE: string[];
  MEDIUM: string[];
  COMPLEX: string[];
  REASONING: string[];
}

export interface ClassifierLLMConfig {
  model: string;
  timeout_ms: number;
}

export type ClassifierType = "heuristic" | "llm";

export interface AdaptiveRouterWeights {
  quality: number;
  cost: number;
}

export const DEFAULT_ADAPTIVE_WEIGHTS: AdaptiveRouterWeights = { quality: 0.3, cost: 0.7 };

export type AdaptiveEligible = "all" | "classified_tier";

export interface ComplexityRouterConfigValue {
  tiers: ComplexityTiers;
  classifier_type: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
  adaptive?: boolean;
  adaptive_weights?: AdaptiveRouterWeights;
  tier_distance_penalty?: number;
  adaptive_eligible?: AdaptiveEligible;
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
  showValidationErrors?: boolean;
}

const TIER_DESCRIPTIONS: Record<keyof ComplexityTiers, { label: string; description: string; examples: string }> = {
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
  showValidationErrors = false,
}) => {
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

  return (
    <div className="w-full max-w-none">
      <Space align="center" style={{ marginBottom: 16 }}>
        <Typography.Title level={4} style={{ margin: 0 }}>
          Complexity Tier Configuration
        </Typography.Title>
        <Tooltip title="Map each complexity tier to one or more models. Simple queries use cheaper/faster models, complex queries use more capable models.">
          <InfoCircleOutlined className="text-gray-400" />
        </Tooltip>
      </Space>

      <Text type="secondary" style={{ display: "block", marginBottom: 24 }}>
        The complexity router automatically classifies requests by complexity using rule-based scoring (no API calls,
        &lt;1ms latency). Configure which model(s) handle each tier.
      </Text>

      <Card>
        {(Object.keys(TIER_DESCRIPTIONS) as Array<keyof ComplexityTiers>).map((tier, index) => {
          const tierInfo = TIER_DESCRIPTIONS[tier];
          const tierMissing = showValidationErrors && value.tiers[tier].length === 0;
          return (
            <div key={tier}>
              {index > 0 && <Divider style={{ margin: "16px 0" }} />}
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <Text strong style={{ fontSize: 16 }}>
                    {tierInfo.label} Tier
                  </Text>
                  <Tooltip title={tierInfo.description}>
                    <InfoCircleOutlined className="text-gray-400" />
                  </Tooltip>
                </div>
                <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
                  Examples: {tierInfo.examples}
                </Text>
                <AntdSelect
                  mode="multiple"
                  value={value.tiers[tier]}
                  onChange={(models) => handleTierChange(tier, models)}
                  placeholder={`Select model(s) for ${tierInfo.label.toLowerCase()} queries`}
                  showSearch
                  style={{ width: "100%" }}
                  options={modelOptions}
                  status={tierMissing ? "error" : undefined}
                />
                {value.tiers[tier].length > 1 && (
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    Multiple models selected — the router randomly picks among them per request (or Thompson-samples
                    within the pool when adaptive routing is on).
                  </Text>
                )}
                {tierMissing && (
                  <Text type="danger" style={{ fontSize: 12 }}>
                    This tier is required
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
                Advanced: Classification Method
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
              />
            ),
          },
          {
            key: "adaptive",
            label: (
              <Text strong style={{ color: "#374151" }}>
                Advanced: Adaptive Routing
              </Text>
            ),
            children: <AdaptiveRoutingConfig value={value} onChange={onChange} />,
          },
          ...(onKeywordTierRulesChange || onSemanticMatchingEnabledChange
            ? [
                {
                  key: "keyword-semantic",
                  label: (
                    <Text strong style={{ color: "#374151" }}>
                      Advanced: Keyword/Semantic Matching
                    </Text>
                  ),
                  children: (
                    <>
                      {onKeywordTierRulesChange && (
                        <KeywordTierRules rules={keywordTierRules} onChange={onKeywordTierRulesChange} />
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
