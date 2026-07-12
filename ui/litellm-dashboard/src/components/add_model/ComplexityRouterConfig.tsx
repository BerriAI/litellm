import { InfoCircleOutlined } from "@ant-design/icons";
import { Select as AntdSelect, Card, Collapse, Divider, InputNumber, Radio, Space, Tooltip, Typography } from "antd";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import KeywordTierRules, { KeywordTierRule } from "./KeywordTierRules";
import SemanticKeywordMatching from "./SemanticKeywordMatching";

const { Text } = Typography;

export const DEFAULT_CLASSIFIER_TIMEOUT_MS = 3000;

export interface ComplexityTiers {
  SIMPLE: string;
  MEDIUM: string;
  COMPLEX: string;
  REASONING: string;
}

export interface ClassifierLLMConfig {
  model: string;
  timeout_ms: number;
}

export type ClassifierType = "heuristic" | "llm";

export interface ComplexityRouterConfigValue {
  tiers: ComplexityTiers;
  classifier_type: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
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

  const handleTierChange = (tier: keyof ComplexityTiers, model: string) => {
    onChange({
      ...value,
      tiers: { ...value.tiers, [tier]: model },
    });
  };

  const handleClassifierTypeChange = (classifierType: ClassifierType) => {
    onChange({
      ...value,
      classifier_type: classifierType,
      classifier_llm_config:
        classifierType === "llm"
          ? value.classifier_llm_config ?? { model: "", timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS }
          : undefined,
    });
  };

  const handleClassifierModelChange = (model: string) => {
    onChange({
      ...value,
      classifier_llm_config: {
        model,
        timeout_ms: value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
      },
    });
  };

  const handleClassifierTimeoutChange = (timeoutMs: number | null) => {
    onChange({
      ...value,
      classifier_llm_config: {
        model: value.classifier_llm_config?.model ?? "",
        timeout_ms: timeoutMs ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
      },
    });
  };

  return (
    <div className="w-full max-w-none">
      <Space align="center" style={{ marginBottom: 16 }}>
        <Typography.Title level={4} style={{ margin: 0 }}>
          Complexity Tier Configuration
        </Typography.Title>
        <Tooltip title="Map each complexity tier to a model. Simple queries use cheaper/faster models, complex queries use more capable models.">
          <InfoCircleOutlined className="text-gray-400" />
        </Tooltip>
      </Space>

      <Text type="secondary" style={{ display: "block", marginBottom: 24 }}>
        The complexity router automatically classifies requests by complexity using rule-based scoring (no API calls,
        &lt;1ms latency). Configure which model handles each tier.
      </Text>

      <Card>
        {(Object.keys(TIER_DESCRIPTIONS) as Array<keyof ComplexityTiers>).map((tier, index) => {
          const tierInfo = TIER_DESCRIPTIONS[tier];
          const tierMissing = showValidationErrors && !value.tiers[tier];
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
                  value={value.tiers[tier]}
                  onChange={(model) => handleTierChange(tier, model)}
                  placeholder={`Select model for ${tierInfo.label.toLowerCase()} queries`}
                  showSearch
                  style={{ width: "100%" }}
                  options={modelOptions}
                  status={tierMissing ? "error" : undefined}
                />
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
              <>
                <Radio.Group
                  value={value.classifier_type}
                  onChange={(e) => handleClassifierTypeChange(e.target.value)}
                  className="w-full"
                >
                  <Space direction="vertical" className="w-full">
                    <Radio value="heuristic">
                      <Text strong>Heuristic</Text>{" "}
                      <Text type="secondary">(default) — rule-based scoring, no API calls, &lt;1ms latency</Text>
                    </Radio>
                    <Radio value="llm">
                      <Text strong>LLM Classifier</Text>{" "}
                      <Text type="secondary">— use a model to decide the tier (e.g. a small/fast model)</Text>
                    </Radio>
                  </Space>
                </Radio.Group>

                {value.classifier_type === "llm" && (
                  <div className="mt-4 space-y-3">
                    <div>
                      <Text strong style={{ display: "block", marginBottom: 4 }}>
                        Classifier Model
                      </Text>
                      <AntdSelect
                        value={value.classifier_llm_config?.model || undefined}
                        onChange={handleClassifierModelChange}
                        placeholder="Select the model that will classify request complexity"
                        showSearch
                        style={{ width: "100%" }}
                        options={modelOptions}
                      />
                    </div>
                    <div>
                      <Text strong style={{ display: "block", marginBottom: 4 }}>
                        Timeout (ms)
                      </Text>
                      <InputNumber
                        value={value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS}
                        onChange={handleClassifierTimeoutChange}
                        min={1}
                        style={{ width: "100%" }}
                      />
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        Falls back to the heuristic scorer if the classifier call errors, times out, or returns an
                        unparseable response.
                      </Text>
                    </div>
                  </div>
                )}
              </>
            ),
          },
        ]}
      />

      <Divider />

      <Card>
        <div className="flex items-center gap-2 mb-2">
          <Text strong style={{ fontSize: 16 }}>
            Custom Technical Keywords
          </Text>
          <Tooltip title="Domain-specific terms appended to the built-in technical keyword list. Prompts containing these terms score higher on the technical dimension and route to more capable models.">
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </div>
        <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
          Optional: Add terms to the built-in list to improve classification accuracy on the technical dimension. (e.g.,
          udp, kafka, terraform).
        </Text>
        <AntdSelect
          mode="tags"
          value={customTechnicalKeywords ?? []}
          onChange={(keywords: string[]) => onCustomTechnicalKeywordsChange?.(keywords)}
          placeholder="Type a keyword and press Enter, or paste a comma-separated list"
          tokenSeparators={[","]}
          open={false}
          suffixIcon={null}
          style={{ width: "100%" }}
          allowClear
        />
      </Card>

      <Divider />

      <Card className="bg-gray-50">
        <Text strong style={{ display: "block", marginBottom: 8 }}>
          How Classification Works
        </Text>
        <Text type="secondary" style={{ fontSize: 13 }}>
          The router scores each request across 7 dimensions: token count, code presence, reasoning markers, technical
          terms, simple indicators, multi-step patterns, and question complexity. The weighted score determines the
          tier:
        </Text>
        <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20, fontSize: 13, color: "rgba(0, 0, 0, 0.45)" }}>
          <li>
            <strong>SIMPLE</strong>: Score &lt; 0.15
          </li>
          <li>
            <strong>MEDIUM</strong>: Score 0.15 - 0.35
          </li>
          <li>
            <strong>COMPLEX</strong>: Score 0.35 - 0.60
          </li>
          <li>
            <strong>REASONING</strong>: Score &gt; 0.60 (or 2+ reasoning markers)
          </li>
        </ul>
      </Card>

      {/* Keyword-tier and semantic sections only render when their change handlers are
          wired (the add-router flow). The edit-auto-router modal doesn't pass them yet, so
          they stay hidden there rather than rendering interactive-but-dead controls. */}
      {onKeywordTierRulesChange && (
        <>
          <Divider />
          <KeywordTierRules rules={keywordTierRules} onChange={onKeywordTierRulesChange} />
        </>
      )}

      {onSemanticMatchingEnabledChange && (
        <>
          <Divider />
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
        </>
      )}
    </div>
  );
};

export default ComplexityRouterConfig;
