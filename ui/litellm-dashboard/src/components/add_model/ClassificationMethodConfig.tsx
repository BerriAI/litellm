import { InfoCircleOutlined } from "@ant-design/icons";
import { Select as AntdSelect, Card, InputNumber, Radio, Space, Switch, Tooltip, Typography } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import ClassifierPromptEditor from "./ClassifierPromptEditor";
import {
  ClassifierFallback,
  ClassifierType,
  ComplexityRouterConfigValue,
  DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS,
  DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
  DEFAULT_CLASSIFIER_FALLBACK,
  DEFAULT_CLASSIFIER_TIMEOUT_MS,
} from "./ComplexityRouterConfig";

const { Text } = Typography;

/**
 * What the scoring breakdown below it actually describes. A custom prompt means the score no longer
 * decides the tier, and pairing one with the default-model fallback means the heuristic never runs
 * at all, so the panel must not keep implying a score is involved on either router.
 */
const scoringExplanation = (value: ComplexityRouterConfigValue, t: TFunction<"gateway">): string => {
  const usesCustomPrompt =
    value.classifier_type === "llm" && Boolean(value.classifier_llm_config?.system_prompt?.trim());
  if (!usesCustomPrompt) return t("models.autoRouters.details.classification.scoringDefault");
  return value.classifier_fallback === "default_model"
    ? t("models.autoRouters.details.classification.scoringCustomDefault")
    : t("models.autoRouters.details.classification.scoringCustomHeuristic");
};

interface ClassificationMethodConfigProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelOptions: { value: string; label: string }[];
  customTechnicalKeywords?: string[];
  onCustomTechnicalKeywordsChange?: (keywords: string[]) => void;
  showValidationErrors?: boolean;
  /** Enables the default-model fallback, which the backend rejects without a default model. */
  hasDefaultModel?: boolean;
}

const ClassificationMethodConfig: React.FC<ClassificationMethodConfigProps> = ({
  value,
  onChange,
  modelOptions,
  customTechnicalKeywords,
  onCustomTechnicalKeywordsChange,
  showValidationErrors = false,
  hasDefaultModel = false,
}) => {
  const { t } = useTranslation("gateway");
  const classifierModelMissing =
    showValidationErrors && value.classifier_type === "llm" && !value.classifier_llm_config?.model;
  const displayTierLabel = (tier: "SIMPLE" | "MEDIUM" | "COMPLEX" | "REASONING") =>
    value.tier_labels?.[tier]?.trim() ||
    t(`models.autoRouters.details.tiers.${tier.toLowerCase()}.label`, { defaultValue: tier });

  const handleClassifierTypeChange = (classifierType: ClassifierType) => {
    const nextValue: ComplexityRouterConfigValue = {
      ...value,
      classifier_type: classifierType,
      classifier_llm_config:
        classifierType === "llm"
          ? value.classifier_llm_config ?? { model: "", timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS }
          : undefined,
      classifier_context_window_size:
        classifierType === "llm"
          ? value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE
          : undefined,
      classifier_context_per_turn_chars:
        classifierType === "llm"
          ? value.classifier_context_per_turn_chars ?? DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS
          : undefined,
      classifier_context_include_assistant_turns:
        classifierType === "llm" ? value.classifier_context_include_assistant_turns : undefined,
      classifier_fallback: classifierType === "llm" ? value.classifier_fallback : undefined,
    };
    onChange(nextValue);
  };

  const handleClassifierModelChange = (model: string) => {
    onChange({
      ...value,
      classifier_llm_config: {
        ...value.classifier_llm_config,
        model,
        timeout_ms: value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
      },
    });
  };

  const handleClassifierTimeoutChange = (timeoutMs: number | null) => {
    onChange({
      ...value,
      classifier_llm_config: {
        ...value.classifier_llm_config,
        model: value.classifier_llm_config?.model ?? "",
        timeout_ms: timeoutMs ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
      },
    });
  };

  const handleClassifierSystemPromptChange = (systemPrompt: string | undefined) => {
    onChange({
      ...value,
      classifier_llm_config: {
        ...value.classifier_llm_config,
        model: value.classifier_llm_config?.model ?? "",
        timeout_ms: value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
        system_prompt: systemPrompt,
      },
    });
  };

  const handleClassifierFallbackChange = (fallback: ClassifierFallback) => {
    onChange({ ...value, classifier_fallback: fallback });
  };

  const handleClassifierContextWindowSizeChange = (windowSize: number | null) => {
    onChange({
      ...value,
      classifier_context_window_size: windowSize ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
    });
  };

  const handleClassifierContextPerTurnCharsChange = (perTurnChars: number | null) => {
    onChange({
      ...value,
      classifier_context_per_turn_chars: perTurnChars ?? DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS,
    });
  };

  const handleClassifierContextIncludeAssistantTurnsChange = (includeAssistantTurns: boolean) => {
    onChange({
      ...value,
      classifier_context_include_assistant_turns: includeAssistantTurns,
    });
  };

  return (
    <>
      <Radio.Group
        value={value.classifier_type}
        onChange={(e) => handleClassifierTypeChange(e.target.value)}
        className="w-full"
      >
        <Space direction="vertical" className="w-full">
          <Radio value="heuristic">
            <Text strong>{t("models.autoRouters.details.classification.heuristic")}</Text>{" "}
            <Text type="secondary">{t("models.autoRouters.details.classification.heuristicDescription")}</Text>
          </Radio>
          <Radio value="llm">
            <Text strong>{t("models.autoRouters.details.classification.llm")}</Text>{" "}
            <Text type="secondary">{t("models.autoRouters.details.classification.llmDescription")}</Text>
          </Radio>
        </Space>
      </Radio.Group>

      {value.classifier_type === "llm" && (
        <div className="mt-4 space-y-3">
          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              {t("models.autoRouters.details.classification.model")}
            </Text>
            <AntdSelect
              value={value.classifier_llm_config?.model || undefined}
              onChange={handleClassifierModelChange}
              placeholder={t("models.autoRouters.details.classification.modelPlaceholder")}
              showSearch
              style={{ width: "100%" }}
              options={modelOptions}
              status={classifierModelMissing ? "error" : undefined}
            />
            {classifierModelMissing && (
              <Text type="danger" style={{ fontSize: 12 }}>
                {t("models.autoRouters.details.classification.modelRequired")}
              </Text>
            )}
          </div>
          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              {t("models.autoRouters.details.classification.timeout")}
            </Text>
            <InputNumber
              value={value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS}
              onChange={handleClassifierTimeoutChange}
              min={1}
              style={{ width: "100%" }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {t("models.autoRouters.details.classification.timeoutDescription")}
            </Text>
          </div>
          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              {t("models.autoRouters.details.classification.prompt")}
            </Text>
            <ClassifierPromptEditor
              systemPrompt={value.classifier_llm_config?.system_prompt}
              onChange={handleClassifierSystemPromptChange}
              contextWindowSize={value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE}
              tierLabels={value.tier_labels}
            />
          </div>
          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              {t("models.autoRouters.details.classification.failure")}
            </Text>
            <Radio.Group
              value={value.classifier_fallback ?? DEFAULT_CLASSIFIER_FALLBACK}
              onChange={(e) => handleClassifierFallbackChange(e.target.value)}
            >
              <Space direction="vertical">
                <Radio value="heuristic">
                  <Text>{t("models.autoRouters.details.classification.heuristicFallback")}</Text>{" "}
                  <Text type="secondary">
                    {t("models.autoRouters.details.classification.heuristicFallbackDescription")}
                  </Text>
                </Radio>
                <Radio value="default_model" disabled={!hasDefaultModel}>
                  <Tooltip
                    title={hasDefaultModel ? undefined : t("models.autoRouters.details.classification.defaultRequired")}
                  >
                    <span>
                      <Text>{t("models.autoRouters.details.classification.defaultFallback")}</Text>{" "}
                      <Text type="secondary">
                        {t("models.autoRouters.details.classification.defaultFallbackDescription")}
                      </Text>
                    </span>
                  </Tooltip>
                </Radio>
              </Space>
            </Radio.Group>
            <Text type="secondary" style={{ display: "block", fontSize: 12 }}>
              {t("models.autoRouters.details.classification.fallbackDescription")}
            </Text>
          </div>
          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              {t("models.autoRouters.details.classification.contextWindow")}
            </Text>
            <InputNumber
              value={value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE}
              onChange={handleClassifierContextWindowSizeChange}
              min={0}
              style={{ width: "100%" }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {t("models.autoRouters.details.classification.contextWindowDescription")}
            </Text>
          </div>
          <div>
            <Text strong style={{ display: "block", marginBottom: 4 }}>
              {t("models.autoRouters.details.classification.perTurnLimit")}
            </Text>
            <InputNumber
              value={value.classifier_context_per_turn_chars ?? DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS}
              onChange={handleClassifierContextPerTurnCharsChange}
              min={1}
              style={{ width: "100%" }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {t("models.autoRouters.details.classification.perTurnLimitDescription")}
            </Text>
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Switch
                checked={value.classifier_context_include_assistant_turns ?? false}
                onChange={handleClassifierContextIncludeAssistantTurnsChange}
                size="small"
                aria-label={t("models.autoRouters.details.classification.includeAssistant")}
              />
              <Text strong>{t("models.autoRouters.details.classification.includeAssistant")}</Text>
              <Tooltip title={t("models.autoRouters.details.classification.includeAssistantTooltip")}>
                <InfoCircleOutlined className="text-gray-400" />
              </Tooltip>
            </div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {t("models.autoRouters.details.classification.includeAssistantDescription")}
            </Text>
          </div>
        </div>
      )}

      {value.classifier_type === "heuristic" && (
        <div className="mt-4">
          <div className="flex items-center gap-2 mb-1">
            <Text strong>{t("models.autoRouters.details.classification.customKeywords")}</Text>
            <Tooltip title={t("models.autoRouters.details.classification.customKeywordsTooltip")}>
              <InfoCircleOutlined className="text-gray-400" />
            </Tooltip>
          </div>
          <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
            {t("models.autoRouters.details.classification.customKeywordsDescription")}
          </Text>
          <AntdSelect
            mode="tags"
            value={customTechnicalKeywords ?? []}
            onChange={(keywords: string[]) => onCustomTechnicalKeywordsChange?.(keywords)}
            placeholder={t("models.autoRouters.details.classification.customKeywordsPlaceholder")}
            tokenSeparators={[","]}
            open={false}
            suffixIcon={null}
            style={{ width: "100%" }}
            allowClear
          />
        </div>
      )}

      <Card className="bg-gray-50 mt-4">
        <Text strong style={{ display: "block", marginBottom: 8 }}>
          {t("models.autoRouters.details.classification.howItWorks")}
        </Text>
        <Text type="secondary" style={{ fontSize: 13 }}>
          {scoringExplanation(value, t)}
        </Text>
        <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20, fontSize: 13, color: "rgba(0, 0, 0, 0.45)" }}>
          <li>
            <strong>{displayTierLabel("SIMPLE")}</strong>: {t("models.autoRouters.details.classification.score")} &lt;
            0.15
          </li>
          <li>
            <strong>{displayTierLabel("MEDIUM")}</strong>: {t("models.autoRouters.details.classification.score")} 0.15 -
            0.35
          </li>
          <li>
            <strong>{displayTierLabel("COMPLEX")}</strong>: {t("models.autoRouters.details.classification.score")} 0.35
            - 0.60
          </li>
          <li>
            <strong>{displayTierLabel("REASONING")}</strong>: {t("models.autoRouters.details.classification.score")}{" "}
            &gt; 0.60 ({t("models.autoRouters.details.classification.reasoningMarkers")})
          </li>
        </ul>
      </Card>
    </>
  );
};

export default ClassificationMethodConfig;
