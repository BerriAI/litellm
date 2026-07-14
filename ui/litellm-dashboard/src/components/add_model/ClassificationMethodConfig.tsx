import { Select as AntdSelect, Card, InputNumber, Radio, Space, Typography } from "antd";
import React from "react";
import { ClassifierType, ComplexityRouterConfigValue, DEFAULT_CLASSIFIER_TIMEOUT_MS } from "./ComplexityRouterConfig";

const { Text } = Typography;

interface ClassificationMethodConfigProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelOptions: { value: string; label: string }[];
  customTechnicalKeywords?: string[];
  onCustomTechnicalKeywordsChange?: (keywords: string[]) => void;
  showValidationErrors?: boolean;
}

const ClassificationMethodConfig: React.FC<ClassificationMethodConfigProps> = ({
  value,
  onChange,
  modelOptions,
  customTechnicalKeywords,
  onCustomTechnicalKeywordsChange,
  showValidationErrors = false,
}) => {
  const classifierModelMissing =
    showValidationErrors && value.classifier_type === "llm" && !value.classifier_llm_config?.model;

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
              status={classifierModelMissing ? "error" : undefined}
            />
            {classifierModelMissing && (
              <Text type="danger" style={{ fontSize: 12 }}>
                A classifier model is required
              </Text>
            )}
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
              Falls back to the heuristic scorer if the classifier call errors, times out, or returns an unparseable
              response.
            </Text>
          </div>
        </div>
      )}

      {value.classifier_type === "heuristic" && (
        <div className="mt-4">
          <Text strong style={{ display: "block", marginBottom: 4 }}>
            Custom Technical Keywords
          </Text>
          <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
            Optional: Add terms to the built-in list to improve classification accuracy on the technical dimension.
            (e.g., udp, kafka, terraform).
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
        </div>
      )}

      <Card className="bg-gray-50 mt-4">
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
    </>
  );
};

export default ClassificationMethodConfig;
