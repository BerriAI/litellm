import React from "react";
import { Card, Select as AntdSelect, Typography, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { ModelGroup } from "../playground/llm_calls/fetch_models";

const { Text } = Typography;

interface ComplexityTiers {
  SIMPLE: string;
  MEDIUM: string;
  COMPLEX: string;
  REASONING: string;
}

interface ComplexityRouterConfigProps {
  modelInfo: ModelGroup[];
  value?: ComplexityTiers;
  onChange?: (tiers: ComplexityTiers) => void;
}

const tierDescriptions = {
  SIMPLE: "Quick questions, greetings, simple lookups (e.g., \"What is the capital of France?\")",
  MEDIUM: "Moderate complexity, explanations, summaries",
  COMPLEX: "Code generation, technical analysis, detailed research",
  REASONING: "Multi-step reasoning, complex problem solving, chain-of-thought tasks",
};

const tierLabels = {
  SIMPLE: "Simple Tasks",
  MEDIUM: "Medium Tasks",
  COMPLEX: "Complex Tasks",
  REASONING: "Reasoning Tasks",
};

const ComplexityRouterConfig: React.FC<ComplexityRouterConfigProps> = ({ 
  modelInfo, 
  value, 
  onChange 
}) => {
  const tiers: ComplexityTiers = value || {
    SIMPLE: "",
    MEDIUM: "",
    COMPLEX: "",
    REASONING: "",
  };

  const handleTierChange = (tier: keyof ComplexityTiers, model: string) => {
    const updatedTiers = { ...tiers, [tier]: model };
    onChange?.(updatedTiers);
  };

  // Prepare model options for dropdowns
  const modelOptions = Array.from(
    new Set(modelInfo.map((model) => model.model_group))
  ).map((model_group) => ({
    value: model_group,
    label: model_group,
  }));

  return (
    <div className="w-full">
      <div className="mb-4">
        <Text className="text-gray-600">
          Configure which model handles each complexity tier. Requests are automatically classified and routed — no training data needed.
        </Text>
      </div>

      <Card className="w-full">
        <div className="space-y-6">
          {(["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"] as const).map((tier) => (
            <div key={tier} className="w-full">
              <div className="flex items-center gap-2 mb-2">
                <Text className="text-sm font-medium">{tierLabels[tier]}</Text>
                <Tooltip title={tierDescriptions[tier]}>
                  <InfoCircleOutlined className="text-gray-400" />
                </Tooltip>
              </div>
              <AntdSelect
                value={tiers[tier] || undefined}
                onChange={(value) => handleTierChange(tier, value)}
                placeholder={`Select model for ${tierLabels[tier].toLowerCase()}`}
                showSearch
                style={{ width: "100%" }}
                options={modelOptions}
                allowClear
              />
              <Text className="text-xs text-gray-400 mt-1 block">
                {tierDescriptions[tier]}
              </Text>
            </div>
          ))}
        </div>
      </Card>

      {/* Recommendations */}
      <Card className="mt-4 bg-blue-50 border-blue-200">
        <div className="flex items-start gap-2">
          <InfoCircleOutlined className="text-blue-500 mt-1" />
          <div>
            <Text className="text-sm font-medium text-blue-800 block mb-1">
              Recommendations
            </Text>
            <Text className="text-xs text-blue-700">
              • <strong>Simple:</strong> Use fast, cheap models (e.g., GPT-4o-mini, Gemini Flash)
              <br />
              • <strong>Medium:</strong> Balanced models (e.g., GPT-4o, Claude Sonnet)
              <br />
              • <strong>Complex:</strong> Capable models (e.g., Claude Sonnet, GPT-4o)
              <br />
              • <strong>Reasoning:</strong> Best reasoning models (e.g., Claude Opus, o1-preview)
            </Text>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ComplexityRouterConfig;
