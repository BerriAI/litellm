import { InfoCircleOutlined } from "@ant-design/icons";
import { Select as AntdSelect, Card, Divider, Space, Tooltip, Typography } from "antd";
import React from "react";
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
  value: ComplexityTiers;
  onChange: (tiers: ComplexityTiers) => void;
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

const ComplexityRouterConfig: React.FC<ComplexityRouterConfigProps> = ({ modelInfo, value, onChange }) => {
  // Prepare model options for dropdowns
  const modelOptions = modelInfo.map((model) => ({
    value: model.model_group,
    label: model.model_group,
  }));

  const handleTierChange = (tier: keyof ComplexityTiers, model: string) => {
    onChange({
      ...value,
      [tier]: model,
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
                  value={value[tier]}
                  onChange={(model) => handleTierChange(tier, model)}
                  placeholder={`Select model for ${tierInfo.label.toLowerCase()} queries`}
                  showSearch
                  style={{ width: "100%" }}
                  options={modelOptions}
                />
              </div>
            </div>
          );
        })}
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
    </div>
  );
};

export default ComplexityRouterConfig;
