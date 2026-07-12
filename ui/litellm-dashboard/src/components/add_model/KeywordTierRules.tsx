import { DeleteOutlined, InfoCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, Card, Empty, Select as AntdSelect, Tooltip, Typography } from "antd";
import React from "react";

const { Text } = Typography;

export type ComplexityTier = "SIMPLE" | "MEDIUM" | "COMPLEX" | "REASONING";

export interface KeywordTierRule {
  id: string;
  keywords: string[];
  tier: ComplexityTier;
}

interface KeywordTierRulesProps {
  rules: KeywordTierRule[];
  onChange: (rules: KeywordTierRule[]) => void;
}

const TIER_OPTIONS: { value: ComplexityTier; label: string }[] = [
  { value: "SIMPLE", label: "Simple" },
  { value: "MEDIUM", label: "Medium" },
  { value: "COMPLEX", label: "Complex" },
  { value: "REASONING", label: "Reasoning" },
];

const KeywordTierRules: React.FC<KeywordTierRulesProps> = ({ rules, onChange }) => {
  const addRule = () => {
    onChange([...rules, { id: `${Date.now()}`, keywords: [], tier: "COMPLEX" }]);
  };

  const updateRule = (id: string, updates: Partial<Omit<KeywordTierRule, "id">>) => {
    onChange(rules.map((rule) => (rule.id === id ? { ...rule, ...updates } : rule)));
  };

  const removeRule = (id: string) => {
    onChange(rules.filter((rule) => rule.id !== id));
  };

  return (
    <div className="w-full max-w-none">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Typography.Title level={4} style={{ margin: 0 }}>
            Keyword Tier Overrides
          </Typography.Title>
          <Tooltip title="Match known terms and force the request straight to a chosen complexity tier, bypassing rule-based scoring.">
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </div>
        <Button icon={<PlusOutlined />} onClick={addRule}>
          Add keyword rule
        </Button>
      </div>
      <Text type="secondary" style={{ display: "block", marginBottom: 16 }}>
        Optional: route requests containing specific keywords directly to a tier, e.g. route &quot;invoice, refund,
        billing&quot; to the medium tier.
      </Text>

      {rules.length === 0 ? (
        <Card className="bg-gray-50">
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No keyword tier overrides configured" />
        </Card>
      ) : (
        <div className="flex flex-col gap-3">
          {rules.map((rule, index) => (
            <Card key={rule.id} size="small">
              <div className="flex items-end gap-3">
                <div className="flex-1">
                  <Text strong style={{ display: "block", marginBottom: 8 }}>
                    Keywords {index + 1}
                  </Text>
                  <AntdSelect
                    mode="tags"
                    value={rule.keywords}
                    onChange={(keywords: string[]) => updateRule(rule.id, { keywords })}
                    placeholder="e.g., invoice, refund, billing"
                    tokenSeparators={[","]}
                    open={false}
                    suffixIcon={null}
                    style={{ width: "100%" }}
                    allowClear
                  />
                </div>
                <div style={{ width: 220 }}>
                  <Text strong style={{ display: "block", marginBottom: 8 }}>
                    Route to tier
                  </Text>
                  <AntdSelect
                    value={rule.tier}
                    onChange={(tier: ComplexityTier) => updateRule(rule.id, { tier })}
                    options={TIER_OPTIONS}
                    style={{ width: "100%" }}
                  />
                </div>
                <Button
                  danger
                  type="text"
                  icon={<DeleteOutlined />}
                  aria-label={`Remove keyword rule ${index + 1}`}
                  onClick={() => removeRule(rule.id)}
                />
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default KeywordTierRules;
