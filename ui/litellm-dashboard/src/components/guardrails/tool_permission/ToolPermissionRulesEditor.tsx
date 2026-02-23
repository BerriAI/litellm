import React from "react";
import { Card, Text } from "@tremor/react";
import { Button, Divider, Empty, Input, Select, Space, Tooltip } from "antd";
import { InfoCircleOutlined, PlusOutlined, DeleteOutlined } from "@ant-design/icons";

export type ToolPermissionDecision = "allow" | "deny";
export type ToolPermissionDefaultAction = "allow" | "deny";
export type ToolPermissionOnDisallowedAction = "block" | "rewrite";

export interface ToolPermissionRuleConfig {
  id: string;
  tool_name?: string;
  tool_type?: string;
  decision: ToolPermissionDecision;
  allowed_param_patterns?: Record<string, string>;
}

export interface ToolPermissionConfig {
  rules: ToolPermissionRuleConfig[];
  default_action: ToolPermissionDefaultAction;
  on_disallowed_action: ToolPermissionOnDisallowedAction;
  violation_message_template?: string;
}

interface ToolPermissionRulesEditorProps {
  value?: ToolPermissionConfig;
  onChange?: (config: ToolPermissionConfig) => void;
  disabled?: boolean;
}

const DEFAULT_CONFIG: ToolPermissionConfig = {
  rules: [],
  default_action: "deny",
  on_disallowed_action: "block",
  violation_message_template: "",
};

const ensureConfig = (config?: ToolPermissionConfig): ToolPermissionConfig => ({
  ...DEFAULT_CONFIG,
  ...(config || {}),
  rules: config?.rules ? [...config.rules] : [],
});

const ToolPermissionRulesEditor: React.FC<ToolPermissionRulesEditorProps> = ({
  value,
  onChange,
  disabled = false,
}) => {
  const config = ensureConfig(value);

  const updateConfig = (partial: Partial<ToolPermissionConfig>) => {
    const nextConfig: ToolPermissionConfig = {
      ...config,
      ...partial,
    };
    onChange?.(nextConfig);
  };

  const updateRule = (ruleIndex: number, updates: Partial<ToolPermissionRuleConfig>) => {
    const nextRules = config.rules.map((rule, index) =>
      index === ruleIndex ? { ...rule, ...updates } : rule,
    );
    updateConfig({ rules: nextRules });
  };

  const addRule = () => {
    const nextRules = [
      ...config.rules,
      {
        id: `rule_${Math.random().toString(36).slice(2, 8)}`,
        decision: "allow" as ToolPermissionDecision,
        allowed_param_patterns: undefined,
      },
    ];
    updateConfig({ rules: nextRules });
  };

  const removeRule = (ruleIndex: number) => {
    const nextRules = config.rules.filter((_, index) => index !== ruleIndex);
    updateConfig({ rules: nextRules });
  };

  const updateAllowedParamEntries = (
    ruleIndex: number,
    mutate: (entries: [string, string][]) => void,
  ) => {
    const targetRule = config.rules[ruleIndex];
    if (!targetRule) {
      return;
    }
    const entries = Object.entries(targetRule.allowed_param_patterns || {});
    mutate(entries);
    const updatedObject: Record<string, string> = {};
    entries.forEach(([key, value]) => {
      updatedObject[key] = value;
    });
    updateRule(ruleIndex, {
      allowed_param_patterns:
        Object.keys(updatedObject).length > 0 ? updatedObject : undefined,
    });
  };

  const updateAllowedParamPath = (
    ruleIndex: number,
    entryIndex: number,
    nextPath: string,
  ) => {
    updateAllowedParamEntries(ruleIndex, (entries) => {
      if (!entries[entryIndex]) {
        return;
      }
      const [, value] = entries[entryIndex];
      entries[entryIndex] = [nextPath, value];
    });
  };

  const updateAllowedParamPattern = (
    ruleIndex: number,
    entryIndex: number,
    pattern: string,
  ) => {
    updateAllowedParamEntries(ruleIndex, (entries) => {
      if (!entries[entryIndex]) {
        return;
      }
      const [path] = entries[entryIndex];
      entries[entryIndex] = [path, pattern];
    });
  };

  const renderAllowedParamPatterns = (rule: ToolPermissionRuleConfig, index: number) => {
    const entries = Object.entries(rule.allowed_param_patterns || {});
    if (entries.length === 0) {
      return (
        <Button
          disabled={disabled}
          size="small"
          onClick={() => updateRule(index, { allowed_param_patterns: { "": "" } })}
        >
          + Restrict tool arguments (optional)
        </Button>
      );
    }

    return (
      <div className="space-y-2">
        <Text className="text-sm text-gray-600">Argument constraints (dot or array paths)</Text>
        {entries.map(([path, pattern], patternIndex) => (
          <Space key={`${rule.id || index}-${patternIndex}`} align="start">
            <Input
              disabled={disabled}
              placeholder="messages[0].content"
              value={path}
              onChange={(e) => updateAllowedParamPath(index, patternIndex, e.target.value)}
            />
            <Input
              disabled={disabled}
              placeholder="^email@.*$"
              value={pattern}
              onChange={(e) => updateAllowedParamPattern(index, patternIndex, e.target.value)}
            />
            <Button
              disabled={disabled}
              icon={<DeleteOutlined />}
              danger
              onClick={() =>
                updateAllowedParamEntries(index, (entries) => {
                  entries.splice(patternIndex, 1);
                })
              }
            />
          </Space>
        ))}
        <Button
          disabled={disabled}
          size="small"
          onClick={() =>
            updateRule(index, {
              allowed_param_patterns: {
                ...(rule.allowed_param_patterns || {}),
                "": "",
              },
            })
          }
        >
          + Add another constraint
        </Button>
      </div>
    );
  };

  return (
    <Card>
      <div className="flex items-center justify-between">
        <div>
          <Text className="text-lg font-semibold">LiteLLM Tool Permission Guardrail</Text>
          <Text className="text-sm text-gray-500">
            Provide regex patterns (e.g., ^mcp__github_.*$) for tool names or types and optionally
            constrain payload fields.
          </Text>
        </div>
        {!disabled && (
          <Button
            icon={<PlusOutlined />}
            type="primary"
            onClick={addRule}
            className="!bg-blue-600 !text-white hover:!bg-blue-500"
          >
            Add Rule
          </Button>
        )}
      </div>

      <Divider />

      {config.rules.length === 0 ? (
        <Empty description="No tool rules added yet" />
      ) : (
        <div className="space-y-4">
          {config.rules.map((rule, index) => (
            <Card key={rule.id || index} className="bg-gray-50">
              <div className="flex items-center justify-between mb-3">
                <Text className="font-semibold">Rule {index + 1}</Text>
                <Button
                  icon={<DeleteOutlined />}
                  danger
                  type="text"
                  disabled={disabled}
                  onClick={() => removeRule(index)}
                >
                  Remove
                </Button>
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <Text className="text-sm font-medium">Rule ID</Text>
                  <Input
                    disabled={disabled}
                    placeholder="unique_rule_id"
                    value={rule.id}
                    onChange={(e) => updateRule(index, { id: e.target.value })}
                  />
                </div>
                <div>
                  <Text className="text-sm font-medium">Tool Name (optional)</Text>
                  <Input
                    disabled={disabled}
                    placeholder="^mcp__github_.*$"
                    value={rule.tool_name ?? ""}
                    onChange={(e) =>
                      updateRule(index, {
                        tool_name: e.target.value.trim() === "" ? undefined : e.target.value,
                      })
                    }
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 mt-4">
                <div>
                  <Text className="text-sm font-medium">Tool Type (optional)</Text>
                  <Input
                    disabled={disabled}
                    placeholder="^function$"
                    value={rule.tool_type ?? ""}
                    onChange={(e) =>
                      updateRule(index, {
                        tool_type: e.target.value.trim() === "" ? undefined : e.target.value,
                      })
                    }
                  />
                </div>
              </div>

              <div className="mt-4 flex flex-col gap-2">
                <Text className="text-sm font-medium">Decision</Text>
                <Select
                  disabled={disabled}
                  value={rule.decision}
                  style={{ width: 200 }}
                  onChange={(value) => updateRule(index, { decision: value as ToolPermissionDecision })}
                >
                  <Select.Option value="allow">Allow</Select.Option>
                  <Select.Option value="deny">Deny</Select.Option>
                </Select>
              </div>

              <div className="mt-4">{renderAllowedParamPatterns(rule, index)}</div>
            </Card>
          ))}
        </div>
      )}

      <Divider />

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <Text className="text-sm font-medium">Default action</Text>
          <Select
            disabled={disabled}
            value={config.default_action}
            onChange={(value) => updateConfig({ default_action: value as ToolPermissionDefaultAction })}
          >
            <Select.Option value="allow">Allow</Select.Option>
            <Select.Option value="deny">Deny</Select.Option>
          </Select>
        </div>
        <div>
          <Text className="text-sm font-medium flex items-center gap-1">
            On disallowed action
            <Tooltip title="Block returns an error when a forbidden tool is invoked. Rewrite strips the tool call but lets the rest of the response continue.">
              <InfoCircleOutlined />
            </Tooltip>
          </Text>
          <Select
            disabled={disabled}
            value={config.on_disallowed_action}
            onChange={(value) =>
              updateConfig({ on_disallowed_action: value as ToolPermissionOnDisallowedAction })
            }
          >
            <Select.Option value="block">Block</Select.Option>
            <Select.Option value="rewrite">Rewrite</Select.Option>
          </Select>
        </div>
      </div>

      <div className="mt-4">
        <Text className="text-sm font-medium">Violation message (optional)</Text>
        <Input.TextArea
          disabled={disabled}
          rows={3}
          placeholder="This violates our org policy..."
          value={config.violation_message_template}
          onChange={(e) => updateConfig({ violation_message_template: e.target.value })}
        />
      </div>
    </Card>
  );
};

export default ToolPermissionRulesEditor;
