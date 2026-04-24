import React from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Info as InfoCircleOutlined,
  Plus as PlusOutlined,
  Trash2 as DeleteOutlined,
} from "lucide-react";

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

  const updateRule = (
    ruleIndex: number,
    updates: Partial<ToolPermissionRuleConfig>,
  ) => {
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
    entries.forEach(([key, v]) => {
      updatedObject[key] = v;
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
      const [, v] = entries[entryIndex];
      entries[entryIndex] = [nextPath, v];
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

  const renderAllowedParamPatterns = (
    rule: ToolPermissionRuleConfig,
    index: number,
  ) => {
    const entries = Object.entries(rule.allowed_param_patterns || {});
    if (entries.length === 0) {
      return (
        <Button
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={() =>
            updateRule(index, { allowed_param_patterns: { "": "" } })
          }
        >
          + Restrict tool arguments (optional)
        </Button>
      );
    }

    return (
      <div className="space-y-2">
        <span className="text-sm text-muted-foreground">
          Argument constraints (dot or array paths)
        </span>
        {entries.map(([path, pattern], patternIndex) => (
          <div
            key={`${rule.id || index}-${patternIndex}`}
            className="flex items-start gap-2"
          >
            <Input
              disabled={disabled}
              placeholder="messages[0].content"
              value={path}
              onChange={(e) =>
                updateAllowedParamPath(index, patternIndex, e.target.value)
              }
            />
            <Input
              disabled={disabled}
              placeholder="^email@.*$"
              value={pattern}
              onChange={(e) =>
                updateAllowedParamPattern(index, patternIndex, e.target.value)
              }
            />
            <Button
              variant="destructive"
              size="icon"
              disabled={disabled}
              onClick={() =>
                updateAllowedParamEntries(index, (entries) => {
                  entries.splice(patternIndex, 1);
                })
              }
              aria-label="Remove constraint"
            >
              <DeleteOutlined className="h-4 w-4" />
            </Button>
          </div>
        ))}
        <Button
          variant="outline"
          size="sm"
          disabled={disabled}
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
    <Card className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-lg font-semibold">
            LiteLLM Tool Permission Guardrail
          </span>
          <p className="text-sm text-muted-foreground m-0">
            Provide regex patterns (e.g., ^mcp__github_.*$) for tool names or
            types and optionally constrain payload fields.
          </p>
        </div>
        {!disabled && (
          <Button onClick={addRule}>
            <PlusOutlined className="h-4 w-4 mr-1" />
            Add Rule
          </Button>
        )}
      </div>

      <Separator className="my-4" />

      {config.rules.length === 0 ? (
        <div className="py-10 text-center text-muted-foreground">
          No tool rules added yet
        </div>
      ) : (
        <div className="space-y-4">
          {config.rules.map((rule, index) => (
            <Card key={rule.id || index} className="bg-muted p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="font-semibold">Rule {index + 1}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={disabled}
                  onClick={() => removeRule(index)}
                  className="text-destructive hover:text-destructive"
                >
                  <DeleteOutlined className="h-4 w-4 mr-1" />
                  Remove
                </Button>
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Rule ID</Label>
                  <Input
                    disabled={disabled}
                    placeholder="unique_rule_id"
                    value={rule.id}
                    onChange={(e) =>
                      updateRule(index, { id: e.target.value })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-sm font-medium">
                    Tool Name (optional)
                  </Label>
                  <Input
                    disabled={disabled}
                    placeholder="^mcp__github_.*$"
                    value={rule.tool_name ?? ""}
                    onChange={(e) =>
                      updateRule(index, {
                        tool_name:
                          e.target.value.trim() === ""
                            ? undefined
                            : e.target.value,
                      })
                    }
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 mt-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium">
                    Tool Type (optional)
                  </Label>
                  <Input
                    disabled={disabled}
                    placeholder="^function$"
                    value={rule.tool_type ?? ""}
                    onChange={(e) =>
                      updateRule(index, {
                        tool_type:
                          e.target.value.trim() === ""
                            ? undefined
                            : e.target.value,
                      })
                    }
                  />
                </div>
              </div>

              <div className="mt-4 flex flex-col gap-2">
                <Label className="text-sm font-medium">Decision</Label>
                <Select
                  value={rule.decision}
                  disabled={disabled}
                  onValueChange={(value) =>
                    updateRule(index, {
                      decision: value as ToolPermissionDecision,
                    })
                  }
                >
                  <SelectTrigger className="w-[200px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="allow">Allow</SelectItem>
                    <SelectItem value="deny">Deny</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="mt-4">
                {renderAllowedParamPatterns(rule, index)}
              </div>
            </Card>
          ))}
        </div>
      )}

      <Separator className="my-4" />

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label className="text-sm font-medium">Default action</Label>
          <Select
            value={config.default_action}
            disabled={disabled}
            onValueChange={(value) =>
              updateConfig({
                default_action: value as ToolPermissionDefaultAction,
              })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="allow">Allow</SelectItem>
              <SelectItem value="deny">Deny</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label className="text-sm font-medium flex items-center gap-1">
            On disallowed action
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span>
                    <InfoCircleOutlined className="h-3.5 w-3.5 text-muted-foreground" />
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  Block returns an error when a forbidden tool is invoked.
                  Rewrite strips the tool call but lets the rest of the
                  response continue.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </Label>
          <Select
            value={config.on_disallowed_action}
            disabled={disabled}
            onValueChange={(value) =>
              updateConfig({
                on_disallowed_action:
                  value as ToolPermissionOnDisallowedAction,
              })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="block">Block</SelectItem>
              <SelectItem value="rewrite">Rewrite</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        <Label className="text-sm font-medium">
          Violation message (optional)
        </Label>
        <Textarea
          disabled={disabled}
          rows={3}
          placeholder="This violates our org policy..."
          value={config.violation_message_template}
          onChange={(e) =>
            updateConfig({ violation_message_template: e.target.value })
          }
        />
      </div>
    </Card>
  );
};

export default ToolPermissionRulesEditor;
