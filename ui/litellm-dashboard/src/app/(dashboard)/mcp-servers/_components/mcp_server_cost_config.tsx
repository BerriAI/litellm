import React, { useState } from "react";
import { Info, DollarSign, Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { MCPServerCostInfo } from "@/components/mcp_tools/types";

interface MCPServerCostConfigProps {
  value?: MCPServerCostInfo;
  onChange?: (value: MCPServerCostInfo) => void;
  tools?: any[]; // Receive tools from connection component
  disabled?: boolean;
}

interface CostInputProps {
  value: number | null | undefined;
  placeholder: string;
  disabled?: boolean;
  className?: string;
  onChange: (cost: number | null) => void;
}

/**
 * Costs are shown to four decimal places when idle, but the field keeps the raw
 * keystrokes while it is being edited so partial input like "0." survives.
 */
const CostInput: React.FC<CostInputProps> = ({ value, placeholder, disabled, className, onChange }) => {
  const [draft, setDraft] = useState<string | null>(null);
  const display = draft ?? (value === null || value === undefined ? "" : value.toFixed(4));

  const handleChange = (next: string) => {
    setDraft(next);
    const parsed = Number(next);
    onChange(next.trim() === "" || Number.isNaN(parsed) ? null : parsed);
  };

  return (
    <InputGroup className={className}>
      <InputGroupAddon>
        <InputGroupText>$</InputGroupText>
      </InputGroupAddon>
      <InputGroupInput
        type="text"
        inputMode="decimal"
        placeholder={placeholder}
        disabled={disabled}
        value={display}
        onFocus={() => setDraft(value === null || value === undefined ? "" : String(value))}
        onBlur={() => setDraft(null)}
        onChange={(e) => handleChange(e.target.value)}
      />
    </InputGroup>
  );
};

const MCPServerCostConfig: React.FC<MCPServerCostConfigProps> = ({
  value = {},
  onChange,
  tools = [],
  disabled = false,
}) => {
  const handleDefaultCostChange = (defaultCost: number | null) => {
    const updated = {
      ...value,
      default_cost_per_query: defaultCost,
    };
    onChange?.(updated);
  };

  const handleToolCostChange = (toolName: string, cost: number | null) => {
    const updated = {
      ...value,
      tool_name_to_cost_per_query: {
        ...value.tool_name_to_cost_per_query,
        [toolName]: cost,
      },
    };
    onChange?.(updated);
  };

  return (
    <TooltipProvider>
      <Card className="p-6">
        <div className="space-y-6">
          <div className="mb-4 flex items-center gap-2">
            <DollarSign className="size-4 text-muted-foreground" />
            <h3 className="text-lg font-medium">Cost Configuration</h3>
            <Tooltip>
              <TooltipTrigger
                render={<Info className="size-4 text-muted-foreground" aria-label="About cost configuration" />}
              />
              <TooltipContent>
                Configure costs for this MCP server&apos;s tool calls. Set a default rate and per-tool overrides.
              </TooltipContent>
            </Tooltip>
          </div>

          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium">
                Default Cost per Query ($)
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <Info className="ml-1 inline size-4 text-muted-foreground" aria-label="About the default cost" />
                    }
                  />
                  <TooltipContent>Default cost charged for each tool call to this server.</TooltipContent>
                </Tooltip>
              </label>
              <CostInput
                value={value.default_cost_per_query}
                placeholder="0.0000"
                disabled={disabled}
                className="w-50"
                onChange={handleDefaultCostChange}
              />
              <p className="mt-1 block text-sm text-muted-foreground">
                Set a default cost for all tool calls to this server
              </p>
            </div>

            {tools.length > 0 && (
              <div className="space-y-4">
                <label className="block text-sm font-medium">
                  Tool-Specific Costs ($)
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <Info className="ml-1 inline size-4 text-muted-foreground" aria-label="About per-tool costs" />
                      }
                    />
                    <TooltipContent>
                      Override the default cost for specific tools. Leave blank to use the default rate.
                    </TooltipContent>
                  </Tooltip>
                </label>
                <Collapsible className="rounded-lg border border-border">
                  <CollapsibleTrigger
                    render={
                      <button type="button" className="flex w-full items-center gap-2 p-3 text-left">
                        <Wrench className="size-4 text-muted-foreground" />
                        <span className="font-medium">Available Tools</span>
                        <Badge variant="secondary">{tools.length}</Badge>
                      </button>
                    }
                  />
                  <CollapsibleContent>
                    <div className="max-h-64 space-y-3 overflow-y-auto p-3">
                      {tools.map((tool, index) => (
                        <div key={index} className="flex items-center justify-between rounded-lg bg-muted p-3">
                          <div className="flex-1">
                            <p className="text-sm font-medium">{tool.name}</p>
                            {tool.description && (
                              <p className="mt-1 block text-sm text-muted-foreground">{tool.description}</p>
                            )}
                          </div>
                          <div className="ml-4">
                            <CostInput
                              value={value.tool_name_to_cost_per_query?.[tool.name]}
                              placeholder="Use default"
                              disabled={disabled}
                              className="w-40"
                              onChange={(cost) => handleToolCostChange(tool.name, cost)}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            )}
          </div>

          {(value.default_cost_per_query ||
            (value.tool_name_to_cost_per_query && Object.keys(value.tool_name_to_cost_per_query).length > 0)) && (
            <div className="mt-6 rounded-lg border border-border bg-muted p-4">
              <p className="text-sm font-medium">Cost Summary:</p>
              <div className="mt-2 space-y-1">
                {value.default_cost_per_query && (
                  <p className="text-sm text-muted-foreground">
                    • Default cost: ${value.default_cost_per_query.toFixed(4)} per query
                  </p>
                )}
                {value.tool_name_to_cost_per_query &&
                  Object.entries(value.tool_name_to_cost_per_query).map(
                    ([toolName, cost]) =>
                      cost !== null &&
                      cost !== undefined && (
                        <p key={toolName} className="text-sm text-muted-foreground">
                          • {toolName}: ${cost.toFixed(4)} per query
                        </p>
                      ),
                  )}
              </div>
            </div>
          )}
        </div>
      </Card>
    </TooltipProvider>
  );
};

export default MCPServerCostConfig;
