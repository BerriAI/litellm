import React from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { DollarSign, Info, Wrench } from "lucide-react";
import { MCPServerCostInfo } from "./types";

interface MCPServerCostConfigProps {
  value?: MCPServerCostInfo;
  onChange?: (value: MCPServerCostInfo) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tools?: any[];
  disabled?: boolean;
}

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const MoneyInput: React.FC<{
  value?: number | null;
  onChange: (v: number | null) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}> = ({ value, onChange, placeholder, disabled, className }) => (
  <div className={`inline-flex items-center rounded-md border border-input overflow-hidden ${className ?? ""}`}>
    <span className="px-2 text-sm bg-muted text-muted-foreground">$</span>
    <Input
      type="number"
      min={0}
      step={0.0001}
      value={value ?? ""}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") {
          onChange(null);
          return;
        }
        const n = Number(raw);
        onChange(isNaN(n) ? null : n);
      }}
      placeholder={placeholder}
      disabled={disabled}
      className="border-0 rounded-none shadow-none"
    />
  </div>
);

const MCPServerCostConfig: React.FC<MCPServerCostConfigProps> = ({
  value = {},
  onChange,
  tools = [],
  disabled = false,
}) => {
  const handleDefaultCostChange = (defaultCost: number | null) => {
    onChange?.({
      ...value,
      default_cost_per_query: defaultCost,
    });
  };

  const handleToolCostChange = (toolName: string, cost: number | null) => {
    onChange?.({
      ...value,
      tool_name_to_cost_per_query: {
        ...value.tool_name_to_cost_per_query,
        [toolName]: cost,
      },
    });
  };

  return (
    <Card className="p-4">
      <div className="space-y-6">
        <div className="flex items-center gap-2 mb-4">
          <DollarSign className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <h3 className="text-lg font-semibold">Cost Configuration</h3>
          <InfoTip>
            Configure costs for this MCP server&apos;s tool calls. Set a
            default rate and per-tool overrides.
          </InfoTip>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Default Cost per Query ($)
              <InfoTip>
                Default cost charged for each tool call to this server.
              </InfoTip>
            </label>
            <MoneyInput
              value={value.default_cost_per_query}
              onChange={handleDefaultCostChange}
              placeholder="0.0000"
              disabled={disabled}
              className="w-[200px]"
            />
            <p className="mt-1 text-muted-foreground text-sm">
              Set a default cost for all tool calls to this server
            </p>
          </div>

          {tools.length > 0 && (
            <div className="space-y-4">
              <label className="block text-sm font-medium text-foreground">
                Tool-Specific Costs ($)
                <InfoTip>
                  Override the default cost for specific tools. Leave blank
                  to use the default rate.
                </InfoTip>
              </label>
              <Accordion type="single" collapsible>
                <AccordionItem value="tools">
                  <AccordionTrigger>
                    <div className="flex items-center gap-2">
                      <Wrench className="h-4 w-4 text-blue-500" />
                      <span className="font-medium">Available Tools</span>
                      <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                        {tools.length}
                      </Badge>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 max-h-64 overflow-y-auto">
                      {tools.map((tool, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 bg-muted rounded-lg"
                        >
                          <div className="flex-1">
                            <span className="font-medium text-foreground">
                              {tool.name}
                            </span>
                            {tool.description && (
                              <span className="text-muted-foreground text-sm block mt-1">
                                {tool.description}
                              </span>
                            )}
                          </div>
                          <div className="ml-4">
                            <MoneyInput
                              value={
                                value.tool_name_to_cost_per_query?.[tool.name]
                              }
                              onChange={(cost) =>
                                handleToolCostChange(tool.name, cost)
                              }
                              placeholder="Use default"
                              disabled={disabled}
                              className="w-[160px]"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          )}
        </div>

        {(value.default_cost_per_query ||
          (value.tool_name_to_cost_per_query &&
            Object.keys(value.tool_name_to_cost_per_query).length > 0)) && (
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg">
            <p className="text-blue-800 dark:text-blue-200 font-medium">
              Cost Summary:
            </p>
            <div className="mt-2 space-y-1">
              {value.default_cost_per_query && (
                <p className="text-blue-700 dark:text-blue-300">
                  • Default cost: $
                  {value.default_cost_per_query.toFixed(4)} per query
                </p>
              )}
              {value.tool_name_to_cost_per_query &&
                Object.entries(value.tool_name_to_cost_per_query).map(
                  ([toolName, cost]) =>
                    cost !== null &&
                    cost !== undefined && (
                      <p
                        key={toolName}
                        className="text-blue-700 dark:text-blue-300"
                      >
                        • {toolName}: ${cost.toFixed(4)} per query
                      </p>
                    ),
                )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPServerCostConfig;
