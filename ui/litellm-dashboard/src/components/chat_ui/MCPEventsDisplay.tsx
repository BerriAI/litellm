import React, { useState } from "react";
import { ChevronRight } from "lucide-react";
import type { MCPEvent } from "@/components/mcp_tools/types";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/cva.config";

interface MCPEventsDisplayProps {
  events: MCPEvent[];
  className?: string;
}

function formatArguments(raw: string | undefined): string {
  if (!raw) {
    return "";
  }
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

const MCPEventsDisplay: React.FC<MCPEventsDisplayProps> = ({ events, className }) => {
  if (!events || events.length === 0) {
    return null;
  }

  const isListToolsEvent = (event: MCPEvent): boolean => {
    if (event.type !== "response.output_item.done") {
      return false;
    }
    if (event.item?.type !== "mcp_list_tools") {
      return false;
    }
    return Boolean(event.item.tools && event.item.tools.length > 0);
  };

  const isMcpCallEvent = (event: MCPEvent): boolean =>
    event.type === "response.output_item.done" && event.item?.type === "mcp_call";

  const toolsEvent = events.find(isListToolsEvent);
  const mcpCallEvents = events.filter(isMcpCallEvent);

  if (!toolsEvent && mcpCallEvents.length === 0) {
    return null;
  }

  const defaultOpenKeys = new Set<string>(
    toolsEvent ? ["list-tools"] : mcpCallEvents.map((_, index) => `mcp-call-${index}`),
  );

  return (
    <div className={cn("mcp-events-display", className)}>
      <MCPEventsPanels toolsEvent={toolsEvent} mcpCallEvents={mcpCallEvents} defaultOpenKeys={defaultOpenKeys} />
    </div>
  );
};

interface MCPEventsPanelsProps {
  toolsEvent: MCPEvent | undefined;
  mcpCallEvents: MCPEvent[];
  defaultOpenKeys: Set<string>;
}

function MCPEventsPanels({ toolsEvent, mcpCallEvents, defaultOpenKeys }: MCPEventsPanelsProps) {
  const [openKeys, setOpenKeys] = useState<Set<string>>(defaultOpenKeys);

  const toggleKey = (key: string, open: boolean) => {
    setOpenKeys((prev) => {
      const next = new Set(prev);
      if (open) {
        next.add(key);
      } else {
        next.delete(key);
      }
      return next;
    });
  };

  return (
    <div className="relative m-0 p-0">
      <div className="absolute bottom-0 left-[9px] top-[18px] w-px bg-muted opacity-80" aria-hidden="true" />

      <div className="space-y-1">
        {toolsEvent && (
          <MCPEventPanel
            panelKey="list-tools"
            title="List tools"
            open={openKeys.has("list-tools")}
            onOpenChange={(open) => toggleKey("list-tools", open)}
          >
            <div>
              {toolsEvent.item?.tools?.map((tool, index) => (
                <div
                  key={index}
                  className="relative z-raised bg-card font-mono text-[13px] leading-[18px] text-muted-foreground"
                >
                  {tool.name}
                </div>
              ))}
            </div>
          </MCPEventPanel>
        )}

        {mcpCallEvents.map((callEvent, index) => {
          const key = `mcp-call-${index}`;
          return (
            <MCPEventPanel
              key={key}
              panelKey={key}
              title={callEvent.item?.name || "Tool call"}
              open={openKeys.has(key)}
              onOpenChange={(open) => toggleKey(key, open)}
            >
              <div>
                <div className="relative z-raised mb-3 bg-card last:mb-0">
                  <div className="mb-1 text-[13px] font-medium text-muted-foreground">Request</div>
                  <div className="rounded-md border border-border bg-muted p-2 text-xs">
                    {callEvent.item?.arguments && (
                      <pre className="m-0 whitespace-pre-wrap break-words font-mono text-foreground">
                        {formatArguments(callEvent.item.arguments)}
                      </pre>
                    )}
                  </div>
                </div>

                <div className="relative z-raised mb-3 bg-card last:mb-0">
                  <div className="flex items-center text-[13px] text-muted-foreground">
                    <span className="mr-1.5 font-bold text-success" aria-hidden="true">
                      ✓
                    </span>
                    Approved
                  </div>
                </div>

                {callEvent.item?.output && (
                  <div className="relative z-raised mb-3 bg-card last:mb-0">
                    <div className="mb-1 text-[13px] font-medium text-muted-foreground">Response</div>
                    <div className="whitespace-pre-wrap font-mono text-[13px] leading-normal text-foreground">
                      {callEvent.item.output}
                    </div>
                  </div>
                )}
              </div>
            </MCPEventPanel>
          );
        })}
      </div>
    </div>
  );
}

interface MCPEventPanelProps {
  panelKey: string;
  title: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
}

function MCPEventPanel({ title, open, onOpenChange, children }: MCPEventPanelProps) {
  return (
    <Collapsible open={open} onOpenChange={onOpenChange}>
      <CollapsibleTrigger className="relative flex min-h-5 w-full items-center gap-1 pl-5 text-left text-sm font-normal leading-5 text-muted-foreground hover:text-foreground">
        <ChevronRight
          className={cn(
            "absolute left-0.5 top-0.5 size-4 text-muted-foreground transition-transform",
            open && "rotate-90",
          )}
          aria-hidden="true"
        />
        {title}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="pt-1 pl-5">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export default MCPEventsDisplay;
