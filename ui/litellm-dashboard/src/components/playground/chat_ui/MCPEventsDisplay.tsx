import React from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Check } from "lucide-react";
import type { MCPEvent } from "../../mcp_tools/types";

interface MCPEventsDisplayProps {
  events: MCPEvent[];
  className?: string;
}

const MCPEventsDisplay: React.FC<MCPEventsDisplayProps> = ({
  events,
  className,
}) => {
  if (!events || events.length === 0) {
    return null;
  }

  // Find the list tools event
  const toolsEvent = events.find(
    (event) =>
      event.type === "response.output_item.done" &&
      event.item?.type === "mcp_list_tools" &&
      event.item.tools &&
      event.item.tools.length > 0,
  );

  // Find MCP call events
  const mcpCallEvents = events.filter(
    (event) =>
      event.type === "response.output_item.done" &&
      event.item?.type === "mcp_call",
  );

  if (!toolsEvent && mcpCallEvents.length === 0) {
    return null;
  }

  const defaultValue = toolsEvent
    ? ["list-tools"]
    : mcpCallEvents.map((_, index) => `mcp-call-${index}`);

  return (
    <div className={`mcp-events-display ${className || ""}`}>
      <div className="relative pl-5">
        {/* Vertical guide line */}
        <div className="absolute left-[9px] top-[18px] bottom-0 w-[0.5px] bg-border opacity-80" />

        <Accordion
          type="multiple"
          defaultValue={defaultValue}
          className="w-full"
        >
          {toolsEvent && (
            <AccordionItem value="list-tools" className="border-none">
              <AccordionTrigger className="py-1 text-sm text-muted-foreground hover:text-foreground hover:no-underline">
                List tools
              </AccordionTrigger>
              <AccordionContent className="pt-1">
                <div>
                  {toolsEvent.item?.tools?.map((tool, index) => (
                    <div
                      key={index}
                      className="font-mono text-sm text-foreground/80 bg-background relative z-[1] py-0"
                    >
                      {tool.name}
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          )}

          {mcpCallEvents.map((callEvent, index) => (
            <AccordionItem
              key={`mcp-call-${index}`}
              value={`mcp-call-${index}`}
              className="border-none"
            >
              <AccordionTrigger className="py-1 text-sm text-muted-foreground hover:text-foreground hover:no-underline">
                {callEvent.item?.name || "Tool call"}
              </AccordionTrigger>
              <AccordionContent className="pt-1">
                <div>
                  {/* Request section */}
                  <div className="mb-3 bg-background relative z-[1]">
                    <div className="text-sm text-muted-foreground font-medium mb-1">
                      Request
                    </div>
                    <div className="bg-muted border border-border rounded-md p-2 text-xs">
                      {callEvent.item?.arguments && (
                        <pre className="font-mono text-foreground m-0 whitespace-pre-wrap break-words">
                          {(() => {
                            try {
                              return JSON.stringify(
                                JSON.parse(callEvent.item.arguments),
                                null,
                                2,
                              );
                              // eslint-disable-next-line @typescript-eslint/no-unused-vars
                            } catch (_e) {
                              return callEvent.item.arguments;
                            }
                          })()}
                        </pre>
                      )}
                    </div>
                  </div>

                  {/* Approved section */}
                  <div className="mb-3 bg-background relative z-[1]">
                    <div className="flex items-center text-sm text-muted-foreground">
                      <Check className="h-4 w-4 mr-1.5 text-emerald-500" />
                      Approved
                    </div>
                  </div>

                  {/* Response section */}
                  {callEvent.item?.output && (
                    <div className="mb-0 bg-background relative z-[1]">
                      <div className="text-sm text-muted-foreground font-medium mb-1">
                        Response
                      </div>
                      <div className="text-sm text-foreground/80 leading-relaxed whitespace-pre-wrap font-mono">
                        {callEvent.item.output}
                      </div>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </div>
  );
};

export default MCPEventsDisplay;
