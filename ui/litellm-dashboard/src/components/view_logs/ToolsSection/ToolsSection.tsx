/**
 * Tools section component that displays all available tools from the request
 * and indicates which ones were actually called in the response
 */

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { LogEntry } from "../columns";
import { parseToolsFromLog } from "./utils";
import { ToolItem } from "./ToolItem";

interface ToolsSectionProps {
  log: LogEntry;
}

export function ToolsSection({ log }: ToolsSectionProps) {
  const [open, setOpen] = useState(false);
  const tools = parseToolsFromLog(log);

  // Don't render if no tools
  if (tools.length === 0) return null;

  // Calculate summary stats
  const totalTools = tools.length;
  const calledTools = tools.filter((t) => t.called).length;

  // Get preview of first 2 tool names
  const toolNamePreview = tools
    .slice(0, 2)
    .map((t) => t.name)
    .join(", ");
  const hasMoreTools = tools.length > 2;

  return (
    <div className="mb-6 w-full max-w-full overflow-hidden rounded-lg bg-background shadow-sm">
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-muted">
          {open ? (
            <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
          )}
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-lg font-medium text-foreground">Tools</h3>
            <span className="text-sm text-muted-foreground">
              {totalTools} provided, {calledTools} called
            </span>
            <span className="text-sm text-muted-foreground">
              • {toolNamePreview}
              {hasMoreTools && "..."}
            </span>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent keepMounted>
          <div className="flex flex-col gap-2 px-4 pb-4">
            {tools.map((tool) => (
              <ToolItem key={tool.name} tool={tool} />
            ))}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
