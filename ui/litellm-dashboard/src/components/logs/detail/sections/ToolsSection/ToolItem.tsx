/**
 * Individual tool item component with expandable details
 */

import { useState } from "react";
import { ChevronDown, ChevronRight, Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";
import { ParsedTool } from "./types";
import { ToolExpandedContent } from "./ToolExpandedContent";

interface ToolItemProps {
  tool: ParsedTool;
}

export function ToolItem({ tool }: ToolItemProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="overflow-hidden rounded-lg border border-border">
      {/* Header Row - Always Visible */}
      <div
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "flex cursor-pointer items-center justify-between gap-3 px-4 py-3 text-card-foreground transition-colors",
          expanded ? "bg-muted" : "bg-card",
        )}
      >
        <div className="flex items-center gap-2.5">
          <Wrench className="size-3.5 text-muted-foreground" />
          <span className="text-sm">
            {tool.index}. {tool.name}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant={tool.called ? "default" : "secondary"}>{tool.called ? "called" : "not called"}</Badge>
          {expanded ? (
            <ChevronDown className="size-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-3 text-muted-foreground" />
          )}
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="border-t border-border bg-card p-4 text-card-foreground">
          <ToolExpandedContent tool={tool} />
        </div>
      )}
    </div>
  );
}
